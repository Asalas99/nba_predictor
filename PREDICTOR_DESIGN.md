# Diseño del predictor — borrador para revisar

Objetivo: predecir la temporada en cascada (**wins → seeding → playoffs →
campeón**) actualizando el plantel de cada equipo **antes** de que la temporada
empiece, incluso para jugadores que aún no tienen estadísticas del año, y
contemplando el efecto del entrenador.

Este documento es para acordar el enfoque; lo modificamos sobre la marcha.

---

## 1. Principio que manda todo: nada de fuga temporal

Para *predecir* la temporada T solo se puede usar información disponible **antes**
de que arranque: historia de los jugadores hasta T-1, quién está en cada plantel
para T, y quién es el entrenador. Métricas del propio año T (net rating, PIE de
esa temporada, victorias) **no** se pueden usar como entrada — solo sirven como
lo que queremos predecir o para validar.

Esto es justo lo que aprendimos en el EDA: `net_rating` correlaciona 0.95 con
wins pero es del mismo año, así que es inútil para pronosticar. La `squad_strength`
que ya construimos correlaciona 0.76, pero ojo: la calculamos con stats del año T.
El predictor necesita una versión **proyectada** de esa fuerza a partir del pasado.

---

## 2. Arquitectura (5 piezas encadenadas)

```
(A) Roster de T          quién juega en cada equipo la temporada T (altas/bajas)
        |
(B) Proyección jugador   para cada jugador, estimar su PIE y su rol en T
        |                 desde su HISTORIA (hasta T-1) + curva de edad
        v
(C) Agregación a equipo  squad_strength_proj(T), composición de roster proyectada,
        |                 continuidad respecto a T-1
(D) Efecto entrenador    ajuste por el sistema y el rendimiento del técnico
        |
        v
(E) Modelo M1            features proyectadas -> victorias esperadas de T
```

De ahí sigue la cascada (seeding → playoffs → campeón) que ya está en el PLAN.

---

## 3. Detalle de cada pieza

### (A) Roster de destino
Quién compone cada equipo en T. En **backtesting** esto se conoce: `player_clean`
ya dice en qué equipo estuvo cada jugador cada temporada, así que podemos
reconstruir el plantel real de cada año pasado y validar sin problema. Para una
**predicción viva** de una temporada futura (p. ej. 2026-27) haría falta la lista
de rosters/movimientos nueva — ese es el único dato que no se deriva solo (ver §5).

### (B) Proyección de cada jugador (el corazón de tu idea)
Para cada jugador y temporada T, estimar qué va a aportar usando solo su pasado:

- **PIE proyectado**: promedio de sus PIE previos ponderado por recencia
  (el año más reciente pesa más) y por minutos, **ajustado por curva de edad**
  (los jugadores mejoran hasta ~26-27 y declinan después) y con **regresión a la
  media** para muestras chicas o jóvenes con poca historia.
- **Minutos proyectados**: cuántos minutos jugará en T. Empezamos simple (sus
  minutos de T-1 con tope y ajuste por edad); se puede refinar por profundidad del
  equipo.
- **Vector de rol proyectado**: su arquetipo tiende a persistir; se proyecta con
  su composición histórica.
- **Sin historia (rookies o llegadas sin datos NBA)**: se usa un *prior* por edad
  y arquetipo esperado (línea base). Esto cubre exactamente tu caso de "el jugador
  no tiene stats de esta temporada": se predice lo que puede ser desde su historia,
  y si no hay historia, desde un prior razonable.

Validación propia de esta pieza: ¿el PIE proyectado se parece al PIE real que el
jugador tuvo en T? (error de proyección por jugador).

### (C) Agregación a nivel equipo
Con los jugadores del roster de T y su proyección:
- `squad_strength_proj(T)` = PIE proyectado ponderado por minutos proyectados
  (misma fórmula que ya validamos, pero con valores proyectados).
- Composición de roster proyectada → tipo de plantel esperado.
- `continuidad(T)` = % de minutos proyectados que vienen de jugadores que ya
  estaban en el equipo en T-1 (mide cuánto cambió el plantel).

### (D) Efecto entrenador
Un técnico impone su forma de jugar y suele rendir por encima o por debajo de lo
que dice el talento. Lo modelamos en dos partes:
- **Empuje de estilo**: el estilo esperado del equipo se corre hacia el estilo
  histórico característico del entrenador (su promedio de pace, 3PA, defensa…).
- **Efecto de rendimiento**: cuántas victorias suele sacar el técnico *por encima
  de lo que predice el talento* (su residual histórico wins − wins_esperadas).
  Entrenador nuevo/sin historia → prior neutro.

Arranca como un término de ajuste por entrenador en M1, estimado de su historial,
y validamos si **añadirlo baja el error**. Requiere el dato de qué técnico dirigió
cada equipo cada temporada (§5).

### (E) Modelo M1 (victorias)
Entrada: `squad_strength_proj`, tipo de plantel proyectado, continuidad, efecto
entrenador, wins de T-1. Salida: victorias esperadas de T (con incertidumbre).
Modelo regularizado (Ridge/ElasticNet o Gradient Boosting chico), entrenado
**walk-forward**: se entrena con temporadas ≤ T-1 y se predice T.

---

## 4. Qué se puede construir YA vs. qué necesita datos nuevos

| Pieza | ¿Construible ahora con lo que hay? |
|---|---|
| (A) Roster histórico para backtest | Sí — de `player_clean` |
| (B) Proyección de jugador desde historia + edad | Sí |
| (C) squad_strength proyectada + continuidad | Sí |
| (E) M1 wins + backtesting walk-forward | Sí |
| (D) Efecto entrenador | **Falta dato**: técnico por equipo/temporada |
| Predicción viva de una temporada futura | **Falta dato**: rosters/altas-bajas nuevas |

Traducción: podemos construir y validar el predictor completo (A, B, C, E) con lo
que ya tienes, y meter el entrenador (D) en cuanto consigamos la tabla de técnicos.

---

## 5. Datos que faltan (para decidir)

1. **Entrenador por equipo y temporada** (2019-20 → 2025-26), idealmente con fecha
   de inicio para detectar cambios a mitad de año. Opciones: bajarlo vía `nba_api`
   (hay endpoints de coaches), scrapear una tabla, o armarlo a mano para los casos
   relevantes. Lo defino cuando lleguemos a la pieza (D).
2. **Rosters de la temporada futura** (solo si quieres predicción viva de 2026-27
   antes de que empiece). Para backtesting no hace falta.

---

## 6. Validación

Walk-forward por temporada, comparando contra baselines que hay que superar:
victorias del año pasado, y squad_strength proyectada sola. Métrica principal:
MAE en victorias (meta competitiva < 6). Cada pieza nueva (proyección, entrenador)
se acepta solo si **mejora** la métrica en backtest.

---

## 7. Roadmap incremental (para ir modificando en el camino)

- **Fase A — Proyección de jugador.** Módulo que, para cada jugador-temporada,
  proyecta PIE y minutos desde su historia + edad. Entregable: tabla de
  proyecciones + error de proyección vs. real.
- **Fase B — Fuerza proyectada del equipo.** Agregar a nivel equipo y comprobar,
  en backtest sin fuga, que `squad_strength_proj(T)` predice wins de T.
- **Fase C — M1 wins.** Modelo con features proyectadas + continuidad,
  walk-forward, contra baselines.
- **Fase D — Entrenador.** Conseguir datos de técnicos e incorporar su efecto;
  validar mejora.
- **Fase E — Cascada.** Seeding → playoffs → campeón sobre las wins proyectadas.

Propongo **empezar por la Fase A** (proyección de jugador), que es 100%
construible ahora y es la base de todo lo demás.

---

## 8. Preguntas abiertas para ti

- Curva de edad: ¿la estimamos de tus propios datos (cómo cambia el PIE con la
  edad en 2019-2026) o usamos una curva estándar? (recomiendo estimarla de los datos)
- Entrenador: ¿lo quieres desde la Fase A o lo dejamos para la Fase D como está
  planteado?
- ¿Te interesa además la predicción viva de 2026-27 (implica conseguir rosters
  nuevos), o por ahora nos enfocamos en que el modelo prediga bien en backtest?
