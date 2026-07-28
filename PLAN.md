# Plan maestro — `nba_predictor`

Modelo robusto que predice cada temporada de la NBA en cascada
(**wins → seeding → playoffs → campeón**), combinando lo que ya construiste:
clustering de estilo y de construcción de roster (`nba_clustering_comp`) + medición
de la fuerza real del núcleo con APM/GNN limpia de tanking (`nba_tanking`).

Decisiones ya tomadas:
- **Target:** cascada completa (proyectar victorias → armar seeds → simular playoffs → sacar campeón).
- **Arquitectura:** repo nuevo `nba_predictor` que *orquesta* los dos proyectos existentes sin tocarlos.
- **Validación:** backtesting histórico (walk-forward) + predicción viva de 2025-26.

---

## 1. La idea central

Un récord y un rating de equipo dicen *qué tan bueno se ve* un equipo. Tus dos
proyectos aportan dos señales que el mercado de modelos suele ignorar:

1. **Estilo y construcción** (clustering): no todos los equipos buenos ganan
   igual. Ciertos estilos y ciertas plantillas (roles) tienen tasa de campeonato
   más alta. → *señal de "forma de ganar"*.
2. **Fuerza real del núcleo** (APM ridge + GNN anti-tanking): cuánto valen de
   verdad los 5-8 jugadores clave cuando el equipo sí compite, sin ruido de
   tanking ni de lesiones puntuales. → *señal de "techo real"*.

La hipótesis del proyecto: **un equipo predice mejor su resultado de temporada
cuando combinas cuánto talento real tiene (APM) con qué tan parecido es su
estilo/plantilla a los que históricamente ganan (proximidad a campeones)**. Eso
es lo que ningún Elo o rating simple captura.

---

## 2. Qué se reutiliza de cada proyecto (inventario)

### De `nba_clustering_comp` — SOLO lo verificado al 100%
> Auditoría del repo (jul 2026): únicamente el pipeline de 5 comandos del README
> y su dependencia `champions_finalists.py` están validados. Todo lo demás
> (roles, proximidad, pipelines de "finalists") se trata como **no confiable** y
> **no** se reutiliza como código: se reimplementa limpio en `nba_predictor`.

| Activo (CONFIABLE) | Qué aporta al predictor |
|---|---|
| `combined/team_style_clusters.csv` (2019-20 → 2025-26) | Estilo por equipo/temporada + cluster + nombre (8 features escaladas) |
| `combined/team_stats_clean.csv` | **W, L, W_PCT, NET_RATING sin escalar** → label de victorias ya servido |
| `combined/kmeans_model.pkl` + `by_year/*/scaler.pkl` | Modelos entrenados, reusables para proyectar equipos nuevos |
| `src/data/champions_finalists.py` | **Etiquetas** campeón/finalista/CF 2019-20 → 2024-25 (dependencia sana del clustering) |

**Descartado tras la auditoría (NO reutilizar):**
- Clustering de **roles / construcción de roster** (`cluster_roles/*.csv`, `roster_cleaning.py`,
  `team_clustering.py`, etc.): CSVs huérfanos, sin datos crudos de rosters/players
  descargados para todos los equipos. → La "construcción de plantilla" se obtiene
  mejor de los `players.csv`/`lineups.csv` de **nba_tanking** (traen APM por jugador).
- `championship_proximity_analysis.py`: no está en el pipeline validado. La
  proximidad a campeón se **reimplementa limpia** en `nba_predictor` usando
  `kmeans_model.pkl` + `champions_finalists.py`.
- Pipelines viejos de "finalists" (`load_data.py`, `finalists.py`, `player_stats.py`,
  `rosters.py`, `team_stats.py`): no se tocan.

### De `nba_tanking`
| Activo | Qué aporta al predictor |
|---|---|
| `part2_classification.csv` | `true_strength` (fuerza real APM), `tank_prob`, categoría por equipo |
| `common/utils.py` (APM ridge) | Motor para estimar valor individual de jugadores → fuerza de núcleo |
| `lineups.csv` / `standings.csv` (2014-15 → 2025-26) | Insumo crudo para recomputar APM en cualquier temporada |
| GNN anti-tanking | Bandera para **descontar** récords contaminados al construir el target |

### Dato clave de cobertura
- APM/tanking: **2014-15 → 2025-26** (12 temporadas).
- Clustering: **2019-20 → 2025-26** (7 temporadas).
- **Ventana con ambas señales: 2019-20 → 2025-26 (7 temporadas).**

Implicación para el diseño: el set de entrenamiento "rico" (estilo + APM) es de 7
temporadas. Es suficiente para un modelo con pocas features bien elegidas, no para
una red profunda. El plan lo asume y prioriza **modelos interpretables y
regularizados** sobre modelos con muchos parámetros.

---

## 3. Arquitectura del repo nuevo

```
nba_predictor/
├── config.py                  # temporadas, rutas a los dos repos, semillas
├── run_all.py                 # pipeline completo end-to-end
├── src/
│   ├── ingest/
│   │   ├── from_clustering.py  # lee outputs de nba_clustering_comp
│   │   ├── from_tanking.py     # lee outputs de nba_tanking (o recomputa APM)
│   │   └── build_dataset.py    # une todo en una tabla equipo×temporada
│   ├── features/
│   │   ├── team_features.py     # estilo, roles, APM núcleo, proximidad campeón
│   │   └── labels.py            # wins reales, seed real, ronda playoff, campeón
│   ├── models/
│   │   ├── m1_wins.py           # regresión de victorias regulares
│   │   ├── m2_seeding.py        # wins → ranking por conferencia
│   │   ├── m3_playoffs.py       # simulador Monte Carlo de bracket
│   │   └── m4_champion.py       # prob. de título (ensamble)
│   ├── eval/
│   │   ├── backtest.py          # walk-forward por temporada
│   │   └── metrics.py           # MAE wins, Spearman seeds, Brier/log-loss campeón
│   └── report/
│       └── make_report.py       # figuras + tablas + predicción 2025-26
├── data/                       # dataset unificado (generado)
└── outputs/{figures,tables}
```

Principio: `nba_predictor` **consume** archivos CSV/pkl de los otros dos. No los
modifica. Si un output no existe, el script de ingest te dice qué comando correr
en el repo origen. Esto mantiene los tres proyectos independientes y evita
romper lo que ya funciona.

---

## 4. El dataset unificado (una fila = equipo × temporada)

La tabla que alimenta todo. Columnas por bloque:

**Identidad:** `season`, `team_id`, `team_abbr`.

**Bloque estilo (clustering):** `style_cluster`, `off_rating`, `def_rating`,
`net_rating`, `pace`, `ts_pct`, `ast_pct`, … (ya escaladas).

**Bloque construcción (roles):** conteo de roles del núcleo, `role_cluster`,
balance del roster (¿tiene creación primaria + spacing + defensa interior?).

**Bloque talento real (APM/tanking):** `true_strength` (top-5 APM),
`core_apm_sum`, `depth_apm` (jugadores 6-8), `tank_prob`, `star_power` (mejor APM
individual), `age_núcleo`.

**Bloque proximidad a campeón:** `dist_champion` (distancia PCA al centroide de
campeones históricos), `dist_finalist`, `champ_similarity_score`.

**Contexto:** `prev_season_wins`, `roster_continuity` (% minutos retenidos de la
temporada previa), `salary_tier` agregada.

**Labels (target):** `wins`, `conf_seed`, `playoff_round_reached`
(0=fuera, 1=R1, …, 5=campeón), `is_champion`.

> El bloque APM y el de roles son lo que diferencia este modelo de un predictor
> genérico. Sin ellos, `net_rating` de la temporada previa ya predice bastante;
> el valor científico está en demostrar **cuánta señal incremental** aportan
> talento-real-limpio-de-tanking y similitud-de-construcción.

---

## 5. El modelo en cascada (los 4 eslabones)

**M1 — Proyección de victorias.** Regresión regularizada (Ridge/ElasticNet, o
Gradient Boosting con pocos árboles) `features → wins`. Entrada: APM del núcleo +
continuidad + estilo. Salida: victorias esperadas ± incertidumbre. Métrica: MAE
en victorias (objetivo competitivo: < 6 wins, referencia de mercado ≈ 5-6).

**M2 — Seeding.** Ordena por victorias proyectadas dentro de cada conferencia →
seeds 1-15. Puede añadir un ajuste por fuerza de calendario. Métrica: correlación
de Spearman entre seed proyectado y real, y acierto de play-in (7-10).

**M3 — Simulación de playoffs.** Monte Carlo del bracket: cada serie se resuelve
con un modelo de probabilidad de serie que usa la diferencia de `true_strength`
**y** un término de estilo (matchups: ciertos estilos rinden distinto en
playoffs que en regular season — aquí el clustering brilla). Miles de
simulaciones → distribución de resultados. Métrica: log-loss de avance por ronda.

**M4 — Probabilidad de campeón.** Sale natural de M3 (frecuencia de títulos en la
simulación), pero se **calibra/ensambla** con la señal de proximidad a campeón del
clustering (un equipo que se *parece* a campeones históricos recibe un ajuste
bayesiano). Métrica: Brier score y log-loss sobre el campeón real.

Cada eslabón se puede validar por separado, y la incertidumbre se propaga hacia
adelante (M1 da distribuciones, no puntos, para que M3 sea honesto).

---

## 6. Validación (el corazón del rigor)

**Backtesting walk-forward.** Entrenar solo con temporadas pasadas y predecir la
siguiente, avanzando en el tiempo:

| Entrena con | Predice |
|---|---|
| 2019-20 … 2021-22 | 2022-23 |
| 2019-20 … 2022-23 | 2023-24 |
| 2019-20 … 2023-24 | 2024-25 |

Nunca se usa información del futuro (sin fuga temporal). Para el bloque APM, que
existe desde 2014-15, se puede pre-entrenar M1 con más historia aunque falte el
estilo, y medir cuánto ayuda cada bloque.

**Baselines contra los que hay que ganar** (si no le ganas a estos, el modelo no
aporta):
1. Victorias del año pasado (persistencia).
2. `net_rating` previo → wins.
3. Un Elo/rating simple.
El resultado que vende el proyecto es: *"añadir APM-real + similitud-de-campeón
baja el MAE de X a Y y mejora el Brier del campeón de A a B"*.

**Predicción viva 2025-26.** Con el modelo ya validado, generar la predicción de
la temporada en curso: wins proyectados, seeds, probabilidades de playoffs y de
título por equipo. Es la demo y, al cerrar la temporada, la prueba real.

**Ablations** (para el reporte): quitar el bloque APM, quitar el de estilo, quitar
proximidad-a-campeón, y mostrar la caída de métricas de cada uno. Esto responde
"¿de verdad sirve cada pieza?".

---

## 7. Roadmap por fases

**Fase 0 — Andamiaje (0.5 día).** Crear el repo, `config.py` con rutas a los dos
proyectos, verificar que se leen sus outputs. Entregable: `build_dataset.py`
produce la tabla unificada de 7 temporadas.

**Fase 1 — Dataset + EDA (1-2 días).** Unir bloques, resolver mapeos de team_id
entre proyectos, imputar faltantes, y un EDA que ya conteste: ¿`true_strength`
correlaciona con wins? ¿`dist_champion` separa campeones? Entregable: dataset
validado + notebook de correlaciones.

**Fase 2 — M1 wins + backtest (2 días).** Modelo de victorias con walk-forward y
comparación contra baselines. Entregable: tabla de MAE por temporada y por
configuración de features. **Es el hito que decide si el enfoque funciona.**

**Fase 3 — M2/M3/M4 cascada (2-3 días).** Seeding, simulador de playoffs y
probabilidad de campeón, con el modelo de serie que usa APM + estilo.
Entregable: bracket simulado + probabilidades calibradas.

**Fase 4 — Ablations + reporte (1-2 días).** Ablations, figuras, y `REPORTE.md`
al estilo de los que ya escribes. Entregable: reporte reproducible.

**Fase 5 — Predicción viva 2025-26 (0.5 día).** Correr todo sobre la temporada
actual y publicar la predicción. Entregable: tabla + figura de contención por
equipo.

---

## 8. Riesgos y cómo mitigarlos

- **Pocas temporadas con ambas señales (7).** → Modelos simples y regularizados,
  validación estricta walk-forward, y extender M1 con historia APM 2014-19.
- **Mapeo de identidades de equipo** entre los dos proyectos (uno usa índices
  0-29, otro `TEAM_ID` de la NBA). → Tabla de cruce explícita en `ingest/`, es lo
  primero a resolver en Fase 1.
- **APM recomputado vs. cacheado.** Reusar `part2_classification.csv` si la
  temporada ya está; solo recomputar cuando falte, para no rehacer 15 min de
  cómputo cada corrida.
- **Sobreajuste al campeón** (solo ~7 campeones etiquetados). → No optimizar M4
  directamente sobre "campeón"; derivarlo de M3 y calibrar suavemente. Reportar
  con honestidad la incertidumbre.
- **Fuga temporal accidental.** → Un test automático que verifica que ninguna
  feature de la temporada t usa datos de t o posteriores.

---

## 9. Primer paso concreto (para arrancar cuando decidas)

Lo mínimo para desbloquear todo lo demás y validar la hipótesis rápido:

1. Crear `nba_predictor/` con `config.py` apuntando a los dos repos.
2. Escribir `build_dataset.py` que produzca la tabla equipo×temporada uniendo
   `team_style_clusters.csv` + `part2_classification.csv` + labels de wins.
3. Un EDA de una página: correlación de `true_strength` y `dist_champion` contra
   `wins` y contra `is_champion`.

Si esas correlaciones existen, el proyecto tiene piso y seguimos a M1. Si no,
ajustamos las features antes de invertir en la cascada completa.
