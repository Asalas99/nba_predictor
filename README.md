# nba_predictor

Sistema **self-contained y reproducible** que predice la temporada de la NBA en
cascada: **victorias → seeding → playoffs → campeón**. Tiene su propia ingesta
de datos, limpieza, features y modelos; todo corre con un solo comando.

Combina dos ideas que un rating simple no captura: el **estilo y la construcción
de plantel** (clustering) y la **fuerza real del núcleo** medida por el impacto
de los jugadores (PIE), proyectada desde su historia sin mirar el año a predecir.

---

## Puesta en marcha (3 pasos)

```bash
# 1. entorno
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. descargar los datos crudos (nba_api, necesita internet) — NO vienen en el repo
python -m src.ingest.download_teams   --start 2019 --end 2025
python -m src.ingest.download_players --start 2019 --end 2025
python -m src.ingest.download_lineups --start 2019 --end 2025
python -m src.ingest.download_coaches --start 2019 --end 2025
python -m src.ingest.download_gamelogs --start 2021 --end 2025   # opcional: partido a partido

# 3. correr TODO el pipeline
python run_all.py

# 4. ver resultados
#   outputs/figures/{clustering,correlacion,fuerza,predicciones}/  -> gráficas
#   outputs/tables/{...}/   -> tablas
#   outputs/*.pdf           -> documentación y resumen técnico
```

Los **datos NO se versionan** (se regeneran con los descargadores de arriba, que
son las mismas fuentes oficiales de la NBA). El repo contiene solo el código, la
documentación y los PDFs. Tras descargar, `run_all.py` genera todo en ~20 s.

---

## Actualizar con datos nuevos (cada temporada)

La descarga usa `nba_api` y **necesita internet** (corre en tu máquina). Baja la
temporada nueva y reprocesa:

```bash
python -m src.ingest.download_teams   --start 2026 --end 2026
python -m src.ingest.download_players --start 2026 --end 2026
python -m src.ingest.download_lineups --start 2026 --end 2026
python -m src.ingest.download_coaches --start 2026 --end 2026
python run_all.py
```

Y añade el campeón de la temporada que cerró en `src/data/champions.py`.

### Opcional: probar M1 sobre el calendario (game by game)

Descarga los resultados por partido y ejecuta el backtest de calendario
(probabilidad de victoria por partido, sin sesgo: la fuerza sale de M1, que solo
usa temporadas anteriores):

```bash
python -m src.ingest.download_gamelogs --start 2021 --end 2025   # en tu máquina
python -m src.models.m1_schedule
```

Produce victorias esperadas vs reales, el acierto partido a partido y un
calendario coloreado por probabilidad de victoria. `run_all.py` lo corre solo si
los game logs ya están descargados (si no, lo omite).

### Preparar la temporada por empezar (2026-27)

Para predecir una temporada que aún no arranca (no tiene estadísticas), se bajan
rosters, entrenadores y el calendario ya definido:

```bash
python -m src.ingest.download_rosters  --start 2026 --end 2026   # plantillas + coach
python -m src.ingest.download_schedule --season 2026             # calendario 2026-27
```

Ojo con el año: la temporada 2026-27 se pide con `2026` (año de inicio). Los
scripts avisan qué temporada bajan y omiten las que aún no tienen datos.

---

## Qué hace, paso a paso (lo que corre `run_all.py`)

| # | Módulo | Qué produce |
|---|---|---|
| 1 | `features.clean_teams` | stats de equipo limpias + escaladas (8 features) |
| 2 | `features.clean_players` | stats de jugador limpias |
| 3 | `models.style_clustering` | clusters de **estilo de juego** (KMeans k=3) |
| 4 | `features.squad_strength` | **fuerza del plantel** (PIE ponderado por minutos) |
| 5 | `features.player_roles` | **6 arquetipos** de jugador |
| 6 | `models.roster_type_clustering` | **tipo de plantel** (construcción de roster) |
| 7 | `ingest.build_dataset` | dataset unificado equipo×temporada |
| 8 | `features.player_projection` | proyección de PIE/minutos por jugador (sin fuga) |
| 9 | `features.team_projection` | **fuerza proyectada** + continuidad de plantilla |
| 10 | `features.coach_features` | huella de estilo y residual del entrenador |
| 11 | `models.m1_wins` | **M1** — predicción de victorias + backtest |
| 12 | `models.m2_seeding` | **M2** — seeding por conferencia |
| 13 | `models.m3_playoffs` | **M3** — simulación Monte Carlo de playoffs |
| 14 | `models.m4_champion` | **M4** — probabilidad de campeón |
| 15-18 | `viz.*` | gráficas de estilo, fuerza, proximidad, arquetipos |
| 19-20 | `report.*` | PDFs de documentación y resumen técnico |

## Los modelos (cascada)

- **M1 (victorias)** — regresión Ridge sobre fuerza proyectada + continuidad +
  victorias previas + entrenador. MAE 7.45 (mejor que la persistencia, 8.89).
- **M2 (seeding)** — ordena las victorias por conferencia. Correlación 0.67.
- **M3 (playoffs)** — bracket Monte Carlo (log5 + ventaja de local). Brier 0.059.
- **M4 (campeón)** — calibración; el backtest indica que la mejor estimación es M3.

Todo se valida con **backtesting walk-forward** (entrenar con temporadas pasadas,
predecir la siguiente) y sin fuga temporal.

## Estructura

```
run_all.py                 corre todo el pipeline de procesamiento
config.py                  rutas, features, semillas
src/
  ingest/                  descarga (nba_api) + dataset unificado
  features/                limpieza, fuerza, proyección, arquetipos, entrenador
  models/                  style/roster clustering + M1, M2, M3, M4
  viz/                     gráficas
  report/                  generadores de PDF
  data/                    campeones y conferencias (tablas fijas)
data/raw/                  datos crudos (incluidos)
data/processed/            generado por el pipeline
outputs/{figures,tables}/  generado por el pipeline
PLAN.md, PREDICTOR_DESIGN.md   diseño del proyecto
```

## Documentos

- `outputs/Documentacion_nba_predictor.pdf` — explicación paso a paso (nivel
  sencillo + técnico), con gráficas.
- `outputs/Resumen_tecnico_modelos.pdf` — por cada modelo: parámetros, matemática,
  cómputo y el porqué.

## Requisitos

Python 3.9+. Dependencias en `requirements.txt` (pandas, numpy, scikit-learn,
matplotlib, joblib, nba_api, reportlab, pillow).
