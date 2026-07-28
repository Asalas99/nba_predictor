# nba_predictor

Pipeline **self-contained y actualizable año con año** para predecir la NBA en
cascada (**wins → seeding → playoffs → campeón**). Tiene su propia ingesta de
datos (equipos + jugadores + lineups), limpieza y modelos; no depende de otros
repos en tiempo de ejecución.

Combina dos señales que un rating simple no captura:

1. **Estilo de juego** — clustering KMeans sobre stats avanzadas de equipo.
2. **Fuerza real del núcleo** — APM/RAPM desde lineups *(en construcción: el
   `true_strength` heredado de nba_tanking no correlaciona con victorias en
   datos reales; se reemplazará por un estimador propio — ver `PLAN.md`)*.

## Flujo de datos

```
data/raw/teams/{season}/team_stats_raw.csv     <- download_teams   (nba_api)
data/raw/players/{season}/player_stats_raw.csv <- download_players (nba_api)
data/raw/lineups/{lineups,standings,players}.csv <- download_lineups (nba_api)
        |
        v  (limpieza)
data/processed/teams/combined/team_clean.csv + team_cluster_input.csv
data/processed/players/combined/player_clean.csv
        |
        v  (modelos)
data/processed/teams/combined/team_style_clusters.csv + kmeans_model.pkl
        |
        v  (union)
data/unified_team_season.csv        (una fila = equipo x temporada)
```

## Uso

### 1. Descargar crudos — EN TU MÁQUINA (necesita internet; el sandbox no)

```bash
pip install -r requirements.txt
python -m src.ingest.download_teams   --start 2019 --end 2025
python -m src.ingest.download_players --start 2019 --end 2025
python -m src.ingest.download_lineups --start 2019 --end 2025
```

> En el sandbox de Cowork `stats.nba.com` está bloqueado. Para construir/probar
> sin descargar, siembra con los datos reales de tus otros repos:
> `python -m src.ingest.seed_from_repos`

### 2. Procesar (limpieza + modelos + dataset)

```bash
python run_all.py
```

### 3. Actualizar una sola temporada nueva (cada año)

```bash
# baja solo el año nuevo y reprocesa todo
python -m src.ingest.download_teams   --start 2026 --end 2026
python -m src.ingest.download_players --start 2026 --end 2026
python -m src.ingest.download_lineups --start 2026 --end 2026
python run_all.py
# y añade el campeón del año cerrado en src/data/champions.py
```

## Estructura

```
config.py                       rutas, features de estilo, k, semilla
run_all.py                      orquesta limpieza -> modelos -> dataset
src/
  ingest/
    nba_client.py               helpers nba_api (reintentos, mapa de equipos)
    download_teams.py           stats avanzadas de equipo
    download_players.py         stats de jugador (base + avanzadas)
    download_lineups.py         quintetos 5-man + standings + rosters
    seed_from_repos.py          siembra data/raw desde tus otros repos (sandbox)
    build_dataset.py            une estilo + wins + true_strength
  features/
    clean_teams.py              limpieza + escalado (8 features)
    clean_players.py            limpieza de jugadores
  models/
    style_clustering.py         KMeans k=3 + nombres + campeones + PCA
  data/champions.py             campeones/finalistas por temporada
data/{raw,processed}/           datos
outputs/{figures,tables}/       resultados
```

## Estado

- [x] Ingesta propia (teams / players / lineups) lista para tu máquina
- [x] Siembra desde datos reales existentes (sandbox)
- [x] Limpieza de equipos (reproduce el pipeline validado) + jugadores
- [x] Clustering de estilo self-contained (k=3)
- [x] Dataset unificado (210 equipos-temporada, 2019-20 → 2025-26)
- [ ] APM/RAPM propio para la fuerza real del núcleo *(el heredado falló — decidir método)*
- [ ] M1 wins + backtest
- [ ] M2/M3/M4 cascada (seeding, playoffs, campeón)

Ver `PLAN.md` para el plan maestro completo.
