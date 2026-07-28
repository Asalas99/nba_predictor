"""
Configuración global de nba_predictor.

Rutas a los dos proyectos que este repo orquesta. Ajusta CLUSTERING_REPO y
TANKING_REPO si mueves las carpetas.
"""

import os

# --- Rutas a los repos de origen ---------------------------------------------
# Por defecto asume que los tres repos son carpetas hermanas bajo el mismo padre.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

CLUSTERING_REPO = os.path.join(PARENT, "nba_clustering_comp")
TANKING_REPO = os.path.join(PARENT, "nba_tanking")

# --- Outputs PROPIOS (self-contained, generados por este repo) ----------------
# Se llenan al correr: clean_teams -> style_clustering
STYLE_CLUSTERS = os.path.join(
    HERE, "data/processed/teams/combined/team_style_clusters.csv")
TEAM_STATS_CLEAN = os.path.join(
    HERE, "data/processed/teams/combined/team_clean.csv")
KMEANS_MODEL = os.path.join(
    HERE, "data/processed/teams/combined/kmeans_model.pkl")

# --- Externo (provisional) ----------------------------------------------------
# Fuerza real del nucleo (APM) del repo nba_tanking. OJO: en datos reales este
# true_strength NO correlaciona con victorias (ver EDA). Se reemplazara por un
# APM/RAPM propio; se mantiene como referencia opcional mientras tanto.
TANKING_CLASSIFICATION = os.path.join(
    TANKING_REPO, "outputs/tables/part2_classification.csv")

# --- Rutas propias de nba_predictor ------------------------------------------
DATA_DIR = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # crudos descargados (nba_api)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # limpios + combinados
OUTPUTS_DIR = os.path.join(HERE, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")

UNIFIED_DATASET = os.path.join(DATA_DIR, "unified_team_season.csv")

# --- Modelos de estilo (features y k del clustering que ya funciona) ---------
STYLE_FEATURES = [
    "OFF_RATING", "DEF_RATING", "AST_PCT", "OREB_PCT",
    "DREB_PCT", "TM_TOV_PCT", "TS_PCT", "PACE",
]
STYLE_K = 3

# --- Parámetros --------------------------------------------------------------
SEED = 42
# Temporadas con AMBAS señales disponibles (estilo existe desde 2019-20)
SEASONS_FULL = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)
