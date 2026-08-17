"""
Construye el dataset unificado equipo x temporada de nba_predictor.

Une tres fuentes (todas ya generadas por los repos de origen):

  1. team_style_clusters.csv  (nba_clustering_comp)
       -> estilo: cluster de juego + 8 features escaladas dentro de cada temporada
  2. team_stats_clean.csv     (nba_clustering_comp)
       -> stats avanzadas SIN escalar + W/L/W_PCT/NET_RATING (labels + contexto)
  3. part2_classification.csv (nba_tanking)
       -> true_strength (fuerza real del nucleo, APM) + tank_prob + categoria

Salida: data/unified_team_season.csv  (una fila = equipo x temporada)

Cruce:
  - clustering interno por (SEASON, TEAM_ID)
  - tanking por (SEASON, TEAM_NAME) con normalizacion de nombres
"""

import os
import sys

import pandas as pd

# Permite `python -m src.ingest.build_dataset` y ejecucion directa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402


# --- Normalizacion de nombres de equipo --------------------------------------
# Alias conocidos entre fuentes (nba_api vs. tabla de tanking).
TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
    "clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "lakers": "los angeles lakers",
}


def norm_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    return TEAM_ALIASES.get(key, key)


def _require(path: str, hint: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe:\n  {path}\n-> {hint}"
        )


def load_style() -> pd.DataFrame:
    _require(
        config.STYLE_CLUSTERS,
        "Falta el clustering de estilo. Corre `python -m src.features.clean_teams` "
        "y luego `python -m src.models.style_clustering` (o simplemente run_all.py).",
    )
    df = pd.read_csv(config.STYLE_CLUSTERS)
    # Prefijo a las 8 features de estilo escaladas para evitar colisiones.
    style_feats = [
        "OFF_RATING", "DEF_RATING", "AST_PCT", "OREB_PCT",
        "DREB_PCT", "TM_TOV_PCT", "TS_PCT", "PACE",
    ]
    present = [c for c in style_feats if c in df.columns]
    df = df.rename(columns={c: f"style_z_{c.lower()}" for c in present})
    keep = ["SEASON", "TEAM_ID", "TEAM_NAME", "TEAM_STYLE_CLUSTER", "CLUSTER_NAME"]
    keep += [f"style_z_{c.lower()}" for c in present]
    return df[[c for c in keep if c in df.columns]].copy()


def load_stats() -> pd.DataFrame:
    _require(
        config.TEAM_STATS_CLEAN,
        "Falta team_clean.csv. Corre `python -m src.features.clean_teams` "
        "(o simplemente run_all.py).",
    )
    df = pd.read_csv(config.TEAM_STATS_CLEAN)
    # Stats sin escalar + labels de victorias. Renombra a minusculas claras.
    ren = {
        "W": "wins", "L": "losses", "W_PCT": "win_pct",
        "OFF_RATING": "off_rating", "DEF_RATING": "def_rating",
        "NET_RATING": "net_rating", "PACE": "pace", "TS_PCT": "ts_pct",
        "PIE": "pie", "GP": "gp",
    }
    df = df.rename(columns=ren)
    keep = ["SEASON", "TEAM_ID", "gp", "wins", "losses", "win_pct",
            "off_rating", "def_rating", "net_rating", "pace", "ts_pct", "pie"]
    return df[[c for c in keep if c in df.columns]].copy()


def load_tanking():
    # OPCIONAL y DEPRECADO: el true_strength de nba_tanking no correlaciona con
    # victorias (ver EDA); lo reemplaza squad_strength. Si el repo externo no esta
    # presente, el pipeline sigue sin problema.
    if not os.path.exists(config.TANKING_CLASSIFICATION):
        print("[build] (info) nba_tanking no disponible: se omite true_strength "
              "(ya reemplazado por squad_strength).")
        return None
    df = pd.read_csv(config.TANKING_CLASSIFICATION)
    ren = {
        "season": "SEASON", "team_name": "TEAM_NAME_TANK",
        "true_strength": "true_strength", "strength_rank": "strength_rank",
        "tank_prob": "tank_prob", "tank_pred": "tank_pred",
        "categoria": "tank_categoria", "team_abbr": "team_abbr",
    }
    df = df.rename(columns=ren)
    keep = ["SEASON", "TEAM_NAME_TANK", "team_abbr", "true_strength",
            "strength_rank", "tank_prob", "tank_pred", "tank_categoria"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["_key_name"] = df["TEAM_NAME_TANK"].map(norm_team)
    return df


def build() -> pd.DataFrame:
    style = load_style()
    stats = load_stats()
    tank = load_tanking()

    # 1) estilo + stats por (SEASON, TEAM_ID)
    merged = style.merge(stats, on=["SEASON", "TEAM_ID"], how="left",
                         validate="one_to_one")

    # 2) + tanking (OPCIONAL) por (SEASON, nombre normalizado)
    if tank is not None:
        merged["_key_name"] = merged["TEAM_NAME"].map(norm_team)
        merged = merged.merge(
            tank.drop(columns=["TEAM_NAME_TANK"]),
            on=["SEASON", "_key_name"], how="left", validate="one_to_one",
        ).drop(columns=["_key_name"])

    # --- Diagnostico de cruce -------------------------------------------------
    n = len(merged)
    miss_stats = merged["wins"].isna().sum() if "wins" in merged else n
    miss_tank = merged["true_strength"].isna().sum() if "true_strength" in merged else n
    print(f"[build] filas totales: {n}")
    print(f"[build] temporadas:    {sorted(merged['SEASON'].unique())}")
    print(f"[build] sin wins (stats):        {miss_stats}")
    if "true_strength" in merged.columns:
        print(f"[build] sin true_strength (tank): {miss_tank}")

    # 3) + fuerza del plantel (squad_strength) por (SEASON, TEAM_ID)
    ss_path = os.path.join(config.PROCESSED_DIR, "players", "combined",
                           "squad_strength.csv")
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path)[
            ["SEASON", "TEAM_ID", "squad_strength", "pie_wmean", "pie_top5",
             "best_pie", "net_wmean", "avg_age_core"]]
        merged = merged.merge(ss, on=["SEASON", "TEAM_ID"], how="left",
                              validate="one_to_one")
        print(f"[build] sin squad_strength: {merged['squad_strength'].isna().sum()}")
    else:
        print("[build] (aviso) squad_strength.csv no existe: corre "
              "src.features.squad_strength")

    # 4) + tipo de plantel (roster_type) por (SEASON, TEAM_ID)
    rt_path = os.path.join(config.PROCESSED_DIR, "teams", "combined",
                           "roster_type_clusters.csv")
    if os.path.exists(rt_path):
        rt = pd.read_csv(rt_path)
        rt_cols = ["SEASON", "TEAM_ID", "ROSTER_TYPE", "ROSTER_TYPE_NAME"] + \
                  [c for c in rt.columns if c.startswith("share_")]
        merged = merged.merge(rt[rt_cols], on=["SEASON", "TEAM_ID"], how="left",
                              validate="one_to_one")
        print(f"[build] sin roster_type: {merged['ROSTER_TYPE'].isna().sum()}")
    else:
        print("[build] (aviso) roster_type_clusters.csv no existe: corre "
              "src.models.roster_type_clustering")

    return merged


def main() -> None:
    df = build()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.UNIFIED_DATASET, index=False)
    print(f"\n[build] guardado -> {config.UNIFIED_DATASET}")
    print(f"[build] shape: {df.shape}")
    print(f"[build] columnas: {list(df.columns)}")


if __name__ == "__main__":
    main()
