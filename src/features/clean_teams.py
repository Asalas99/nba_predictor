"""
Limpieza de stats de equipos (portada del pipeline que funciona en
nba_clustering_comp: mismas 8 features, escalado por temporada con StandardScaler).

  python -m src.features.clean_teams

Lee : data/raw/teams/{season}/team_stats_raw.csv
Crea:
  data/processed/teams/by_year/{season}/team_clean.csv         (sin escalar + W/L)
  data/processed/teams/by_year/{season}/team_cluster_input.csv (8 features escaladas)
  data/processed/teams/by_year/{season}/scaler.pkl
  data/processed/teams/combined/team_clean.csv
  data/processed/teams/combined/team_cluster_input.csv
"""

import glob
import os
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

ID_COLS = ["SEASON", "TEAM_ID", "TEAM_NAME"]
# columnas sin escalar que conservamos (labels + contexto)
CLEAN_KEEP = ID_COLS + [
    "GP", "W", "L", "W_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "AST_RATIO", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT",
    "EFG_PCT", "TS_PCT", "PACE", "PIE",
]


def clean_season(raw_path: str) -> tuple:
    season = os.path.basename(os.path.dirname(raw_path))
    df = pd.read_csv(raw_path)
    if "SEASON" not in df.columns:
        df["SEASON"] = season

    keep = [c for c in CLEAN_KEEP if c in df.columns]
    clean = df[keep].dropna().copy()

    feats = [c for c in config.STYLE_FEATURES if c in clean.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean[feats])
    scaled_df = pd.DataFrame(scaled, columns=feats, index=clean.index)
    cluster_input = pd.concat(
        [clean[ID_COLS].reset_index(drop=True), scaled_df.reset_index(drop=True)],
        axis=1,
    )

    out_dir = os.path.join(config.PROCESSED_DIR, "teams", "by_year", season)
    os.makedirs(out_dir, exist_ok=True)
    clean.to_csv(os.path.join(out_dir, "team_clean.csv"), index=False)
    cluster_input.to_csv(os.path.join(out_dir, "team_cluster_input.csv"), index=False)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print(f"[clean] {season}: {len(clean)} equipos, {len(feats)} features")
    return clean, cluster_input


def main() -> None:
    raw_glob = os.path.join(config.RAW_DIR, "teams", "*", "team_stats_raw.csv")
    paths = sorted(glob.glob(raw_glob))
    if not paths:
        raise FileNotFoundError(
            f"No hay crudos en {raw_glob}. Corre download_teams (tu maquina) "
            "o seed_from_repos (sandbox)."
        )
    cleans, inputs = [], []
    for p in paths:
        c, i = clean_season(p)
        cleans.append(c)
        inputs.append(i)

    comb_dir = os.path.join(config.PROCESSED_DIR, "teams", "combined")
    os.makedirs(comb_dir, exist_ok=True)
    pd.concat(cleans, ignore_index=True).to_csv(
        os.path.join(comb_dir, "team_clean.csv"), index=False)
    pd.concat(inputs, ignore_index=True).to_csv(
        os.path.join(comb_dir, "team_cluster_input.csv"), index=False)
    print(f"[clean] combinado -> {comb_dir}  ({len(paths)} temporadas)")


if __name__ == "__main__":
    main()
