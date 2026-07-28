"""
Descarga stats AVANZADAS de equipos por temporada (leaguedashteamstats).

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_teams --start 2019 --end 2025

Guarda: data/raw/teams/{season}/team_stats_raw.csv
Esquema identico al que ya usa nba_clustering_comp (mismas columnas nba_api).
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, seasons_range  # noqa: E402


def fetch_team_stats(season: str, measure_type: str = "Advanced") -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashteamstats
    print(f"  [API] team {measure_type} {season}...")
    df = retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
        timeout=90,
    ).get_data_frames()[0])
    df["SEASON"] = season
    df["SEASON_TYPE"] = "Regular Season"
    print(f"  OK {len(df)} equipos")
    return df


def download(start_year: int, end_year: int, min_games: int = 20) -> None:
    out_base = os.path.join(config.RAW_DIR, "teams")
    for season in seasons_range(start_year, end_year):
        df = fetch_team_stats(season)
        if "GP" in df.columns and min_games > 0:
            df = df[df["GP"] >= min_games].copy()
        season_dir = os.path.join(out_base, season)
        os.makedirs(season_dir, exist_ok=True)
        path = os.path.join(season_dir, "team_stats_raw.csv")
        df.to_csv(path, index=False)
        print(f"  guardado -> {path}  ({df.shape})")
        time.sleep(1.5)
    print("LISTO: equipos descargados.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--min-games", type=int, default=20)
    a = ap.parse_args()
    download(a.start, a.end, a.min_games)
