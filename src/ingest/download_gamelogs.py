"""
Descarga los GAME LOGS de equipo por temporada (calendario + resultados).

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_gamelogs --start 2021 --end 2025

Una llamada por temporada (leaguegamelog). Cada partido genera 2 filas (una por
equipo), con fecha, MATCHUP (local si dice 'vs.', visitante si dice '@'),
resultado (WL) y puntos. Se emparejan por GAME_ID para obtener el rival.

Guarda: data/raw/gamelogs/{season}/team_gamelog.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, seasons_range  # noqa: E402

KEEP = ["SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
        "TEAM_NAME", "MATCHUP", "WL", "PTS"]


def fetch_season(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog
    print(f"  [API] gamelog {season}...")
    df = retry(lambda: leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season",
        player_or_team_abbreviation="T", timeout=90).get_data_frames()[0])
    df["SEASON"] = season
    df = df[[c for c in KEEP if c in df.columns]].copy()
    print(f"  OK {len(df)} filas ({df['GAME_ID'].nunique()} partidos)")
    return df


def download(start_year: int, end_year: int) -> None:
    seasons = seasons_range(start_year, end_year)
    print(f"Temporadas a descargar: {seasons}")
    print("(recuerda: usa temporadas YA JUGADAS. La que esta por empezar no "
          "tiene partidos todavia y saldria vacia.)\n")
    guardadas, vacias = [], []
    for season in seasons:
        df = fetch_season(season)
        if df.empty or df["GAME_ID"].nunique() == 0:
            print(f"  ! {season}: 0 partidos -> NO se guarda (¿temporada sin empezar?)")
            vacias.append(season)
            time.sleep(1.5)
            continue
        out_dir = os.path.join(config.RAW_DIR, "gamelogs", season)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "team_gamelog.csv")
        df.to_csv(path, index=False)
        print(f"  guardado -> {path}")
        guardadas.append(season)
        time.sleep(1.5)
    print(f"\nLISTO. Guardadas: {guardadas or 'ninguna'}")
    if vacias:
        print(f"Vacias (omitidas): {vacias}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2025)
    a = ap.parse_args()
    download(a.start, a.end)
