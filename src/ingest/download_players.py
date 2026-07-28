"""
Descarga stats de JUGADORES por temporada (leaguedashplayerstats), base +
avanzadas, y las une por jugador. Sirven para talento del nucleo y roles.

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_players --start 2019 --end 2025

Guarda: data/raw/players/{season}/player_stats_raw.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, seasons_range  # noqa: E402


def fetch_player_stats(season: str, measure_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerstats
    print(f"  [API] player {measure_type} {season}...")
    df = retry(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
        timeout=90,
    ).get_data_frames()[0])
    print(f"  OK {len(df)} jugadores ({measure_type})")
    return df


def download(start_year: int, end_year: int, min_minutes: float = 5.0) -> None:
    out_base = os.path.join(config.RAW_DIR, "players")
    # columnas avanzadas de interes (evita duplicar las de Base)
    adv_keep = [
        "PLAYER_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT",
        "AST_TO", "USG_PCT", "TS_PCT", "EFG_PCT", "PIE", "PACE", "REB_PCT",
    ]
    for season in seasons_range(start_year, end_year):
        base = fetch_player_stats(season, "Base")
        time.sleep(1.2)
        adv = fetch_player_stats(season, "Advanced")
        time.sleep(1.2)

        adv_cols = [c for c in adv_keep if c in adv.columns]
        merged = base.merge(adv[adv_cols], on="PLAYER_ID", how="left",
                            suffixes=("", "_ADV"))
        merged["SEASON"] = season
        if "MIN" in merged.columns and min_minutes > 0:
            merged = merged[merged["MIN"] >= min_minutes].copy()

        season_dir = os.path.join(out_base, season)
        os.makedirs(season_dir, exist_ok=True)
        path = os.path.join(season_dir, "player_stats_raw.csv")
        merged.to_csv(path, index=False)
        print(f"  guardado -> {path}  ({merged.shape})")
        time.sleep(1.0)
    print("LISTO: jugadores descargados.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--min-minutes", type=float, default=5.0)
    a = ap.parse_args()
    download(a.start, a.end, a.min_minutes)
