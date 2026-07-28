"""
Limpieza de stats de jugadores (base + avanzadas descargadas por
download_players). Prepara una tabla por jugador/temporada para estimar talento
del nucleo y roles.

  python -m src.features.clean_players

Lee : data/raw/players/{season}/player_stats_raw.csv
Crea:
  data/processed/players/by_year/{season}/player_clean.csv
  data/processed/players/combined/player_clean.csv

Nota: requiere haber corrido download_players en tu maquina. En el sandbox no
hay crudos de jugador (nba_api sin salida), asi que este paso queda listo pero
sin datos hasta que descargues.
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

KEEP = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "SEASON",
    "AGE", "GP", "MIN", "PTS", "AST", "REB", "STL", "BLK", "TOV", "FG3M",
    "FG3A", "FG_PCT", "FG3_PCT", "FT_PCT",
    # avanzadas (de download_players)
    "OFF_RATING", "DEF_RATING", "NET_RATING", "USG_PCT", "TS_PCT", "PIE",
    "AST_PCT", "REB_PCT",
]


def clean_season(raw_path: str) -> pd.DataFrame:
    season = os.path.basename(os.path.dirname(raw_path))
    df = pd.read_csv(raw_path)
    if "SEASON" not in df.columns:
        df["SEASON"] = season
    keep = [c for c in KEEP if c in df.columns]
    clean = df[keep].copy()
    # dropna solo en las columnas clave para no perder jugadores por un NaN suelto
    core = [c for c in ("PLAYER_ID", "TEAM_ID", "MIN", "PTS") if c in clean.columns]
    clean = clean.dropna(subset=core)

    out_dir = os.path.join(config.PROCESSED_DIR, "players", "by_year", season)
    os.makedirs(out_dir, exist_ok=True)
    clean.to_csv(os.path.join(out_dir, "player_clean.csv"), index=False)
    print(f"[clean_players] {season}: {len(clean)} jugadores")
    return clean


def main() -> None:
    raw_glob = os.path.join(config.RAW_DIR, "players", "*", "player_stats_raw.csv")
    paths = sorted(glob.glob(raw_glob))
    if not paths:
        print("[clean_players] No hay crudos de jugador en "
              f"{raw_glob}.\n  -> Corre `python -m src.ingest.download_players` "
              "en tu maquina primero.")
        return
    cleans = [clean_season(p) for p in paths]
    comb_dir = os.path.join(config.PROCESSED_DIR, "players", "combined")
    os.makedirs(comb_dir, exist_ok=True)
    pd.concat(cleans, ignore_index=True).to_csv(
        os.path.join(comb_dir, "player_clean.csv"), index=False)
    print(f"[clean_players] combinado -> {comb_dir} ({len(paths)} temporadas)")


if __name__ == "__main__":
    main()
