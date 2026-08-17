"""
Descarga el CALENDARIO (ya definido) de una temporada, incluida la que va a
empezar. Sirve para predecir partido a partido antes de que se juegue.

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_schedule --season 2026

Usa ScheduleLeagueV2 (frame SeasonGames) y filtra TEMPORADA REGULAR
(gameId que empieza en '002'). Guarda:
  data/raw/schedule/{season}/schedule.csv
     (SEASON, GAME_ID, GAME_DATE, home_team_id, home_team, away_team_id, away_team)
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, season_str  # noqa: E402


def fetch(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import scheduleleaguev2
    print(f"  [API] calendario {season}...")
    frames = retry(lambda: scheduleleaguev2.ScheduleLeagueV2(
        season=season, league_id="00", timeout=90).get_data_frames())
    # el primer frame es SeasonGames
    games = frames[0].copy()
    print(f"  {len(games)} juegos en total (incluye pretemporada/playoffs)")

    # normaliza nombres de columnas posibles (planas o con punto)
    def col(*cands):
        for c in cands:
            if c in games.columns:
                return c
        return None

    gid = col("gameId", "GAME_ID")
    gdate = col("gameDate", "gameDateEst", "GAME_DATE")
    h_id = col("homeTeam_teamId", "homeTeamTeamId")
    h_nm = col("homeTeam_teamName", "homeTeamTeamName")
    a_id = col("awayTeam_teamId", "awayTeamTeamId")
    a_nm = col("awayTeam_teamName", "awayTeamTeamName")
    h_city = col("homeTeam_teamCity", "homeTeamTeamCity")
    a_city = col("awayTeam_teamCity", "awayTeamTeamCity")

    df = pd.DataFrame({
        "SEASON": season,
        "GAME_ID": games[gid].astype(str),
        "GAME_DATE": games[gdate],
        "home_team_id": games[h_id],
        "home_team": (games[h_city].fillna("") + " " + games[h_nm].fillna("")).str.strip()
                     if h_city else games[h_nm],
        "away_team_id": games[a_id],
        "away_team": (games[a_city].fillna("") + " " + games[a_nm].fillna("")).str.strip()
                     if a_city else games[a_nm],
    })
    # temporada regular: gameId '002...'. Descarta pretemporada '001' y playoffs '004'
    df = df[df["GAME_ID"].str.startswith("002")].copy()
    # descarta filas sin equipos (TBD del calendario provisional)
    df = df[(df["home_team_id"] > 0) & (df["away_team_id"] > 0)]
    print(f"  temporada regular: {len(df)} partidos")
    return df


def download(start_year: int) -> None:
    season = season_str(start_year)
    df = fetch(season)
    if df.empty:
        print(f"  ! {season}: sin partidos de temporada regular (¿calendario aun no publicado?)")
        return
    out_dir = os.path.join(config.RAW_DIR, "schedule", season)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "schedule.csv")
    df.to_csv(path, index=False)
    print(f"  guardado -> {path}")
    print("LISTO: calendario descargado.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026,
                    help="anio de inicio (2026 = temporada 2026-27)")
    a = ap.parse_args()
    download(a.season)
