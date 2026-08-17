"""
Descarga los ROSTERS (conformacion de plantilla) + ENTRENADOR de una temporada,
incluida la que esta por empezar (aun sin estadisticas).

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_rosters --start 2026 --end 2026

Una llamada por equipo (commonteamroster) que devuelve jugadores Y cuerpo
tecnico. Guarda:
  data/raw/rosters/{season}/roster.csv     (SEASON, TEAM_ID, PLAYER_ID, PLAYER_NAME, AGE, POSITION, EXP)
  data/raw/coaches/coaches.csv             (MERGE: añade la temporada sin borrar las viejas)
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, seasons_range, team_index  # noqa: E402


def fetch_team(team, season):
    from nba_api.stats.endpoints import commonteamroster
    frames = retry(lambda: commonteamroster.CommonTeamRoster(
        team_id=team["id"], season=season, timeout=60).get_data_frames())
    players = frames[0].copy()
    coaches = frames[1].copy() if len(frames) > 1 else pd.DataFrame()
    players.columns = [c.upper() for c in players.columns]
    coaches.columns = [c.upper() for c in coaches.columns]

    prows = []
    for _, r in players.iterrows():
        try:
            age = int(float(r.get("AGE", 0) or 0))
        except (ValueError, TypeError):
            age = 0
        prows.append(dict(SEASON=season, TEAM_ID=team["id"],
                          PLAYER_ID=int(r["PLAYER_ID"]), PLAYER_NAME=r.get("PLAYER"),
                          AGE=age, POSITION=r.get("POSITION"), EXP=r.get("EXP")))

    coach_row = None
    if len(coaches):
        head = coaches[coaches.get("COACH_TYPE", "") == "Head Coach"]
        if len(head):
            c = head.iloc[0]
            coach_row = dict(SEASON=season, TEAM_ID=team["id"],
                             coach_name=c.get("COACH_NAME"), coach_id=c.get("COACH_ID"))
    return prows, coach_row


def merge_coaches(new_rows, season):
    """Añade la temporada a coaches.csv sin borrar las anteriores."""
    path = os.path.join(config.RAW_DIR, "coaches", "coaches.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame(new_rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["SEASON"] != season]          # reemplaza esa temporada si existia
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)
    print(f"  coaches.csv actualizado ({len(new)} entrenadores de {season})")


def download(start_year, end_year):
    team_list, _ = team_index()
    for season in seasons_range(start_year, end_year):
        print(f"=== {season} ===")
        all_players, coach_rows = [], []
        for t in team_list:
            prows, crow = fetch_team(t, season)
            all_players += prows
            if crow:
                coach_rows.append(crow)
            time.sleep(0.5)
        if not all_players:
            print(f"  ! {season}: sin jugadores (¿rosters aun no publicados?)")
            continue
        out_dir = os.path.join(config.RAW_DIR, "rosters", season)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(all_players).to_csv(os.path.join(out_dir, "roster.csv"), index=False)
        print(f"  roster.csv guardado ({len(all_players)} jugadores en "
              f"{pd.DataFrame(all_players)['TEAM_ID'].nunique()} equipos)")
        if coach_rows:
            merge_coaches(coach_rows, season)
    print("\nLISTO: rosters + entrenadores descargados.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2026)
    ap.add_argument("--end", type=int, default=2026)
    a = ap.parse_args()
    download(a.start, a.end)
