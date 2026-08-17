"""
Descarga el ENTRENADOR PRINCIPAL (head coach) por equipo y temporada.
Usa la segunda tabla de CommonTeamRoster (coaches).

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_coaches --start 2019 --end 2025

Guarda: data/raw/coaches/coaches.csv
  (SEASON, TEAM_ID, coach_name, coach_id)
Base para el efecto entrenador del predictor.
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, seasons_range, team_index  # noqa: E402


def download(start_year: int, end_year: int) -> None:
    from nba_api.stats.endpoints import commonteamroster
    team_list, _ = team_index()
    rows = []
    for season in seasons_range(start_year, end_year):
        print(f"=== {season} ===")
        for t in team_list:
            frames = retry(lambda t=t: commonteamroster.CommonTeamRoster(
                team_id=t["id"], season=season, timeout=60).get_data_frames())
            coaches = frames[1] if len(frames) > 1 else pd.DataFrame()
            coaches.columns = [c.upper() for c in coaches.columns]
            head = coaches[coaches.get("COACH_TYPE", "") == "Head Coach"] \
                if "COACH_TYPE" in coaches.columns else coaches
            if len(head):
                r = head.iloc[0]
                name = r.get("COACH_NAME") or f"{r.get('FIRST_NAME','')} {r.get('LAST_NAME','')}".strip()
                rows.append(dict(SEASON=season, TEAM_ID=t["id"],
                                coach_name=name, coach_id=r.get("COACH_ID")))
            time.sleep(0.5)
    out_dir = os.path.join(config.RAW_DIR, "coaches")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "coaches.csv")
    new = pd.DataFrame(rows)
    # MERGE: no borra temporadas ya guardadas; reemplaza solo las re-descargadas
    if os.path.exists(dst) and not new.empty:
        old = pd.read_csv(dst)
        old = old[~old["SEASON"].isin(new["SEASON"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(dst, index=False)
    print(f"LISTO: coaches.csv actualizado ({len(rows)} nuevos, {len(new)} en total) -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2025)
    a = ap.parse_args()
    download(a.start, a.end)
