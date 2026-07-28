"""
Descarga quintetos 5-man (leaguedashlineups) + standings + rosters por
temporada. Insumo para recalcular la fuerza real del nucleo (APM/RAPM).

  >>> CORRELO EN TU MAQUINA (necesita internet).  <<<
      python -m src.ingest.download_lineups --start 2019 --end 2025

Guarda:
  data/raw/lineups/lineups.csv    (season, team, window, player_ids, min, plus_minus)
  data/raw/lineups/standings.csv  (season, team, wins)
  data/raw/lineups/players.csv    (player_id, season, team, age)

Esquema compatible con nba_tanking (team = indice 0..29 por abreviatura).
Por defecto descarga la temporada COMPLETA (window='full'). Con --split-march
tambien baja las ventanas early/late (util para deteccion de tanking).
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.ingest.nba_client import retry, seasons_range, team_index  # noqa: E402


def _lineups_for_window(season, dfrom, dto, id2idx, window):
    from nba_api.stats.endpoints import leaguedashlineups
    ld = retry(lambda: leaguedashlineups.LeagueDashLineups(
        season=season,
        season_type_all_star="Regular Season",
        group_quantity=5,
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        date_from_nullable=dfrom,
        date_to_nullable=dto,
        timeout=90,
    ).get_data_frames()[0])
    ld.columns = [c.upper() for c in ld.columns]
    rows = []
    for _, r in ld.iterrows():
        tid = int(r["TEAM_ID"])
        if tid not in id2idx:
            continue
        pids = [p for p in str(r["GROUP_ID"]).split("-") if p]
        rows.append(dict(season=season, team=id2idx[tid], window=window,
                         player_ids="-".join(pids),
                         min=float(r["MIN"]), plus_minus=float(r["PLUS_MINUS"])))
    print(f"   {window}: {len(rows)} quintetos")
    return rows


def download(start_year: int, end_year: int, split_march: bool = False) -> None:
    team_list, id2idx = team_index()
    out = os.path.join(config.RAW_DIR, "lineups")
    os.makedirs(out, exist_ok=True)

    lineups, standings, players = [], [], []
    for season in seasons_range(start_year, end_year):
        print(f"\n=== {season} ===")
        end_year_s = int(season[:4]) + 1
        split = f"03/01/{end_year_s}"

        # standings
        from nba_api.stats.endpoints import leaguestandingsv3, commonteamroster
        st = retry(lambda: leaguestandingsv3.LeagueStandingsV3(
            season=season, season_type="Regular Season",
            timeout=60).get_data_frames()[0])
        st.columns = [c.upper() for c in st.columns]
        for _, r in st.iterrows():
            tid = int(r["TEAMID"])
            if tid in id2idx:
                standings.append(dict(season=season, team=id2idx[tid],
                                     wins=int(r["WINS"])))
        time.sleep(1.0)

        # rosters (edad)
        for t in team_list:
            cr = retry(lambda t=t: commonteamroster.CommonTeamRoster(
                team_id=t["id"], season=season, timeout=60).get_data_frames()[0])
            cr.columns = [c.upper() for c in cr.columns]
            for _, r in cr.iterrows():
                try:
                    age = int(float(r.get("AGE", 25) or 25))
                except (ValueError, TypeError):
                    age = 25
                players.append(dict(player_id=int(r["PLAYER_ID"]), season=season,
                                   team=id2idx[t["id"]], age=age))
            time.sleep(0.5)

        # lineups
        lineups += _lineups_for_window(season, "", "", id2idx, "full")
        time.sleep(1.0)
        if split_march:
            lineups += _lineups_for_window(season, "", split, id2idx, "early")
            time.sleep(1.0)
            lineups += _lineups_for_window(season, split, "", id2idx, "late")
            time.sleep(1.0)

    pd.DataFrame(lineups).to_csv(os.path.join(out, "lineups.csv"), index=False)
    pd.DataFrame(standings).drop_duplicates(["season", "team"]).to_csv(
        os.path.join(out, "standings.csv"), index=False)
    pd.DataFrame(players).drop_duplicates(["player_id", "season"]).to_csv(
        os.path.join(out, "players.csv"), index=False)
    print(f"\nLISTO: lineups={len(lineups)} standings={len(standings)} "
          f"players={len(players)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--split-march", action="store_true",
                    help="ademas baja ventanas early/late (deteccion de tanking)")
    a = ap.parse_args()
    download(a.start, a.end, a.split_march)
