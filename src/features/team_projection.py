"""
FASE B: fuerza PROYECTADA del equipo + continuidad de plantilla.

Agrega las proyecciones de jugador (player_projections.csv) a nivel
equipo-temporada, sin fuga: todo sale de la historia < T.

  python -m src.features.team_projection

Lee : data/processed/players/combined/player_projections.csv
      data/processed/players/combined/player_clean.csv   (para continuidad)
Crea: data/processed/teams/combined/team_projection.csv
      (SEASON, TEAM_ID, squad_strength_proj, pie_wmean_proj, pie_top5_proj,
       continuity, n_returning, roster_change)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

PIE_COL = "pie_proj_std"  # la curva que mejor valido en Fase A


def prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{str(y + 1)[2:]}"


def team_aggregate(proj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, team), g in proj.groupby(["SEASON", "TEAM_ID"]):
        g = g.sort_values("min_proj", ascending=False)
        m = g["min_proj"].clip(lower=0)
        w = m / m.sum() if m.sum() > 0 else np.ones(len(g)) / len(g)
        rows.append(dict(
            SEASON=season, TEAM_ID=team,
            pie_wmean_proj=float((g[PIE_COL] * w).sum()),
            pie_top5_proj=float(g.head(5)[PIE_COL].sum()),
            best_pie_proj=float(g[PIE_COL].max()),
        ))
    out = pd.DataFrame(rows)
    # z-score dentro de cada temporada (comparable entre anios, sin fuga: usa
    # solo los 30 equipos de esa misma temporada)
    out["squad_strength_proj"] = out.groupby("SEASON")["pie_wmean_proj"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    return out


def continuity(proj: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """% de minutos proyectados que vienen de jugadores que YA estaban en el
    equipo la temporada anterior."""
    # equipo de cada jugador en cada temporada (real)
    prev_team = clean.set_index(["SEASON", "PLAYER_ID"])["TEAM_ID"].to_dict()
    rows = []
    for (season, team), g in proj.groupby(["SEASON", "TEAM_ID"]):
        ps = prev_season(season)
        m = g["min_proj"].clip(lower=0)
        tot = m.sum()
        ret_mask = [prev_team.get((ps, pid)) == team for pid in g["PLAYER_ID"]]
        ret_min = m[np.array(ret_mask)].sum()
        rows.append(dict(
            SEASON=season, TEAM_ID=team,
            continuity=float(ret_min / tot) if tot > 0 else np.nan,
            n_returning=int(np.sum(ret_mask)),
        ))
    df = pd.DataFrame(rows)
    df["roster_change"] = 1.0 - df["continuity"]
    return df


def main() -> None:
    base = os.path.join(config.PROCESSED_DIR, "players", "combined")
    proj = pd.read_csv(os.path.join(base, "player_projections.csv"))
    clean = pd.read_csv(os.path.join(base, "player_clean.csv"))

    agg = team_aggregate(proj)
    cont = continuity(proj, clean)
    out = agg.merge(cont, on=["SEASON", "TEAM_ID"], how="left")

    dst = os.path.join(config.PROCESSED_DIR, "teams", "combined", "team_projection.csv")
    out.to_csv(dst, index=False)

    # validacion sin fuga: ¿la fuerza PROYECTADA predice las wins reales?
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))[["SEASON", "TEAM_ID", "W"]]
    v = out.merge(tm, on=["SEASON", "TEAM_ID"])
    print(f"[teamproj] {len(out)} equipos-temporada -> {dst}")
    print("[teamproj] correlacion de features PROYECTADAS con wins reales:")
    for c in ["squad_strength_proj", "pie_wmean_proj", "pie_top5_proj", "continuity"]:
        print(f"   {c:20s} pearson={v[c].corr(v['W']):.3f}  "
              f"spearman={v[c].corr(v['W'], method='spearman'):.3f}")


if __name__ == "__main__":
    main()
