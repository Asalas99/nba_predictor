"""
Mapa de contendientes: version PROYECTADA (solo sabiendo el roster, sin datos
del ano) vs. version REAL (estadisticas ya jugadas), para ver el cambio.

Ambos ejes son comparables y proyectables:
  - X = fuerza del plantel  (squad_strength real  vs  squad_strength_proj)
  - Y = parecido a campeon en CONSTRUCCION DE PLANTEL (arquetipos):
        distancia de la composicion del equipo al centroide de los campeones.
        Composicion real (minutos jugados) vs proyectada (minutos y arquetipos
        estimados desde la historia de cada jugador).

  python -m src.viz.contention_compare

Genera en outputs/figures/:
  contention_real_2025-26.png       fuerza y parecido REALES
  contention_proj_2025-26.png       fuerza y parecido PROYECTADOS (preseason)
  contention_shift_2025-26.png      flecha proyectado -> real por equipo
Y outputs/tables/contention_compare_2025-26.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.data.champions import CHAMPIONS  # noqa: E402

PROC, FIG, TAB = config.PROCESSED_DIR, config.FIGURES_DIR, config.TABLES_DIR
CURRENT = "2025-26"
ARCHES = ["Creador / base", "Creador / base 2", "Interior / protector de aro",
          "Interior / protector de aro 2", "Rol / bajo uso", "Wing versatil"]
DEFAULT_ARCH = "Rol / bajo uso"  # para rookies sin arquetipo previo


def nickname(name):
    return {"Portland Trail Blazers": "Blazers"}.get(name, str(name).split()[-1])


def year(s):
    return int(s[:4])


def realized_composition():
    """Composicion realizada (share_*) desde roster_type_clusters."""
    r = pd.read_csv(os.path.join(PROC, "teams", "combined", "roster_type_clusters.csv"))
    cols = {f"share_{a}": a for a in ARCHES}
    keep = ["SEASON", "TEAM_ID", "TEAM_NAME"] + list(cols)
    r = r[keep].rename(columns=cols)
    return r


def projected_composition():
    """Composicion proyectada: minutos proyectados x arquetipo previo del jugador."""
    roles = pd.read_csv(os.path.join(PROC, "players", "combined", "player_roles.csv"))
    roles["yr"] = roles["SEASON"].map(year)
    proj = pd.read_csv(os.path.join(PROC, "players", "combined", "player_projections.csv"))

    # arquetipo mas reciente de cada jugador ANTES de cada temporada objetivo
    roles_sorted = roles.sort_values("yr")
    rows = []
    for target, g in proj.groupby("SEASON"):
        ty = year(target)
        hist = roles_sorted[roles_sorted["yr"] < ty]
        last_arch = (hist.groupby("PLAYER_ID")["ARCHETYPE"].last().to_dict())
        g = g.copy()
        g["arch"] = g["PLAYER_ID"].map(last_arch).fillna(DEFAULT_ARCH)
        g["w"] = g["min_proj"].clip(lower=0)
        for team, tg in g.groupby("TEAM_ID"):
            tot = tg["w"].sum()
            shares = {a: 0.0 for a in ARCHES}
            if tot > 0:
                for a, sub in tg.groupby("arch"):
                    if a in shares:
                        shares[a] = sub["w"].sum() / tot
            rows.append(dict(SEASON=target, TEAM_ID=team, **shares))
    return pd.DataFrame(rows)


def proximity(comp, centroid, dist_max):
    d = np.sqrt(((comp[ARCHES].values - centroid) ** 2).sum(axis=1))
    return 1 - d / dist_max, d


def main():
    real_comp = realized_composition()
    proj_comp = projected_composition()

    # centroide de campeones en el espacio de composicion REAL
    champ_mask = [CHAMPIONS.get(s) == t for s, t in zip(real_comp["SEASON"], real_comp["TEAM_NAME"])]
    centroid = real_comp.loc[champ_mask, ARCHES].mean().values
    # escala comun de distancia (para que real y proyectado sean comparables)
    dref = np.sqrt(((real_comp[ARCHES].values - centroid) ** 2).sum(axis=1))
    dist_max = dref.max()

    real_comp["champ_sim_real"], _ = proximity(real_comp, centroid, dist_max)
    proj_comp["champ_sim_proj"], _ = proximity(proj_comp, centroid, dist_max)

    # fuerzas
    ss = pd.read_csv(os.path.join(PROC, "players", "combined", "squad_strength.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength"]]
    tp = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_projection.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength_proj"]]

    cur = (real_comp[real_comp.SEASON == CURRENT][["TEAM_ID", "TEAM_NAME", "champ_sim_real"]]
           .merge(proj_comp[proj_comp.SEASON == CURRENT][["TEAM_ID", "champ_sim_proj"]], on="TEAM_ID")
           .merge(ss[ss.SEASON == CURRENT][["TEAM_ID", "squad_strength"]], on="TEAM_ID")
           .merge(tp[tp.SEASON == CURRENT][["TEAM_ID", "squad_strength_proj"]], on="TEAM_ID"))
    cur.to_csv(os.path.join(TAB, f"contention_compare_{CURRENT}.csv"), index=False)

    _map(cur, "squad_strength_proj", "champ_sim_proj",
         f"Mapa de contendientes PROYECTADO — {CURRENT}\n(solo sabiendo el roster, sin datos del ano)",
         os.path.join(FIG, f"contention_proj_{CURRENT}.png"))
    _map(cur, "squad_strength", "champ_sim_real",
         f"Mapa de contendientes REAL — {CURRENT}\n(estadisticas ya jugadas)",
         os.path.join(FIG, f"contention_real_{CURRENT}.png"))
    _shift(cur, os.path.join(FIG, f"contention_shift_{CURRENT}.png"))
    print(f"[compare] listo. tabla -> outputs/tables/contention_compare_{CURRENT}.csv")


def _map(cur, xcol, ycol, title, path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(cur[xcol], cur[ycol], s=60, alpha=0.6, color="#5B8DEF")
    for _, r in cur.iterrows():
        ax.annotate(nickname(r["TEAM_NAME"]), (r[xcol], r[ycol]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axvline(cur[xcol].median(), color="gray", ls="--", lw=0.8)
    ax.axhline(cur[ycol].median(), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Fuerza del plantel")
    ax.set_ylabel("Parecido de construccion a campeon")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def _shift(cur, path):
    fig, ax = plt.subplots(figsize=(12, 9))
    for _, r in cur.iterrows():
        ax.annotate("", xy=(r["squad_strength"], r["champ_sim_real"]),
                    xytext=(r["squad_strength_proj"], r["champ_sim_proj"]),
                    arrowprops=dict(arrowstyle="->", color="#B0B0B0", lw=1.1))
        ax.scatter(r["squad_strength_proj"], r["champ_sim_proj"], s=28,
                   color="#9AB8F0", zorder=3)
        ax.scatter(r["squad_strength"], r["champ_sim_real"], s=55,
                   color="#E0763A", zorder=4)
        ax.annotate(nickname(r["TEAM_NAME"]), (r["squad_strength"], r["champ_sim_real"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    ax.scatter([], [], color="#9AB8F0", label="Proyectado (preseason)")
    ax.scatter([], [], color="#E0763A", label="Real (fin de temporada)")
    ax.set_xlabel("Fuerza del plantel")
    ax.set_ylabel("Parecido de construccion a campeon")
    ax.set_title(f"Cambio proyectado -> real — {CURRENT}\n(la flecha va de lo esperado a lo que paso)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
