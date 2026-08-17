"""
M3 — SIMULACION DE PLAYOFFS (Monte Carlo).

Desde los seeds de M2 arma el bracket de cada conferencia y simula las series
miles de veces para estimar la probabilidad de que cada equipo avance por ronda
y gane el titulo.

Probabilidad de un partido: formula log5 (dos win% -> prob head-to-head) +
ventaja de local. Probabilidad de serie al mejor de 7: se calcula exacta por
programacion dinamica sobre el patron de sedes 2-2-1-1-1.

  python -m src.models.m3_playoffs

Usa outputs/tables/m2_seeding.csv. Crea:
  outputs/tables/m3_playoff_probs.csv   prob por equipo-temporada y ronda
  outputs/figures/m3_title_odds_2025-26.png
"""

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.data.champions import CHAMPIONS  # noqa: E402

N_SIMS = 20000
HCA = 0.06                       # ventaja de local sobre la prob de partido
VENUES = ["H", "H", "A", "A", "H", "A", "H"]   # 2-2-1-1-1 para el mejor seed
RNG = np.random.default_rng(config.SEED)


def log5(qa: float, qb: float) -> float:
    """Prob de que A gane un partido dado dos win% (Bill James log5)."""
    denom = qa + qb - 2 * qa * qb
    if denom <= 0:
        return 0.5
    return (qa - qa * qb) / denom


def series_prob(qa: float, qb: float) -> float:
    """Prob de que A (mejor seed, con ventaja de local) gane la serie al 7."""
    base = log5(qa, qb)

    @lru_cache(maxsize=None)
    def rec(i, wa, wb):
        if wa == 4:
            return 1.0
        if wb == 4:
            return 0.0
        p = base + (HCA if VENUES[i] == "H" else -HCA)
        p = min(max(p, 0.02), 0.98)
        return p * rec(i + 1, wa + 1, wb) + (1 - p) * rec(i + 1, wa, wb + 1)

    return rec(0, 0, 0)


def sim_conference(seeds):
    """seeds: lista de 8 dicts {tid, q, seed} ordenada por seed (1..8).
    Devuelve (campeon_conf, dict de ronda alcanzada por tid)."""
    reached = {s["tid"]: 1 for s in seeds}  # 1 = entro a playoffs

    def play(a, b):
        # a,b dicts; el de mejor seed (menor) es local
        hi, lo = (a, b) if a["seed"] < b["seed"] else (b, a)
        p = series_prob(hi["q"], lo["q"])
        return hi if RNG.random() < p else lo

    # Ronda 1: 1v8, 4v5, 3v6, 2v7
    r1 = [play(seeds[0], seeds[7]), play(seeds[3], seeds[4]),
          play(seeds[2], seeds[5]), play(seeds[1], seeds[6])]
    for w in r1:
        reached[w["tid"]] = 2  # llego a semis de conferencia
    # Semis
    s1 = play(r1[0], r1[1])
    s2 = play(r1[2], r1[3])
    for w in (s1, s2):
        reached[w["tid"]] = 3  # llego a final de conferencia
    # Final de conferencia
    cf = play(s1, s2)
    reached[cf["tid"]] = 4  # campeon de conferencia (llego a Finales)
    return cf, reached


def sim_season(df_season):
    east = df_season[df_season.conf == "East"].nsmallest(8, "seed_pred")
    west = df_season[df_season.conf == "West"].nsmallest(8, "seed_pred")

    def to_seeds(d):
        d = d.sort_values("seed_pred")
        return [dict(tid=r.TEAM_ID, q=r.q, seed=int(r.seed_pred))
                for r in d.itertuples()]
    es, ws = to_seeds(east), to_seeds(west)

    tally = {tid: dict(r2=0, cf=0, finals=0, champ=0)
             for tid in list(east.TEAM_ID) + list(west.TEAM_ID)}
    for _ in range(N_SIMS):
        ec, er = sim_conference(es)
        wc, wr = sim_conference(ws)
        for r in (er, wr):
            for tid, stage in r.items():
                if stage >= 2:
                    tally[tid]["r2"] += 1
                if stage >= 3:
                    tally[tid]["cf"] += 1
                if stage >= 4:
                    tally[tid]["finals"] += 1
        # Finales: local para el de mayor q
        a, b = (ec, wc) if ec["q"] >= wc["q"] else (wc, ec)
        champ = a if RNG.random() < series_prob(a["q"], b["q"]) else b
        tally[champ["tid"]]["champ"] += 1

    rows = []
    for tid, t in tally.items():
        rows.append(dict(TEAM_ID=tid,
                         p_round2=t["r2"] / N_SIMS, p_conf_final=t["cf"] / N_SIMS,
                         p_finals=t["finals"] / N_SIMS, p_champion=t["champ"] / N_SIMS))
    return pd.DataFrame(rows)


def main():
    seeding = pd.read_csv(config.find_table("m2_seeding.csv"))
    seeding["q"] = np.clip(seeding["wins_pred"] / 82.0, 0.05, 0.95)

    out = []
    for season, g in seeding.groupby("SEASON"):
        res = sim_season(g)
        res["SEASON"] = season
        res = res.merge(g[["TEAM_ID", "TEAM_NAME", "conf", "seed_pred", "wins_pred"]],
                        on="TEAM_ID", how="left")
        out.append(res)
    allp = pd.concat(out, ignore_index=True)
    allp.to_csv(os.path.join(config.TABLES_DIR, "m3_playoff_probs.csv"), index=False)

    # validacion: prob que M3 daba al campeon REAL
    allp["is_champion"] = [int(CHAMPIONS.get(s) == t)
                           for s, t in zip(allp["SEASON"], allp["TEAM_NAME"])]
    labeled = allp[allp["SEASON"].isin(CHAMPIONS)]
    champ_rows = labeled[labeled["is_champion"] == 1]
    print("=" * 60)
    print(f"M3 — SIMULACION DE PLAYOFFS ({N_SIMS} sims/temporada)")
    print("=" * 60)
    print("Prob de titulo que M3 asignaba al campeon REAL (preseason):")
    print(champ_rows[["SEASON", "TEAM_NAME", "p_champion"]].round(3).to_string(index=False))
    brier = ((labeled["p_champion"] - labeled["is_champion"]) ** 2).mean()
    print(f"\nBrier score (campeon, {len(labeled)} equipos-temporada): {brier:.4f}")

    last = sorted(allp["SEASON"].unique())[-1]
    cur = allp[allp["SEASON"] == last].nlargest(10, "p_champion")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(cur["TEAM_NAME"], cur["p_champion"] * 100, color="#5B8DEF")
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad de campeon (%)")
    ax.set_title(f"M3 — Odds de titulo (preseason) — {last}", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, f"m3_title_odds_{last}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[m3] -> outputs/tables/m3_playoff_probs.csv")
    print(f"[m3] -> outputs/figures/m3_title_odds_{last}.png")


if __name__ == "__main__":
    main()
