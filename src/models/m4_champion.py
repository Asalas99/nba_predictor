"""
M4 — PROBABILIDAD DE CAMPEON calibrada.

Toma la probabilidad de titulo de M3 (frecuencia en la simulacion) y la calibra
con la PROXIMIDAD DE CONSTRUCCION a campeon (proyectada, sin fuga): equipos cuya
plantilla se arma como la de los campeones reciben un empujon.

  p_M4(equipo) proporcional a  p_M3 * exp(w * z_proximidad)   (normalizado por temporada)

El peso w se elige por la mejora en log-loss del campeon en backtest.

  python -m src.models.m4_champion

Crea:
  outputs/tables/m4_champion_odds.csv
  outputs/figures/m4_title_odds_2025-26.png
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
from src.viz.contention_compare import (projected_composition, realized_composition,
                                        proximity, ARCHES)  # noqa: E402

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # candidatos para w


def champion_similarity_all():
    """Proximidad de construccion PROYECTADA a campeon, para todo equipo-temporada."""
    real = realized_composition()
    proj = projected_composition()
    champ_mask = [CHAMPIONS.get(s) == t for s, t in zip(real["SEASON"], real["TEAM_NAME"])]
    centroid = real.loc[champ_mask, ARCHES].mean().values
    dref = np.sqrt(((real[ARCHES].values - centroid) ** 2).sum(axis=1))
    dist_max = dref.max()
    sim, _ = proximity(proj, centroid, dist_max)
    proj = proj.copy()
    proj["champ_sim_proj"] = sim
    return proj[["SEASON", "TEAM_ID", "champ_sim_proj"]]


def blend(df, w):
    """score = p_m3 * exp(w * z(sim)); normalizado por temporada."""
    out = df.copy()
    out["z_sim"] = out.groupby("SEASON")["champ_sim_proj"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    out["score"] = out["p_champion"] * np.exp(w * out["z_sim"])
    out["p_m4"] = out.groupby("SEASON")["score"].transform(lambda s: s / s.sum())
    return out["p_m4"].values


def logloss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def main():
    m3 = pd.read_csv(config.find_table("m3_playoff_probs.csv"))
    sim = champion_similarity_all()
    df = m3.merge(sim, on=["SEASON", "TEAM_ID"], how="left")
    df["champ_sim_proj"] = df["champ_sim_proj"].fillna(df["champ_sim_proj"].mean())
    df["is_champion"] = [int(CHAMPIONS.get(s) == t)
                         for s, t in zip(df["SEASON"], df["TEAM_NAME"])]

    labeled = df[df["SEASON"].isin(CHAMPIONS)].copy()
    y = labeled["is_champion"].values

    print("=" * 60)
    print("M4 — CALIBRACION con proximidad de construccion")
    print("=" * 60)
    print("Eleccion de w (peso de la proximidad) por log-loss del campeon:")
    best_w, best_ll = 0.0, np.inf
    for w in WEIGHTS:
        p = blend(labeled, w)
        ll = logloss(p, y)
        br = ((p - y) ** 2).mean()
        print(f"   w={w:.2f}   logloss={ll:.4f}   brier={br:.4f}")
        if ll < best_ll:
            best_ll, best_w = ll, w
    print(f"-> w elegido = {best_w}")

    df["p_champion_m4"] = blend(df, best_w)
    df.to_csv(os.path.join(config.TABLES_DIR, "m4_champion_odds.csv"), index=False)

    # comparacion M3 vs M4 sobre temporadas etiquetadas
    p3 = labeled["p_champion"].values
    p4 = blend(labeled, best_w)
    print(f"\nComparacion (n={len(labeled)} equipos-temporada etiquetados):")
    print(f"   M3 solo:   logloss={logloss(p3,y):.4f}  brier={((p3-y)**2).mean():.4f}")
    print(f"   M4 (w={best_w}): logloss={logloss(p4,y):.4f}  brier={((p4-y)**2).mean():.4f}")

    last = sorted(df["SEASON"].unique())[-1]
    cur = df[df["SEASON"] == last].nlargest(10, "p_champion_m4")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(cur["TEAM_NAME"], cur["p_champion_m4"] * 100, color="#E0A100")
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad de campeon (%)  —  M4 calibrado")
    ax.set_title(f"M4 — Odds de titulo (preseason, calibrado) — {last}", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, f"m4_title_odds_{last}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[m4] -> outputs/tables/m4_champion_odds.csv")
    print(f"[m4] -> outputs/figures/m4_title_odds_{last}.png")


if __name__ == "__main__":
    main()
