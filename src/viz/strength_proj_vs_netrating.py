"""
¿La fuerza PROYECTADA del plantel (squad_strength_proj, la que alimenta M1,
construida SIN datos del año) anticipa el NET_RATING real de fin de temporada?
Un panel por año, comparado contra la fuerza REALIZADA.

  python -m src.viz.strength_proj_vs_netrating

Genera:
  outputs/figures/strength_proj_vs_netrating.png
  outputs/tables/strength_proj_netrating_corr.csv
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

PROC, FIG, TAB = config.PROCESSED_DIR, config.FIGURES_DIR, config.TABLES_DIR


def main():
    tp = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_projection.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength_proj"]]
    ssr = pd.read_csv(os.path.join(PROC, "players", "combined", "squad_strength.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength"]]
    tm = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_clean.csv"))[
        ["SEASON", "TEAM_ID", "NET_RATING"]]
    df = tp.merge(tm, on=["SEASON", "TEAM_ID"]).merge(ssr, on=["SEASON", "TEAM_ID"])

    seasons = sorted(df["SEASON"].unique())
    rows = []
    ncol = 3
    nrow = int(np.ceil(len(seasons) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.array(axes).ravel()

    for ax, season in zip(axes, seasons):
        d = df[df["SEASON"] == season]
        r_proj = d["squad_strength_proj"].corr(d["NET_RATING"])
        r_real = d["squad_strength"].corr(d["NET_RATING"])
        rows.append(dict(season=season, r_proyectada=round(r_proj, 3),
                         r_realizada=round(r_real, 3), n=len(d)))
        ax.scatter(d["squad_strength_proj"], d["NET_RATING"], s=38, alpha=0.6,
                   color="#2FB380")
        if len(d) > 2:
            m, b = np.polyfit(d["squad_strength_proj"], d["NET_RATING"], 1)
            xs = np.linspace(d["squad_strength_proj"].min(), d["squad_strength_proj"].max(), 50)
            ax.plot(xs, m * xs + b, "--", color="#E0763A", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{season}   r = {r_proj:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("squad_strength_proj (preseason)")
        ax.set_ylabel("NET_RATING real")
        ax.grid(alpha=0.15)

    for ax in axes[len(seasons):]:
        ax.axis("off")

    overall = df["squad_strength_proj"].corr(df["NET_RATING"])
    fig.suptitle(f"Fuerza PROYECTADA (alimenta M1) vs. NET_RATING real de fin de "
                 f"temporada   —   r global = {overall:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "strength_proj_vs_netrating.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(TAB, "strength_proj_netrating_corr.csv"), index=False)
    print("Correlacion con NET_RATING real (proyectada vs realizada):")
    print(corr.to_string(index=False))
    print(f"\nGLOBAL  proyectada r = {overall:.3f}   "
          f"(realizada r = {df['squad_strength'].corr(df['NET_RATING']):.3f})")
    print("[viz] -> outputs/figures/strength_proj_vs_netrating.png")


if __name__ == "__main__":
    main()
