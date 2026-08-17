"""
¿La fuerza del plantel (squad_strength, basada en PIE) correlaciona con el
NET_RATING real de la temporada? Un panel por año.

  python -m src.viz.strength_vs_netrating

Genera:
  outputs/figures/strength_vs_netrating.png
  outputs/tables/strength_netrating_corr.csv
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
    ss = pd.read_csv(os.path.join(PROC, "players", "combined", "squad_strength.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength"]]
    tm = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_clean.csv"))[
        ["SEASON", "TEAM_ID", "TEAM_NAME", "NET_RATING"]]
    df = ss.merge(tm, on=["SEASON", "TEAM_ID"])

    seasons = sorted(df["SEASON"].unique())
    corr_rows = []
    ncol = 4
    nrow = int(np.ceil(len(seasons) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))
    axes = np.array(axes).ravel()

    for ax, season in zip(axes, seasons):
        d = df[df["SEASON"] == season]
        r = d["squad_strength"].corr(d["NET_RATING"])
        corr_rows.append(dict(season=season, r_pearson=round(r, 3),
                              r2=round(r ** 2, 3), n=len(d)))
        ax.scatter(d["squad_strength"], d["NET_RATING"], s=35, alpha=0.6, color="#5B8DEF")
        if len(d) > 2:
            m, b = np.polyfit(d["squad_strength"], d["NET_RATING"], 1)
            xs = np.linspace(d["squad_strength"].min(), d["squad_strength"].max(), 50)
            ax.plot(xs, m * xs + b, "--", color="#E0763A", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{season}   r = {r:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("squad_strength")
        ax.set_ylabel("NET_RATING")
        ax.grid(alpha=0.15)

    for ax in axes[len(seasons):]:
        ax.axis("off")

    overall = df["squad_strength"].corr(df["NET_RATING"])
    fig.suptitle(f"Fuerza del plantel (PIE) vs. NET_RATING real   —   "
                 f"correlacion global r = {overall:.2f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "strength_vs_netrating.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(TAB, "strength_netrating_corr.csv"), index=False)
    print("Correlacion squad_strength vs NET_RATING por temporada:")
    print(corr.to_string(index=False))
    print(f"\nGLOBAL r = {overall:.3f}  (r^2 = {overall**2:.3f})")
    print("[viz] -> outputs/figures/strength_vs_netrating.png")


if __name__ == "__main__":
    main()
