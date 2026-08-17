"""
Revision de la prediccion de M1 en las temporadas con calendario:
  - victorias predichas vs reales
  - fuerza del plantel PROYECTADA vs la REAL (ya con datos del ano)

  python -m src.viz.prediction_review

Genera:
  outputs/tables/prediction_review.csv          todas las temporadas
  outputs/figures/prediction_review.png          2 paneles (wins y fuerza)
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


def build():
    m1 = pd.read_csv(config.find_table("m1_predictions.csv"))          # wins_pred, wins_real
    tp = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_projection.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength_proj"]]
    ss = pd.read_csv(os.path.join(PROC, "players", "combined", "squad_strength.csv"))[
        ["SEASON", "TEAM_ID", "squad_strength"]].rename(columns={"squad_strength": "squad_strength_real"})
    tm = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_clean.csv"))[
        ["SEASON", "TEAM_ID", "TEAM_NAME"]]
    df = (m1.merge(tp, on=["SEASON", "TEAM_ID"])
            .merge(ss, on=["SEASON", "TEAM_ID"])
            .merge(tm, on=["SEASON", "TEAM_ID"]))
    df["wins_pred"] = df["wins_pred"].round(1)
    df["wins_err"] = (df["wins_pred"] - df["wins_real"]).round(1)
    df["squad_strength_proj"] = df["squad_strength_proj"].round(2)
    df["squad_strength_real"] = df["squad_strength_real"].round(2)
    return df[["SEASON", "TEAM_NAME", "wins_pred", "wins_real", "wins_err",
               "squad_strength_proj", "squad_strength_real"]]


def main():
    df = build()
    df.to_csv(os.path.join(TAB, "prediction_review.csv"), index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
    # panel 1: wins
    a1.scatter(df["wins_real"], df["wins_pred"], s=32, alpha=0.5, color="#5B8DEF")
    a1.plot([15, 68], [15, 68], "--", color="gray")
    rw = df["wins_pred"].corr(df["wins_real"])
    mae = (df["wins_pred"] - df["wins_real"]).abs().mean()
    a1.set_xlabel("Victorias reales"); a1.set_ylabel("Victorias predichas (M1)")
    a1.set_title(f"Victorias: predicho vs real\nMAE={mae:.1f}   r={rw:.2f}", fontweight="bold")
    a1.grid(alpha=0.15)
    # panel 2: fuerza
    a2.scatter(df["squad_strength_real"], df["squad_strength_proj"], s=32, alpha=0.5,
               color="#2FB380")
    lim = [df["squad_strength_real"].min() - 0.3, df["squad_strength_real"].max() + 0.3]
    a2.plot(lim, lim, "--", color="gray")
    rs = df["squad_strength_proj"].corr(df["squad_strength_real"])
    a2.set_xlabel("Fuerza REAL (con datos del ano)")
    a2.set_ylabel("Fuerza PROYECTADA (preseason)")
    a2.set_title(f"Fuerza del plantel: proyectada vs real\nr={rs:.2f}", fontweight="bold")
    a2.grid(alpha=0.15)
    fig.suptitle("Revision de la prediccion de M1 (temporadas con calendario 2021-26)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "prediction_review.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # resumen por temporada + detalle de la ultima
    print("Resumen por temporada:")
    for s, g in df.groupby("SEASON"):
        print(f"  {s}: MAE victorias={ (g.wins_pred-g.wins_real).abs().mean():.1f}   "
              f"r(fuerza proy vs real)={g.squad_strength_proj.corr(g.squad_strength_real):.2f}")
    last = sorted(df["SEASON"].unique())[-1]
    print(f"\nDetalle {last} (ordenado por victorias predichas):")
    d = df[df.SEASON == last].sort_values("wins_pred", ascending=False)
    print(d[["TEAM_NAME", "wins_pred", "wins_real", "wins_err",
             "squad_strength_proj", "squad_strength_real"]].to_string(index=False))
    print(f"\n[review] -> outputs/tables/prediction_review.csv")
    print(f"[review] -> outputs/figures/prediction_review.png")


if __name__ == "__main__":
    main()
