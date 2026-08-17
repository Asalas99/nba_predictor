"""
M2 — SEEDING por conferencia.

Convierte las victorias proyectadas por M1 en la clasificacion de cada
conferencia: seeds 1-15 y los tramos playoffs (1-6), play-in (7-10) y loteria
(11-15). Valida en backtest contra la clasificacion real.

  python -m src.models.m2_seeding

Usa la salida de M1 (outputs/tables/m1_predictions.csv). Si no existe, la genera.

Crea:
  outputs/tables/m2_seeding.csv        seed predicho vs real por equipo-temporada
  outputs/tables/m2_metrics.csv        metricas por temporada y conferencia
  outputs/tables/m2_standings_2025-26.csv  clasificacion proyectada vs real
  outputs/figures/m2_seed_pred_vs_real.png
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
from src.data.conferences import conference  # noqa: E402


def bucket(seed: int) -> str:
    if seed <= 6:
        return "Playoffs"
    if seed <= 10:
        return "Play-in"
    return "Loteria"


def ensure_m1() -> pd.DataFrame:
    path = config.find_table("m1_predictions.csv")
    if not os.path.exists(path):
        from src.models import m1_wins
        m1_wins.main()
    return pd.read_csv(path)


def build() -> pd.DataFrame:
    preds = ensure_m1()
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))[["SEASON", "TEAM_ID", "TEAM_NAME"]]
    df = preds.merge(tm, on=["SEASON", "TEAM_ID"], how="left")
    df["conf"] = df["TEAM_NAME"].map(conference)

    # seeds dentro de cada conferencia-temporada (rank por victorias, desc)
    df["seed_pred"] = (df.groupby(["SEASON", "conf"])["wins_pred"]
                         .rank(ascending=False, method="first").astype(int))
    df["seed_real"] = (df.groupby(["SEASON", "conf"])["wins_real"]
                         .rank(ascending=False, method="first").astype(int))
    df["bucket_pred"] = df["seed_pred"].map(bucket)
    df["bucket_real"] = df["seed_real"].map(bucket)
    return df


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, conf), g in df.groupby(["SEASON", "conf"]):
        spearman = g["seed_pred"].corr(g["seed_real"], method="spearman")
        seed_mae = (g["seed_pred"] - g["seed_real"]).abs().mean()
        # ¿cuantos de los 6 de playoffs reales predijimos en top-6?
        real_po = set(g[g["seed_real"] <= 6]["TEAM_ID"])
        pred_po = set(g[g["seed_pred"] <= 6]["TEAM_ID"])
        po_hit = len(real_po & pred_po)
        bucket_acc = (g["bucket_pred"] == g["bucket_real"]).mean()
        rows.append(dict(season=season, conf=conf,
                         spearman_seed=round(spearman, 3),
                         seed_MAE=round(seed_mae, 2),
                         playoffs_aciertos=f"{po_hit}/6",
                         bucket_acc=round(bucket_acc, 2)))
    return pd.DataFrame(rows)


def standings_table(df: pd.DataFrame, season: str) -> pd.DataFrame:
    s = df[df["SEASON"] == season].copy()
    out = []
    for conf in ["East", "West"]:
        c = s[s["conf"] == conf].sort_values("seed_pred")
        for _, r in c.iterrows():
            out.append(dict(conf=conf, seed_pred=r["seed_pred"],
                            team=r["TEAM_NAME"],
                            wins_pred=round(r["wins_pred"], 1),
                            wins_real=int(r["wins_real"]),
                            seed_real=int(r["seed_real"]),
                            bucket_pred=r["bucket_pred"]))
    return pd.DataFrame(out)


def main() -> None:
    df = build()
    met = metrics(df)
    tdir, fdir = config.TABLES_DIR, config.FIGURES_DIR

    df.to_csv(os.path.join(tdir, "m2_seeding.csv"), index=False)
    met.to_csv(os.path.join(tdir, "m2_metrics.csv"), index=False)
    last = sorted(df["SEASON"].unique())[-1]
    st = standings_table(df, last)
    st.to_csv(os.path.join(tdir, f"m2_standings_{last}.csv"), index=False)

    print("=" * 66)
    print("M2 — SEEDING (backtest walk-forward, seeds por conferencia)")
    print("=" * 66)
    print(met.to_string(index=False))
    print("-" * 66)
    print(f"GLOBAL  spearman_seed={df.groupby(['SEASON','conf']).apply(lambda g: g['seed_pred'].corr(g['seed_real'],method='spearman')).mean():.3f}   "
          f"seed_MAE={(df['seed_pred']-df['seed_real']).abs().mean():.2f}   "
          f"bucket_acc={(df['bucket_pred']==df['bucket_real']).mean():.2f}")
    tot_po = sum(len(set(g[g.seed_real<=6].TEAM_ID) & set(g[g.seed_pred<=6].TEAM_ID))
                 for _, g in df.groupby(['SEASON','conf']))
    n_conf = df.groupby(['SEASON','conf']).ngroups
    print(f"GLOBAL  clasificados a playoffs correctos: {tot_po}/{n_conf*6} "
          f"({100*tot_po/(n_conf*6):.0f}%)")

    print(f"\nClasificacion PROYECTADA vs real — {last}:")
    print(st.to_string(index=False))

    # figura: seed predicho vs real
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["seed_real"], df["seed_pred"], s=35, alpha=0.5, color="#5B8DEF")
    ax.plot([1, 15], [1, 15], "--", color="gray")
    for b in (6.5, 10.5):
        ax.axhline(b, color="#E0A100", lw=0.6, ls=":")
        ax.axvline(b, color="#E0A100", lw=0.6, ls=":")
    ax.set_xlabel("Seed real")
    ax.set_ylabel("Seed predicho (M2, walk-forward)")
    ax.set_title("M2: seed predicho vs real\n(lineas: cortes playoffs/play-in/loteria)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "m2_seed_pred_vs_real.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n[m2] tablas -> m2_seeding.csv, m2_metrics.csv, m2_standings_{last}.csv")
    print(f"[m2] figura -> outputs/figures/m2_seed_pred_vs_real.png")


if __name__ == "__main__":
    main()
