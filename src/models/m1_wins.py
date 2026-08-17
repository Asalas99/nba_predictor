"""
FASE C — M1: modelo de VICTORIAS con backtesting walk-forward.

Predice las victorias de la temporada T usando SOLO informacion anterior a T:
  - squad_strength_proj  (fuerza proyectada del plantel, Fase B)
  - continuity           (continuidad de plantilla vs T-1)
  - prior_wins           (victorias de T-1)
  - coach_resid          (residual de rendimiento del entrenador)

Backtest walk-forward: para cada temporada de prueba T se entrena con todas las
temporadas < T y se predice T (sin ver el futuro). Se compara contra baselines.

  python -m src.models.m1_wins

Crea:
  outputs/tables/m1_backtest.csv      metricas por temporada y modelo
  outputs/tables/m1_predictions.csv   prediccion por equipo (todas las de prueba)
  outputs/figures/m1_pred_vs_real.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

# Features de M1. Elegidas por backtest + ablacion (m1_experiments):
#  - prior_net_rating (diferencial previo) predice mejor que prior_wins.
#  - best_pie_proj (estrella): el techo lo marca el mejor jugador -> MAE 7.36->7.10.
#  - avg_age_core (edad del nucleo) -> MAE 7.10->7.02.
#  - coach_resid se descarto: metia ruido.
FEATURES = ["squad_strength_proj", "continuity", "prior_net_rating",
            "best_pie_proj", "avg_age_core"]


def year(season: str) -> int:
    return int(season[:4])


def prev_season(season: str) -> str:
    y = year(season) - 1
    return f"{y}-{str(y + 1)[2:]}"


def assemble() -> pd.DataFrame:
    P = config.PROCESSED_DIR
    proj = pd.read_csv(os.path.join(P, "teams", "combined", "team_projection.csv"))
    coach = pd.read_csv(os.path.join(P, "teams", "combined", "coach_features.csv"))
    tm = pd.read_csv(os.path.join(P, "teams", "combined", "team_clean.csv"))

    wins = tm[["SEASON", "TEAM_ID", "W"]].rename(columns={"W": "wins"})

    def shift_next(frame, col):
        f = frame.copy()
        f["SEASON"] = f["SEASON"].map(lambda s: f"{year(s)+1}-{str(year(s)+2)[2:]}")
        return f.rename(columns={col: f"prior_{col}"})

    # prior_wins (baseline) y prior_net_rating (feature de M1), ambos de T-1
    pw = shift_next(wins.rename(columns={"wins": "wins"}), "wins")
    pnr = shift_next(tm[["SEASON", "TEAM_ID", "NET_RATING"]], "NET_RATING") \
        .rename(columns={"prior_NET_RATING": "prior_net_rating"})

    # avg_age_core (edad del nucleo) desde squad_strength
    age = pd.read_csv(os.path.join(P, "players", "combined", "squad_strength.csv"))[
        ["SEASON", "TEAM_ID", "avg_age_core"]]

    df = (proj.merge(coach[["SEASON", "TEAM_ID", "coach_resid", "is_new_coach"]],
                     on=["SEASON", "TEAM_ID"], how="left")
              .merge(age, on=["SEASON", "TEAM_ID"], how="left")
              .merge(wins, on=["SEASON", "TEAM_ID"], how="left")
              .merge(pw, on=["SEASON", "TEAM_ID"], how="left")
              .merge(pnr, on=["SEASON", "TEAM_ID"], how="left"))
    df["yr"] = df["SEASON"].map(year)
    return df.dropna(subset=FEATURES + ["wins"]).reset_index(drop=True)


def mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def backtest(df: pd.DataFrame):
    seasons = sorted(df["SEASON"].unique())
    metric_rows, pred_rows = [], []
    for T in seasons:
        train = df[df["yr"] < year(T)]
        test = df[df["SEASON"] == T]
        if len(train) < 20 or len(test) == 0:
            continue
        # --- modelo M1 (Ridge sobre features estandarizadas) ---
        sc = StandardScaler().fit(train[FEATURES])
        model = Ridge(alpha=1.0).fit(sc.transform(train[FEATURES]), train["wins"])
        pred = model.predict(sc.transform(test[FEATURES]))
        pred = np.clip(pred, 0, 82)
        # --- baselines ---
        base_prior = test["prior_wins"].values                    # persistencia
        base_mean = np.full(len(test), train["wins"].mean())      # media liga
        m_model = mae(test["wins"], pred)
        metric_rows.append(dict(
            season=T, n=len(test),
            MAE_M1=round(m_model, 2),
            MAE_prior_wins=round(mae(test["wins"], base_prior), 2),
            MAE_media=round(mae(test["wins"], base_mean), 2),
            corr_M1=round(float(np.corrcoef(test["wins"], pred)[0, 1]), 3),
        ))
        for tid, name, yreal, yhat in zip(test["TEAM_ID"], test["SEASON"],
                                          test["wins"], pred):
            pred_rows.append(dict(SEASON=T, TEAM_ID=tid, wins_real=yreal,
                                  wins_pred=round(float(yhat), 1)))
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def main() -> None:
    df = assemble()
    metrics, preds = backtest(df)

    tdir, fdir = config.TABLES_DIR, config.FIGURES_DIR
    metrics.to_csv(os.path.join(tdir, "m1_backtest.csv"), index=False)
    preds.to_csv(os.path.join(tdir, "m1_predictions.csv"), index=False)

    print("=" * 68)
    print("M1 — BACKTEST WALK-FORWARD (predecir T con datos < T)")
    print("=" * 68)
    print(metrics.to_string(index=False))
    print("-" * 68)
    ov = preds["wins_real"].values, preds["wins_pred"].values
    print(f"GLOBAL  MAE_M1={mae(*ov):.2f}   "
          f"MAE_prior={mae(df.set_index(['SEASON','TEAM_ID']).loc[list(zip(preds['SEASON'],preds['TEAM_ID'])),'prior_wins'].values, preds['wins_real'].values):.2f}   "
          f"corr={np.corrcoef(*ov)[0,1]:.3f}   n={len(preds)}")

    # figura pred vs real
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(preds["wins_real"], preds["wins_pred"], s=40, alpha=0.6,
               color="#5B8DEF")
    ax.plot([10, 70], [10, 70], "--", color="gray")
    ax.set_xlabel("Victorias reales")
    ax.set_ylabel("Victorias predichas (M1, walk-forward)")
    ax.set_title(f"M1: predicho vs real  (MAE={mae(*ov):.1f} victorias)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "m1_pred_vs_real.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n[m1] tablas -> outputs/tables/m1_backtest.csv, m1_predictions.csv")
    print(f"[m1] figura -> outputs/figures/m1_pred_vs_real.png")


if __name__ == "__main__":
    main()
