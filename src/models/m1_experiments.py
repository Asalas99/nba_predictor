"""
Pruebas de ROBUSTEZ y busqueda de mejoras para M1.

  python -m src.models.m1_experiments

Corre (todo en backtest walk-forward, sin fuga):
  1. Estabilidad ante la regularizacion (alpha de Ridge)
  2. Comparacion de modelos (Ridge, Lasso, OLS, GradientBoosting, RF, kNN)
  3. Intervalo de confianza del MAE por bootstrap
  4. Ablacion de features: base + cada candidata, para ver que suma
  5. Estabilidad por temporada (¿el MAE se mantiene o brinca?)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.models import m1_wins  # noqa: E402

BASE = ["squad_strength_proj", "continuity", "prior_net_rating"]
RNG = np.random.default_rng(config.SEED)


def year(s):
    return int(s[:4])


def data():
    df = m1_wins.assemble()   # ya incluye avg_age_core y best_pie_proj
    if "avg_age_core" not in df.columns:
        ss = pd.read_csv(os.path.join(config.PROCESSED_DIR, "players", "combined",
                                      "squad_strength.csv"))[["SEASON", "TEAM_ID", "avg_age_core"]]
        df = df.merge(ss, on=["SEASON", "TEAM_ID"], how="left")
    return df


def walk_forward(df, feats, make_model):
    preds = []
    for T in sorted(df.SEASON.unique()):
        tr = df[df.yr < year(T)]
        te = df[df.SEASON == T]
        if len(tr) < 20 or len(te) == 0:
            continue
        sc = StandardScaler().fit(tr[feats])
        mod = make_model().fit(sc.transform(tr[feats]), tr["wins"])
        p = np.clip(mod.predict(sc.transform(te[feats])), 0, 82)
        preds += list(zip(te["wins"].values, p))
    a, b = map(np.array, zip(*preds))
    return np.abs(a - b).mean(), float(np.corrcoef(a, b)[0, 1]), np.array(preds)


def main():
    df = data()
    print("=" * 64)
    print("PRUEBAS DE ROBUSTEZ DE M1  (150 predicciones, walk-forward)")
    print("=" * 64)

    print("\n[1] Estabilidad ante alpha (Ridge):")
    for al in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        mae, corr, _ = walk_forward(df, BASE, lambda al=al: Ridge(alpha=al))
        print(f"     alpha={al:5.1f}  MAE={mae:.2f}  corr={corr:.3f}")

    print("\n[2] Comparacion de modelos (mismas 3 features):")
    models = {
        "Ridge(1.0)": lambda: Ridge(alpha=1.0),
        "Lasso(0.1)": lambda: Lasso(alpha=0.1),
        "ElasticNet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5),
        "OLS": lambda: LinearRegression(),
        "GradBoost": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                                       random_state=config.SEED),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=4,
                                                      random_state=config.SEED),
        "kNN(5)": lambda: KNeighborsRegressor(n_neighbors=5),
    }
    for name, mk in models.items():
        mae, corr, _ = walk_forward(df, BASE, mk)
        print(f"     {name:14s}  MAE={mae:.2f}  corr={corr:.3f}")

    print("\n[3] Intervalo de confianza del MAE (bootstrap, Ridge):")
    _, _, preds = walk_forward(df, BASE, lambda: Ridge(alpha=1.0))
    err = np.abs(preds[:, 0] - preds[:, 1])
    boot = [err[RNG.integers(0, len(err), len(err))].mean() for _ in range(3000)]
    print(f"     MAE = {err.mean():.2f}   IC 90% = [{np.percentile(boot,5):.2f}, "
          f"{np.percentile(boot,95):.2f}]")

    print("\n[4] Ablacion: base + cada feature candidata (¿suma?):")
    base_mae, base_corr, _ = walk_forward(df, BASE, lambda: Ridge(alpha=1.0))
    print(f"     BASE {BASE}  ->  MAE={base_mae:.2f}  corr={base_corr:.3f}")
    for c in ["best_pie_proj", "pie_top5_proj", "avg_age_core", "roster_change",
              "coach_resid", "is_new_coach", "prior_wins"]:
        if c not in df.columns:
            continue
        d = df.dropna(subset=[c])
        mae, corr, _ = walk_forward(d, BASE + [c], lambda: Ridge(alpha=1.0))
        flag = "  <== mejora" if mae < base_mae - 0.03 else ""
        print(f"     + {c:16s}  MAE={mae:.2f}  corr={corr:.3f}{flag}")

    print("\n[5] Estabilidad por temporada (MAE de cada una):")
    for T in sorted(df.SEASON.unique()):
        tr = df[df.yr < year(T)]
        te = df[df.SEASON == T]
        if len(tr) < 20:
            continue
        sc = StandardScaler().fit(tr[BASE])
        mod = Ridge(1.0).fit(sc.transform(tr[BASE]), tr["wins"])
        p = np.clip(mod.predict(sc.transform(te[BASE])), 0, 82)
        print(f"     {T}: MAE={np.abs(te['wins'].values-p).mean():.2f}")


if __name__ == "__main__":
    main()
