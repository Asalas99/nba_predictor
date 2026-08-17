"""
Experimento: DISPONIBILIDAD / LESIONES.  NO modifica M1.

Dos preguntas, ambas en backtest:
  A) EXPLICATIVO: ¿cuanto de los errores de M1 se explica por la disponibilidad
     REAL del ano? (equipos cuyo nucleo se lesiono deberian rendir por debajo)
  B) PREDICTIVO: ¿mejora M1 si añadimos una durabilidad PREVIA (leakage-free)?

Disponibilidad = partidos jugados (GP) ponderados por minutos proyectados,
sobre 82. avail_real usa el GP del ano (solo para explicar); avail_prior usa el
GP de la temporada anterior de cada jugador (sin fuga, para predecir).

  python -m src.models.injury_experiment
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.models import m1_wins, m1_experiments  # noqa: E402

P = config.PROCESSED_DIR


def year(s):
    return int(s[:4])


def prev(s):
    return f"{year(s)-1}-{str(year(s))[2:]}"


def build_availability():
    proj = pd.read_csv(os.path.join(P, "players", "combined", "player_projections.csv"))[
        ["SEASON", "TEAM_ID", "PLAYER_ID", "min_proj"]]
    gp = pd.read_csv(os.path.join(P, "players", "combined", "player_clean.csv"))[
        ["SEASON", "PLAYER_ID", "GP"]]
    gp_now = gp.rename(columns={"GP": "gp_now"})
    gp_prev = gp.copy()
    gp_prev["SEASON"] = gp_prev["SEASON"].map(lambda s: f"{year(s)+1}-{str(year(s)+2)[2:]}")
    gp_prev = gp_prev.rename(columns={"GP": "gp_prev"})

    d = (proj.merge(gp_now, on=["SEASON", "PLAYER_ID"], how="left")
             .merge(gp_prev, on=["SEASON", "PLAYER_ID"], how="left"))
    d["w"] = d["min_proj"].clip(lower=0)

    rows = []
    for (s, t), g in d.groupby(["SEASON", "TEAM_ID"]):
        w = g["w"] / g["w"].sum() if g["w"].sum() > 0 else None
        if w is None:
            continue
        avail_real = float((g["gp_now"].fillna(0) / 82.0 * w).sum())
        # prev: rookies sin gp_prev -> se asume disponibilidad media (0.85)
        gp_prev_fill = g["gp_prev"].fillna(0.85 * 82)
        avail_prior = float((gp_prev_fill / 82.0 * w).sum())
        rows.append(dict(SEASON=s, TEAM_ID=t, avail_real=round(avail_real, 3),
                         avail_prior=round(avail_prior, 3)))
    return pd.DataFrame(rows)


def main():
    av = build_availability()
    m1 = pd.read_csv(config.find_table("m1_predictions.csv"))
    tm = pd.read_csv(os.path.join(P, "teams", "combined", "team_clean.csv"))[
        ["SEASON", "TEAM_ID", "TEAM_NAME"]]
    df = m1.merge(av, on=["SEASON", "TEAM_ID"]).merge(tm, on=["SEASON", "TEAM_ID"])
    df["residual"] = df["wins_real"] - df["wins_pred"]   # + = rindio MAS de lo predicho

    print("=" * 66)
    print("A) EXPLICATIVO: ¿la disponibilidad real explica los errores de M1?")
    print("=" * 66)
    r = df["avail_real"].corr(df["residual"])
    print(f"  corr(disponibilidad_real, residual de M1) = {r:.3f}")
    print("  (positivo = equipos sanos rinden por ENCIMA de lo predicho, "
          "lesionados por debajo)")
    print("\n  Equipos MENOS disponibles (nucleo mas lesionado) y su residual:")
    worst = df.nsmallest(6, "avail_real")[["SEASON", "TEAM_NAME", "avail_real",
                                           "wins_pred", "wins_real", "residual"]]
    print(worst.to_string(index=False))

    print("\n" + "=" * 66)
    print("B) PREDICTIVO: ¿avail_prior (durabilidad previa) mejora M1? (sin fuga)")
    print("=" * 66)
    base = m1_experiments.data()
    F = m1_wins.FEATURES
    bmae, bcorr, _ = m1_experiments.walk_forward(base.dropna(subset=F), F, lambda: Ridge(1.0))
    ext = base.merge(av[["SEASON", "TEAM_ID", "avail_prior"]], on=["SEASON", "TEAM_ID"], how="left")
    F2 = F + ["avail_prior"]
    emae, ecorr, _ = m1_experiments.walk_forward(ext.dropna(subset=F2), F2, lambda: Ridge(1.0))
    print(f"  M1 actual                : MAE={bmae:.2f}  corr={bcorr:.3f}")
    print(f"  M1 + avail_prior         : MAE={emae:.2f}  corr={ecorr:.3f}")
    verdict = "SI mejora" if emae < bmae - 0.03 else ("no cambia" if abs(emae-bmae) <= 0.03 else "EMPEORA")
    print(f"  -> veredicto: {verdict}")


if __name__ == "__main__":
    main()
