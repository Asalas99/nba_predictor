"""
Fuerza del plantel (squad strength) por equipo-temporada.

Reemplaza el `true_strength` heredado de nba_tanking (APM que NO correlacionaba
con victorias en datos reales). Aqui la fuerza se estima agregando el impacto
individual de los jugadores (PIE, box-score) ponderado por minutos.

Validado vs. victorias: PIE ponderado por minutos ~0.76 pearson (vs 0.03 del APM).

  python -m src.features.squad_strength

Lee : data/processed/players/combined/player_clean.csv
Crea: data/processed/players/combined/squad_strength.csv
      (SEASON, TEAM_ID, squad_strength, pie_wmean, pie_top5, best_pie,
       core_size, net_wmean, avg_age_core)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

CORE_N = 8  # nucleo = 8 jugadores con mas minutos


def team_strength(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("MIN", ascending=False)
    minutes = df["MIN"].clip(lower=0)
    w = minutes / minutes.sum() if minutes.sum() > 0 else np.ones(len(df)) / len(df)
    core = df.head(CORE_N)
    wc = core["MIN"] / core["MIN"].sum() if core["MIN"].sum() > 0 else None
    return pd.Series({
        "pie_wmean": float((df["PIE"] * w).sum()),
        "pie_top5": float(df.head(5)["PIE"].sum()),
        "pie_core": float(core["PIE"].sum()),
        "best_pie": float(df["PIE"].max()),
        "net_wmean": float((df["NET_RATING"] * w).sum()),
        "core_size": int(min(CORE_N, len(df))),
        "avg_age_core": float((core["AGE"] * wc).sum()) if wc is not None and "AGE" in core else np.nan,
    })


def main() -> None:
    path = os.path.join(config.PROCESSED_DIR, "players", "combined", "player_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Falta {path}. Corre download_players (tu maquina) + clean_players.")
    pl = pd.read_csv(path)

    g = pl.groupby(["SEASON", "TEAM_ID"]).apply(team_strength).reset_index()

    # squad_strength = z-score de pie_wmean DENTRO de cada temporada (comparable
    # entre anios, ademas de la escala absoluta que queda en pie_wmean).
    g["squad_strength"] = g.groupby("SEASON")["pie_wmean"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))

    out = os.path.join(config.PROCESSED_DIR, "players", "combined", "squad_strength.csv")
    g.to_csv(out, index=False)

    # validacion rapida vs wins
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))[["SEASON", "TEAM_ID", "W"]]
    m = g.merge(tm, on=["SEASON", "TEAM_ID"])
    print(f"[squad] {len(g)} equipos-temporada -> {out}")
    print("[squad] correlacion con victorias:")
    for c in ["squad_strength", "pie_wmean", "pie_top5", "net_wmean", "best_pie"]:
        print(f"   {c:14s} pearson={m[c].corr(m['W']):.3f}  "
              f"spearman={m[c].corr(m['W'], method='spearman'):.3f}")


if __name__ == "__main__":
    main()
