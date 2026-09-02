"""
Compara nuestras victorias proyectadas 2026-27 (M1) contra las lineas del
mercado (Kalshi / casino). Genera el grafico de distancia y, para los equipos
donde diferimos mas que nuestro MAE, desglosa que feature suma o resta mucho.

  python -m src.predict.vs_casino

Salidas en outputs/prediccion_2026-27/:
  vs_casino.png                nuestras wins vs casino (con la distancia)
  vs_casino.csv                tabla: nuestras, casino, distancia
  vs_casino_desglose.csv       aporte de cada feature en los equipos con gap grande
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
from src.models import m1_wins  # noqa: E402

OUT = os.path.join(config.OUTPUTS_DIR, "prediccion_2026-27")
MAE = 7.02  # MAE promedio de M1

# Victorias proyectadas del mercado (Kalshi, via Sportsbook Review, 29-jul-2026).
# Clave = nombre tal como aparece en nuestra tabla.
CASINO = {
    "Atlanta Hawks": 45.0, "Boston Celtics": 51.0, "Brooklyn Nets": 26.2,
    "Charlotte Hornets": 38.6, "Chicago Bulls": 29.5, "Cleveland Cavaliers": 47.1,
    "Dallas Mavericks": 37.5, "Denver Nuggets": 48.9, "Detroit Pistons": 51.1,
    "Golden State Warriors": 38.9, "Houston Rockets": 46.8, "Indiana Pacers": 44.7,
    "LA Clippers": 29.7, "Los Angeles Lakers": 47.4, "Memphis Grizzlies": 30.0,
    "Miami Heat": 46.4, "Milwaukee Bucks": 27.8, "Minnesota Timberwolves": 50.0,
    "New Orleans Pelicans": 28.4, "New York Knicks": 51.3, "Oklahoma City Thunder": 58.9,
    "Orlando Magic": 45.3, "Philadelphia 76ers": 51.1, "Phoenix Suns": 40.3,
    "Portland Trail Blazers": 40.8, "Sacramento Kings": 23.7, "San Antonio Spurs": 60.8,
    "Toronto Raptors": 44.8, "Utah Jazz": 39.6, "Washington Wizards": 29.6,
}


def main():
    tab = pd.read_csv(os.path.join(OUT, "tabla_2026-27.csv"))
    tab["casino"] = tab["TEAM_NAME"].map(CASINO)
    tab["distancia"] = (tab["wins_pred"] - tab["casino"]).round(1)
    tab["gap_abs"] = tab["distancia"].abs()

    tab[["TEAM_NAME", "wins_pred", "casino", "distancia"]].sort_values(
        "distancia").to_csv(os.path.join(OUT, "vs_casino.csv"), index=False)

    # ---- grafico tipo lollipop: nuestras vs casino con la distancia ----
    d = tab.sort_values("wins_pred").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 11))
    for i, r in d.iterrows():
        big = r["gap_abs"] > MAE
        ax.plot([r["casino"], r["wins_pred"]], [i, i],
                color="#D65A7A" if big else "#C8CDD6", lw=2.5 if big else 1.5, zorder=1)
    ax.scatter(d["casino"], range(len(d)), s=45, color="#8894A8", zorder=3,
               label="Casino (Kalshi)")
    ax.scatter(d["wins_pred"], range(len(d)), s=55, color="#5B8DEF", zorder=3,
               label="Nuestra proyeccion (M1)")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["TEAM_NAME"], fontsize=8.5)
    for i, r in d.iterrows():
        if r["gap_abs"] > MAE:
            ax.text(max(r["casino"], r["wins_pred"]) + 0.6, i,
                    f"{r['distancia']:+.0f}", va="center", fontsize=8,
                    color="#B03A5B", fontweight="bold")
    ax.set_xlabel("Victorias proyectadas")
    ax.set_title("Nuestra proyeccion vs. el casino — 2026-27\n"
                 f"la linea = distancia; rojo = mayor que nuestro MAE ({MAE:.0f})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vs_casino.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- equipos con distancia > MAE ----
    big = tab[tab["gap_abs"] > MAE].sort_values("distancia", ascending=False)

    # ---- desglose de features (que suma/resta) para esos equipos ----
    F = m1_wins.FEATURES
    hist = m1_wins.assemble().dropna(subset=F + ["wins"])
    sc = StandardScaler().fit(hist[F])
    model = Ridge(alpha=1.0).fit(sc.transform(hist[F]), hist["wins"])
    # z de cada feature 2026-27 con el MISMO escalado del modelo
    Z = pd.DataFrame(sc.transform(tab[F]), columns=F, index=tab.index)
    Zc = Z - Z.mean()                       # desviacion respecto a la liga 2026-27
    contrib = Zc * model.coef_              # aporte de cada feature en victorias
    contrib["TEAM_NAME"] = tab["TEAM_NAME"].values

    rows = []
    for _, r in big.iterrows():
        c = contrib[contrib.TEAM_NAME == r["TEAM_NAME"]].iloc[0]
        aportes = {f: round(c[f], 1) for f in F}
        top = sorted(aportes.items(), key=lambda x: -abs(x[1]))[:2]
        rows.append(dict(equipo=r["TEAM_NAME"], nuestras=r["wins_pred"],
                         casino=r["casino"], distancia=r["distancia"],
                         **aportes,
                         explica=", ".join(f"{k} {v:+.1f}" for k, v in top)))
    desg = pd.DataFrame(rows)
    desg.to_csv(os.path.join(OUT, "vs_casino_desglose.csv"), index=False)

    print("=" * 70)
    print("NUESTRA PROYECCION vs CASINO 2026-27")
    print("=" * 70)
    print(f"distancia media absoluta: {tab['gap_abs'].mean():.1f} victorias")
    print(f"equipos con distancia > MAE ({MAE:.0f}): {len(big)}")
    print("\nDonde MAS diferimos (+ = decimos mas que el casino):")
    print(big[["TEAM_NAME", "wins_pred", "casino", "distancia"]].to_string(index=False))
    print("\nQue feature suma/resta mucho en esos equipos:")
    print(desg[["equipo", "distancia", "explica"]].to_string(index=False))
    print(f"\n[vs_casino] -> {OUT}/vs_casino.png, vs_casino.csv, vs_casino_desglose.csv")


if __name__ == "__main__":
    main()
