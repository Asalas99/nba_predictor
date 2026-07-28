"""
EDA de validacion de hipotesis.

Pregunta central: ¿las dos senales que aporta este proyecto
(true_strength = fuerza real del nucleo, y el estilo) tienen relacion con
resultados (wins, campeonato)? Si no hay senal aqui, no tiene sentido montar la
cascada de prediccion todavia.

Genera:
  outputs/tables/eda_correlations.csv   correlaciones con wins
  outputs/tables/eda_champion_gap.csv   campeones vs resto (medias)
  outputs/figures/eda_strength_vs_wins.png
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

# Campeones y finalistas (fuente: nba_clustering_comp/src/data/champions_finalists.py)
CHAMPIONS = {
    "2019-20": "Los Angeles Lakers",
    "2020-21": "Milwaukee Bucks",
    "2021-22": "Golden State Warriors",
    "2022-23": "Denver Nuggets",
    "2023-24": "Boston Celtics",
    "2024-25": "Oklahoma City Thunder",
}
RUNNERS_UP = {
    "2019-20": "Miami Heat",
    "2020-21": "Phoenix Suns",
    "2021-22": "Boston Celtics",
    "2022-23": "Miami Heat",
    "2023-24": "Dallas Mavericks",
    "2024-25": "Indiana Pacers",
}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_champion"] = [
        int(CHAMPIONS.get(s) == t) for s, t in zip(df["SEASON"], df["TEAM_NAME"])
    ]
    df["is_finalist"] = [
        int(CHAMPIONS.get(s) == t or RUNNERS_UP.get(s) == t)
        for s, t in zip(df["SEASON"], df["TEAM_NAME"])
    ]
    # temporada con campeon conocido (excluye 2025-26 en curso)
    df["season_labeled"] = df["SEASON"].isin(CHAMPIONS).astype(int)
    return df


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    feats = ["squad_strength", "pie_wmean", "true_strength", "net_rating",
             "off_rating", "def_rating", "tank_prob", "ts_pct", "pace", "pie"]
    feats = [f for f in feats if f in df.columns]
    rows = []
    for f in feats:
        sub = df[[f, "wins"]].dropna()
        pear = sub[f].corr(sub["wins"], method="pearson")
        spear = sub[f].corr(sub["wins"], method="spearman")
        rows.append({"feature": f, "pearson_vs_wins": round(pear, 3),
                     "spearman_vs_wins": round(spear, 3), "n": len(sub)})
    out = pd.DataFrame(rows).sort_values("pearson_vs_wins",
                                         key=lambda s: s.abs(), ascending=False)
    return out


def champion_gap(df: pd.DataFrame) -> pd.DataFrame:
    lab = df[df["season_labeled"] == 1]
    feats = ["squad_strength", "true_strength", "net_rating", "off_rating",
             "def_rating", "wins"]
    feats = [f for f in feats if f in df.columns]
    rows = []
    for f in feats:
        champ = lab.loc[lab["is_champion"] == 1, f]
        rest = lab.loc[lab["is_champion"] == 0, f]
        rows.append({
            "feature": f,
            "media_campeon": round(champ.mean(), 3),
            "media_resto": round(rest.mean(), 3),
            "gap": round(champ.mean() - rest.mean(), 3),
            "pctil_medio_campeon": round(
                (rest.values[None, :] < champ.values[:, None]).mean() * 100, 1
            ),
        })
    return pd.DataFrame(rows)


def plot_strength_vs_wins(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    normal = df[df["is_champion"] == 0]
    champs = df[df["is_champion"] == 1]
    ax.scatter(normal["true_strength"], normal["wins"], s=28, alpha=0.5,
               color="#5B8DEF", label="Equipos")
    ax.scatter(champs["true_strength"], champs["wins"], s=120, marker="*",
               color="#E0A100", edgecolor="black", zorder=5, label="Campeon")
    # linea de tendencia
    sub = df[["true_strength", "wins"]].dropna()
    if len(sub) > 2:
        m, b = np.polyfit(sub["true_strength"], sub["wins"], 1)
        xs = np.linspace(sub["true_strength"].min(), sub["true_strength"].max(), 50)
        ax.plot(xs, m * xs + b, "--", color="gray", lw=1.5)
    r = sub["true_strength"].corr(sub["wins"])
    ax.set_xlabel("true_strength  (fuerza real del nucleo, APM)")
    ax.set_ylabel("Victorias regular season")
    ax.set_title(f"Fuerza real del nucleo vs. victorias   (r = {r:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not os.path.exists(config.UNIFIED_DATASET):
        raise FileNotFoundError(
            "Falta el dataset. Corre primero: python -m src.ingest.build_dataset"
        )
    df = add_labels(pd.read_csv(config.UNIFIED_DATASET))

    corr = correlations(df)
    gap = champion_gap(df)

    corr_path = os.path.join(config.TABLES_DIR, "eda_correlations.csv")
    gap_path = os.path.join(config.TABLES_DIR, "eda_champion_gap.csv")
    fig_path = os.path.join(config.FIGURES_DIR, "eda_strength_vs_wins.png")
    corr.to_csv(corr_path, index=False)
    gap.to_csv(gap_path, index=False)
    plot_strength_vs_wins(df, fig_path)

    print("=" * 64)
    print("CORRELACION DE FEATURES CON VICTORIAS (210 equipos-temporada)")
    print("=" * 64)
    print(corr.to_string(index=False))
    print("\n" + "=" * 64)
    print("CAMPEONES vs RESTO (6 temporadas etiquetadas 2019-20..2024-25)")
    print("=" * 64)
    print(gap.to_string(index=False))
    print(f"\n[eda] tablas -> {corr_path}")
    print(f"[eda]        -> {gap_path}")
    print(f"[eda] figura -> {fig_path}")


if __name__ == "__main__":
    main()
