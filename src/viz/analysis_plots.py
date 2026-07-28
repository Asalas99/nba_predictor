"""
Graficas de analisis: FUERZA de los equipos y PROXIMIDAD a los campeones.

  python -m src.viz.analysis_plots

Genera en outputs/figures/:
  strength_heatmap.png          squad_strength por equipo x temporada
  strength_ranking_2526.png     fuerza del plantel, temporada actual, ordenada
  champion_proximity_2526.png   parecido de estilo a los campeones (2025-26)
  contention_map_2526.png       fuerza vs proximidad (mapa de contendientes)

Tambien guarda data/processed/teams/combined/champion_proximity.csv
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
from src.data.champions import CHAMPIONS  # noqa: E402

PROC, FIG = config.PROCESSED_DIR, config.FIGURES_DIR
CURRENT = "2025-26"
STYLE_ID = ["SEASON", "TEAM_ID", "TEAM_NAME", "TEAM_STYLE_CLUSTER",
            "CLUSTER_NAME", "PLAYOFF_STATUS"]


def nickname(name):
    return {"Portland Trail Blazers": "Blazers"}.get(name, str(name).split()[-1])


def load():
    style = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_style_clusters.csv"))
    ss = pd.read_csv(os.path.join(PROC, "players", "combined", "squad_strength.csv"))
    feats = [c for c in style.columns if c not in STYLE_ID]
    return style, ss, feats


def champion_proximity(style, feats):
    """Distancia (euclidea) del estilo de cada equipo al CENTROIDE de campeones.
    Menor distancia = estilo mas parecido al de un campeon."""
    champ_mask = [CHAMPIONS.get(s) == t for s, t in zip(style["SEASON"], style["TEAM_NAME"])]
    centroid = style.loc[champ_mask, feats].mean().values
    d = np.sqrt(((style[feats].values - centroid) ** 2).sum(axis=1))
    out = style[["SEASON", "TEAM_ID", "TEAM_NAME"]].copy()
    out["champ_dist"] = d
    # similitud 0..1 (1 = identico al perfil campeon)
    out["champ_similarity"] = 1 - (d - d.min()) / (d.max() - d.min())
    return out


def plot_strength_heatmap(ss, path):
    tm = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_clean.csv"))[
        ["SEASON", "TEAM_ID", "TEAM_NAME"]]
    d = ss.merge(tm, on=["SEASON", "TEAM_ID"])
    d["nick"] = d["TEAM_NAME"].map(nickname)
    piv = d.pivot_table(index="nick", columns="SEASON", values="squad_strength")
    piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(10, 11))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("Fuerza del plantel (z-score por temporada)\nverde = fuerte, rojo = debil",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="squad_strength")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def plot_strength_ranking(ss, path):
    tm = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_clean.csv"))[
        ["SEASON", "TEAM_ID", "TEAM_NAME"]]
    d = ss.merge(tm, on=["SEASON", "TEAM_ID"])
    d = d[d["SEASON"] == CURRENT].sort_values("squad_strength")
    colors = ["#2FB380" if v > 0 else "#D65A7A" for v in d["squad_strength"]]
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(d["TEAM_NAME"], d["squad_strength"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("squad_strength (z-score dentro de la temporada)")
    ax.set_title(f"Fuerza del plantel — {CURRENT}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def plot_proximity_ranking(prox, style, path):
    cur = prox[prox["SEASON"] == CURRENT].merge(
        style[style["SEASON"] == CURRENT][["TEAM_ID", "CLUSTER_NAME"]], on="TEAM_ID")
    cur = cur.sort_values("champ_similarity", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(cur["TEAM_NAME"], cur["champ_similarity"], color="#E0A100")
    ax.set_xlabel("Similitud de estilo al perfil de campeon (1 = identico)")
    ax.set_title(f"¿Que equipos juegan mas 'como campeon'? — {CURRENT}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def plot_contention_map(ss, prox, path):
    d = ss[ss["SEASON"] == CURRENT].merge(
        prox[prox["SEASON"] == CURRENT], on=["SEASON", "TEAM_ID"])
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(d["squad_strength"], d["champ_similarity"], s=60, alpha=0.6,
               color="#5B8DEF")
    for _, r in d.iterrows():
        ax.annotate(nickname(r["TEAM_NAME"]), (r["squad_strength"], r["champ_similarity"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axvline(d["squad_strength"].median(), color="gray", ls="--", lw=0.8)
    ax.axhline(d["champ_similarity"].median(), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Fuerza del plantel (squad_strength)")
    ax.set_ylabel("Parecido de estilo a campeon")
    ax.set_title(f"Mapa de contendientes — {CURRENT}\n(arriba-derecha = fuerte Y con estilo de campeon)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def main():
    style, ss, feats = load()
    prox = champion_proximity(style, feats)
    prox.to_csv(os.path.join(PROC, "teams", "combined", "champion_proximity.csv"),
                index=False)

    plot_strength_heatmap(ss, os.path.join(FIG, "strength_heatmap.png"))
    plot_strength_ranking(ss, os.path.join(FIG, "strength_ranking_2526.png"))
    plot_proximity_ranking(prox, style, os.path.join(FIG, "champion_proximity_2526.png"))
    plot_contention_map(ss, prox, os.path.join(FIG, "contention_map_2526.png"))
    print("[viz] listo: 4 figuras de analisis")


if __name__ == "__main__":
    main()
