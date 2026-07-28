"""
Graficas de exploracion: donde cae cada equipo de cada temporada en los
espacios de ESTILO y de TIPO DE PLANTEL, con los equipos mas relevantes
(campeones / finalistas) etiquetados por nombre.

  python -m src.viz.plots

Genera en outputs/figures/:
  style_pca_labeled.png        estilo (PCA global) con campeones/finalistas
  style_pca_by_season.png      small-multiples: un panel por temporada
  roster_pca_labeled.png       tipo de plantel (PCA global) etiquetado
  champions_style_map.png      solo campeones a lo largo de los anios
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

PROC = config.PROCESSED_DIR
FIG = config.FIGURES_DIR
CLUSTER_COLORS = ["#5B8DEF", "#2FB380", "#E0763A", "#9B6FD1", "#D65A7A", "#4AAEC7"]
ID_COLS = ["SEASON", "TEAM_ID", "TEAM_NAME"]


def nickname(name: str) -> str:
    special = {"Trail Blazers": "Blazers"}
    for k, v in special.items():
        if name.endswith(k):
            return v
    return str(name).split()[-1]


def label_of(row) -> str:
    return f"{nickname(row['TEAM_NAME'])} '{str(row['SEASON'])[5:7]}"


def _load_style():
    df = pd.read_csv(os.path.join(PROC, "teams", "combined", "team_style_clusters.csv"))
    feats = [c for c in df.columns
             if c not in ID_COLS + ["TEAM_STYLE_CLUSTER", "CLUSTER_NAME", "PLAYOFF_STATUS"]]
    return df, feats


def _load_roster():
    df = pd.read_csv(os.path.join(PROC, "teams", "combined", "roster_type_clusters.csv"))
    feats = [c for c in df.columns if c.startswith("share_")]
    return df, feats


def _scatter_pca(df, feats, cluster_col, name_col, title, path,
                 label_status=("champion", "runner_up")):
    X = df[feats].values
    pca = PCA(n_components=2, random_state=config.SEED)
    xy = pca.fit_transform(X)
    df = df.copy()
    df["_x"], df["_y"] = xy[:, 0], xy[:, 1]
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, cl in enumerate(sorted(df[cluster_col].unique())):
        m = df[cluster_col] == cl
        nm = df.loc[m, name_col].iloc[0]
        ax.scatter(df.loc[m, "_x"], df.loc[m, "_y"], s=42, alpha=0.45,
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], label=nm)

    # resaltar y etiquetar los relevantes
    rel = df[df["PLAYOFF_STATUS"].isin(label_status)]
    champs = rel[rel["PLAYOFF_STATUS"] == "champion"]
    others = rel[rel["PLAYOFF_STATUS"] != "champion"]
    ax.scatter(others["_x"], others["_y"], s=70, facecolor="none",
               edgecolor="#444", linewidth=1.2, zorder=4)
    ax.scatter(champs["_x"], champs["_y"], marker="*", s=340, color="#E0A100",
               edgecolor="black", linewidth=0.8, zorder=6, label="Campeon")
    for _, r in rel.iterrows():
        weight = "bold" if r["PLAYOFF_STATUS"] == "champion" else "normal"
        ax.annotate(label_of(r), (r["_x"], r["_y"]),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=8.5, fontweight=weight, zorder=7)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")
    return pca


def style_by_season(df, feats, path):
    """Small-multiples: la MISMA proyeccion global, un panel por temporada."""
    X = df[feats].values
    pca = PCA(n_components=2, random_state=config.SEED)
    xy = pca.fit_transform(X)
    df = df.copy()
    df["_x"], df["_y"] = xy[:, 0], xy[:, 1]
    seasons = sorted(df["SEASON"].unique())
    ncol = 4
    nrow = int(np.ceil(len(seasons) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow),
                             sharex=True, sharey=True)
    axes = np.array(axes).ravel()
    xlim = (df["_x"].min() - 0.5, df["_x"].max() + 0.5)
    ylim = (df["_y"].min() - 0.5, df["_y"].max() + 0.5)
    for ax, season in zip(axes, seasons):
        ax.scatter(df["_x"], df["_y"], s=10, color="#DDD", zorder=1)  # fondo
        sd = df[df["SEASON"] == season]
        for i, cl in enumerate(sorted(sd["TEAM_STYLE_CLUSTER"].unique())):
            m = sd["TEAM_STYLE_CLUSTER"] == cl
            ax.scatter(sd.loc[m, "_x"], sd.loc[m, "_y"], s=34, alpha=0.75,
                       color=CLUSTER_COLORS[cl % len(CLUSTER_COLORS)], zorder=3)
        champ = sd[sd["PLAYOFF_STATUS"] == "champion"]
        for _, r in champ.iterrows():
            ax.scatter(r["_x"], r["_y"], marker="*", s=240, color="#E0A100",
                       edgecolor="black", zorder=6)
            ax.annotate(nickname(r["TEAM_NAME"]), (r["_x"], r["_y"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8, fontweight="bold", zorder=7)
        ax.set_title(season, fontsize=11, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.15)
    for ax in axes[len(seasons):]:
        ax.axis("off")
    fig.suptitle("Estilo de juego por temporada (proyeccion PCA global)  —  ★ = campeon",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def champions_map(df, feats, path):
    X = df[feats].values
    pca = PCA(n_components=2, random_state=config.SEED)
    xy = pca.fit_transform(X)
    df = df.copy()
    df["_x"], df["_y"] = xy[:, 0], xy[:, 1]
    champs = df[df["PLAYOFF_STATUS"] == "champion"]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(df["_x"], df["_y"], s=18, color="#E5E5E5", zorder=1, label="Todos los equipos")
    ax.scatter(champs["_x"], champs["_y"], marker="*", s=360, color="#E0A100",
               edgecolor="black", zorder=5, label="Campeon")
    for _, r in champs.iterrows():
        ax.annotate(label_of(r), (r["_x"], r["_y"]), xytext=(7, 5),
                    textcoords="offset points", fontsize=9.5, fontweight="bold")
    ax.set_title("¿Se parecen los campeones entre si? (espacio de estilo)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def main() -> None:
    style, sfeats = _load_style()
    roster, rfeats = _load_roster()

    _scatter_pca(style, sfeats, "CLUSTER_NAME", "CLUSTER_NAME",
                 "Estilo de juego NBA 2019-20 a 2025-26  (campeones y finalistas etiquetados)",
                 os.path.join(FIG, "style_pca_labeled.png"))
    style_by_season(style, sfeats, os.path.join(FIG, "style_pca_by_season.png"))
    champions_map(style, sfeats, os.path.join(FIG, "champions_style_map.png"))
    _scatter_pca(roster, rfeats, "ROSTER_TYPE_NAME", "ROSTER_TYPE_NAME",
                 "Tipo de plantel NBA  (campeones y finalistas etiquetados)",
                 os.path.join(FIG, "roster_pca_labeled.png"))
    print("[viz] listo: 4 figuras en outputs/figures/")


if __name__ == "__main__":
    main()
