"""
Clustering de estilo de juego (KMeans, k=3) — portado y hecho self-contained
desde nba_clustering_comp. No interactivo (k fijo en config.STYLE_K).

  python -m src.models.style_clustering

Lee : data/processed/teams/combined/team_cluster_input.csv
Crea:
  data/processed/teams/combined/team_style_clusters.csv  (estilo + cluster + nombre + status playoff)
  data/processed/teams/combined/kmeans_model.pkl
  outputs/figures/style_clusters_pca.png
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from src.data.champions import playoff_status  # noqa: E402

ID_COLS = ["SEASON", "TEAM_ID", "TEAM_NAME"]


def name_clusters(profiles: pd.DataFrame) -> dict:
    """Nombres interpretables a partir del perfil medio (z-scores) de cada cluster.

    Heuristica compacta: mira ofensiva (OFF_RATING), defensa (DEF_RATING, mas
    bajo = mejor) y ritmo (PACE).
    """
    names = {}
    for cl, row in profiles.iterrows():
        off = row.get("OFF_RATING", 0)
        deff = row.get("DEF_RATING", 0)
        pace = row.get("PACE", 0)
        tags = []
        if deff < -0.4:
            tags.append("Defensivos")
        elif off > 0.4:
            tags.append("Ofensivos")
        else:
            tags.append("Balanceados")
        if pace > 0.4:
            tags.append("rapidos")
        elif pace < -0.4:
            tags.append("lentos")
        names[cl] = " ".join(tags) if tags else f"Cluster {cl}"
    # desambigua nombres repetidos
    seen = {}
    for cl, nm in list(names.items()):
        if nm in seen.values():
            names[cl] = f"{nm} ({cl})"
        seen[cl] = names[cl]
    return names


def main() -> None:
    inp = os.path.join(config.PROCESSED_DIR, "teams", "combined",
                       "team_cluster_input.csv")
    if not os.path.exists(inp):
        raise FileNotFoundError(
            f"Falta {inp}. Corre `python -m src.features.clean_teams` primero.")
    df = pd.read_csv(inp)
    feats = [c for c in df.columns if c not in ID_COLS]
    X = df[feats].values

    km = KMeans(n_clusters=config.STYLE_K, random_state=config.SEED, n_init=10)
    labels = km.fit_predict(X)
    df["TEAM_STYLE_CLUSTER"] = labels

    profiles = df.groupby("TEAM_STYLE_CLUSTER")[feats].mean()
    names = name_clusters(profiles)
    df["CLUSTER_NAME"] = df["TEAM_STYLE_CLUSTER"].map(names)

    # status de playoff (para colorear campeones)
    df["PLAYOFF_STATUS"] = [playoff_status(t, s)
                            for t, s in zip(df["TEAM_NAME"], df["SEASON"])]

    out_dir = os.path.join(config.PROCESSED_DIR, "teams", "combined")
    out_csv = os.path.join(out_dir, "team_style_clusters.csv")
    df.to_csv(out_csv, index=False)
    joblib.dump(km, os.path.join(out_dir, "kmeans_model.pkl"))

    # figura PCA
    pca = PCA(n_components=2, random_state=config.SEED)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cl in sorted(df["TEAM_STYLE_CLUSTER"].unique()):
        m = labels == cl
        ax.scatter(xy[m, 0], xy[m, 1], s=30, alpha=0.5, label=names[cl])
    champ = df["PLAYOFF_STATUS"] == "champion"
    ax.scatter(xy[champ.values, 0], xy[champ.values, 1], marker="*", s=220,
               color="#E0A100", edgecolor="black", zorder=5, label="Campeon")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"Clusters de estilo NBA (k={config.STYLE_K})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "style_clusters_pca.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[cluster] {len(df)} equipos-temporada, k={config.STYLE_K}")
    print("[cluster] perfiles (z-score medio por cluster):")
    prof_show = profiles.round(2).copy()
    prof_show["NOMBRE"] = [names[c] for c in prof_show.index]
    print(prof_show.to_string())
    print(f"[cluster] -> {out_csv}")
    print(f"[cluster] -> kmeans_model.pkl + outputs/figures/style_clusters_pca.png")


if __name__ == "__main__":
    main()
