"""
Cluster de TIPO DE PLANTEL (construccion de roster).

Representa a cada equipo por su composicion de arquetipos (peso = minutos) y
agrupa equipos que se construyen parecido. Responde: ¿que tipos de plantilla
existen y cuales ganan?

  python -m src.models.roster_type_clustering

Lee : data/processed/players/combined/player_roles.csv
Crea:
  data/processed/teams/combined/roster_type_clusters.csv
      (SEASON, TEAM_ID, share_<arquetipo>..., ROSTER_TYPE, ROSTER_TYPE_NAME, PLAYOFF_STATUS)
  data/processed/teams/combined/roster_kmeans.pkl
  outputs/figures/roster_types_pca.png
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

N_ROSTER_TYPES = 4


def composition(roles: pd.DataFrame) -> pd.DataFrame:
    """Fraccion de minutos de cada equipo-temporada en cada arquetipo."""
    r = roles.copy()
    r["w"] = r["MIN"]
    tot = r.groupby(["SEASON", "TEAM_ID"])["w"].transform("sum")
    r["w"] = r["w"] / tot
    comp = (r.groupby(["SEASON", "TEAM_ID", "ARCHETYPE"])["w"].sum()
              .unstack(fill_value=0.0))
    comp.columns = [f"share_{c}" for c in comp.columns]
    return comp.reset_index()


def name_types(profiles: pd.DataFrame) -> dict:
    """Nombra cada tipo de plantel por los 2 arquetipos que mas lo distinguen."""
    z = (profiles - profiles.mean()) / profiles.std(ddof=0).replace(0, 1)
    names = {}
    for cl in z.index:
        top2 = z.loc[cl].sort_values(ascending=False).head(2).index
        pretty = [c.replace("share_", "") for c in top2]
        names[cl] = "Alto en: " + " + ".join(pretty)
    return names


def main() -> None:
    roles_path = os.path.join(config.PROCESSED_DIR, "players", "combined",
                              "player_roles.csv")
    if not os.path.exists(roles_path):
        raise FileNotFoundError(f"Falta {roles_path}. Corre player_roles primero.")
    roles = pd.read_csv(roles_path)
    comp = composition(roles)

    share_cols = [c for c in comp.columns if c.startswith("share_")]
    X = comp[share_cols].values
    km = KMeans(n_clusters=N_ROSTER_TYPES, random_state=config.SEED, n_init=10)
    comp["ROSTER_TYPE"] = km.fit_predict(X)

    profiles = comp.groupby("ROSTER_TYPE")[share_cols].mean()
    names = name_types(profiles)
    comp["ROSTER_TYPE_NAME"] = comp["ROSTER_TYPE"].map(names)

    # nombres de equipo + status de playoff (join con team_clean)
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))[["SEASON", "TEAM_ID", "TEAM_NAME"]]
    comp = comp.merge(tm, on=["SEASON", "TEAM_ID"], how="left")
    comp["PLAYOFF_STATUS"] = [playoff_status(t, s)
                             for t, s in zip(comp["TEAM_NAME"], comp["SEASON"])]

    out_dir = os.path.join(config.PROCESSED_DIR, "teams", "combined")
    comp.to_csv(os.path.join(out_dir, "roster_type_clusters.csv"), index=False)
    joblib.dump(km, os.path.join(out_dir, "roster_kmeans.pkl"))

    # figura PCA
    pca = PCA(n_components=2, random_state=config.SEED)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cl in sorted(comp["ROSTER_TYPE"].unique()):
        m = comp["ROSTER_TYPE"].values == cl
        ax.scatter(xy[m, 0], xy[m, 1], s=30, alpha=0.5, label=names[cl])
    champ = comp["PLAYOFF_STATUS"] == "champion"
    ax.scatter(xy[champ.values, 0], xy[champ.values, 1], marker="*", s=220,
               color="#E0A100", edgecolor="black", zorder=5, label="Campeon")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"Tipos de plantel NBA (k={N_ROSTER_TYPES})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "roster_types_pca.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[roster] {len(comp)} equipos-temporada, k={N_ROSTER_TYPES}")
    print("[roster] composicion media por tipo (fraccion de minutos):")
    prof = profiles.round(2).copy()
    prof.columns = [c.replace("share_", "") for c in prof.columns]
    prof["NOMBRE"] = [names[i] for i in prof.index]
    print(prof.to_string())
    # tasa de campeones/finalistas por tipo
    lab = comp[comp["PLAYOFF_STATUS"] != "none"]
    print("\n[roster] status de playoff por tipo de plantel:")
    print(pd.crosstab(comp["ROSTER_TYPE_NAME"], comp["PLAYOFF_STATUS"]).to_string())


if __name__ == "__main__":
    main()
