"""
Arquetipos de jugador (roles funcionales) via KMeans sobre stats de estilo.

No usa la posicion oficial: agrupa por lo que el jugador HACE (uso, creacion,
rebote, volumen de triple, defensa, eficiencia).

  python -m src.features.player_roles

Lee : data/processed/players/combined/player_clean.csv
Crea:
  data/processed/players/combined/player_roles.csv         (jugador + ARCHETYPE)
  data/processed/players/combined/archetype_profiles.csv   (perfil medio por rol)
  data/processed/players/combined/role_kmeans.pkl + scaler.pkl
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

N_ARCHETYPES = 6
MIN_GP = 20  # minimo de partidos para clasificar (evita ruido de muestras chicas)


def build_features(pl: pd.DataFrame) -> pd.DataFrame:
    df = pl.copy()
    mins = df["MIN"].replace(0, np.nan)
    # tasas por 36 min (interpretables)
    for col in ["PTS", "AST", "REB", "STL", "BLK", "TOV", "FG3A"]:
        if col in df.columns:
            df[f"{col}_per36"] = df[col] / mins * 36
    # features de rol
    feats = ["PTS_per36", "AST_per36", "REB_per36", "STL_per36", "BLK_per36",
             "FG3A_per36", "USG_PCT", "AST_PCT", "REB_PCT", "TS_PCT"]
    feats = [f for f in feats if f in df.columns]
    df[feats] = df[feats].fillna(0)
    return df, feats


def name_archetypes(profiles: pd.DataFrame) -> dict:
    """Etiqueta cada cluster por su rasgo dominante (z-score sobre el perfil)."""
    z = (profiles - profiles.mean()) / profiles.std(ddof=0).replace(0, 1)
    names = {}
    for cl in z.index:
        row = z.loc[cl]
        # reglas simples por rasgo dominante
        if row.get("BLK_per36", 0) > 0.7 or row.get("REB_PCT", 0) > 0.9:
            nm = "Interior / protector de aro"
        elif row.get("AST_PCT", 0) > 0.8:
            nm = "Creador / base"
        elif row.get("USG_PCT", 0) > 0.8 and row.get("PTS_per36", 0) > 0.5:
            nm = "Anotador principal"
        elif row.get("FG3A_per36", 0) > 0.6 and row.get("TS_PCT", 0) > 0.2:
            nm = "Tirador 3&D"
        elif row.get("PTS_per36", 0) < -0.4 and row.get("USG_PCT", 0) < -0.3:
            nm = "Rol / bajo uso"
        else:
            nm = "Wing versatil"
        names[cl] = nm
    # desambiguar repetidos
    seen = {}
    for cl, nm in list(names.items()):
        base = nm
        k = 2
        while names[cl] in seen:
            names[cl] = f"{base} {k}"
            k += 1
        seen[names[cl]] = cl
    return names


def main() -> None:
    path = os.path.join(config.PROCESSED_DIR, "players", "combined", "player_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta {path}. Corre clean_players primero.")
    pl = pd.read_csv(path)
    df, feats = build_features(pl)

    mask = df["GP"] >= MIN_GP if "GP" in df.columns else np.ones(len(df), bool)
    train = df[mask].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(train[feats])
    km = KMeans(n_clusters=N_ARCHETYPES, random_state=config.SEED, n_init=10)
    train["ARCHETYPE_ID"] = km.fit_predict(X)

    # perfil medio (en unidades reales) + nombres
    profiles = train.groupby("ARCHETYPE_ID")[feats].mean()
    names = name_archetypes(profiles)
    train["ARCHETYPE"] = train["ARCHETYPE_ID"].map(names)
    profiles_named = profiles.copy()
    profiles_named["ARCHETYPE"] = [names[i] for i in profiles_named.index]

    out_dir = os.path.join(config.PROCESSED_DIR, "players", "combined")
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON", "MIN", "GP",
            "ARCHETYPE_ID", "ARCHETYPE"] + feats
    train[[c for c in keep if c in train.columns]].to_csv(
        os.path.join(out_dir, "player_roles.csv"), index=False)
    profiles_named.round(2).to_csv(os.path.join(out_dir, "archetype_profiles.csv"))
    joblib.dump(km, os.path.join(out_dir, "role_kmeans.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "role_scaler.pkl"))

    print(f"[roles] {len(train)} jugadores clasificados (GP>={MIN_GP}), "
          f"{N_ARCHETYPES} arquetipos")
    print("[roles] perfil medio por arquetipo (per36 / %):")
    print(profiles_named.round(1).to_string())
    print("[roles] jugadores por arquetipo:")
    print(train["ARCHETYPE"].value_counts().to_string())


if __name__ == "__main__":
    main()
