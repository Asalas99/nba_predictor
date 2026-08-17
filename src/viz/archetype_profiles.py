"""
Describe los 6 arquetipos de jugador: en que estadistica resalta cada uno.

  python -m src.features.player_roles   (antes, para tener player_roles.csv)
  python -m src.viz.archetype_profiles

Genera:
  outputs/figures/archetype_profiles_heatmap.png   arquetipo x estadistica (z-score)
  outputs/tables/archetype_descriptions.csv        descripcion + stat destacada + ejemplos
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

PROC, FIG, TAB = config.PROCESSED_DIR, config.FIGURES_DIR, config.TABLES_DIR

FEATURES = ["PTS_per36", "AST_per36", "REB_per36", "STL_per36", "BLK_per36",
            "FG3A_per36", "USG_PCT", "AST_PCT", "REB_PCT", "TS_PCT"]
NICE = {"PTS_per36": "Puntos/36", "AST_per36": "Asist/36", "REB_per36": "Reb/36",
        "STL_per36": "Robos/36", "BLK_per36": "Tapones/36", "FG3A_per36": "Triples int/36",
        "USG_PCT": "Uso %", "AST_PCT": "Asist %", "REB_PCT": "Reb %", "TS_PCT": "TS %"}

# descripcion en lenguaje claro por arquetipo (segun su perfil)
DESCRIPTIONS = {
    "Creador / base": "Estrella con balon: mucho volumen de anotacion y creacion, el "
                      "uso mas alto del equipo. Es la primera opcion ofensiva.",
    "Creador / base 2": "Base/guard secundario: reparte y presiona el balon (asistencias "
                        "y robos altos), con anotacion moderada.",
    "Interior / protector de aro": "Interior anotador: rebotea y tapona, pero tambien "
                                   "suma puntos cerca del aro. Poste ofensivo-defensivo.",
    "Interior / protector de aro 2": "Interior defensivo clasico: maximo rebote y tapones, "
                                     "casi no tira de tres. Ancla la defensa en la pintura.",
    "Rol / bajo uso": "Jugador de rol: bajo uso y pocos puntos, aporta minutos y esfuerzo "
                      "sin protagonismo ofensivo.",
    "Wing versatil": "Alero tirador (3&D): mucho volumen de triple con eficiencia, "
                     "anotacion perimetral. Espaciamiento y defensa en las alas.",
}


def main():
    roles = pd.read_csv(os.path.join(PROC, "players", "combined", "player_roles.csv"))
    feats = [f for f in FEATURES if f in roles.columns]

    # perfil medio (unidades reales) por arquetipo
    profiles = roles.groupby("ARCHETYPE")[feats].mean()
    counts = roles["ARCHETYPE"].value_counts()

    # z-score de cada estadistica ENTRE arquetipos (para ver en que resalta)
    z = (profiles - profiles.mean()) / profiles.std(ddof=0).replace(0, 1)

    # stat destacada por arquetipo: la de mayor z. Si ninguna resalta al alza
    # (perfil bajo en todo), se marca la mas caracteristica por lo BAJO.
    def standout_of(row):
        top = row.idxmax()
        if row[top] >= 0.5:
            return NICE[top]
        low = row.idxmin()
        return f"{NICE[low]} (bajo)"
    standout = z[feats].apply(standout_of, axis=1)

    # jugadores de ejemplo: top-3 por minutos en cada arquetipo (nombres unicos)
    examples = {}
    for arch, g in roles.sort_values("MIN", ascending=False).groupby("ARCHETYPE"):
        names = []
        for n in g["PLAYER_NAME"]:
            if n not in names:
                names.append(n)
            if len(names) == 3:
                break
        examples[arch] = ", ".join(names)

    # ---- tabla descripcion ----
    order = list(profiles.index)
    rows = []
    for arch in order:
        rows.append(dict(
            arquetipo=arch,
            jugadores=int(counts[arch]),
            stat_destacada=standout[arch],
            descripcion=DESCRIPTIONS.get(arch, ""),
            ejemplos=examples.get(arch, ""),
        ))
    desc = pd.DataFrame(rows)
    desc.to_csv(os.path.join(TAB, "archetype_descriptions.csv"), index=False)

    # ---- heatmap ----
    zmat = z[feats].values
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(zmat, cmap="RdBu_r", aspect="auto", vmin=-1.8, vmax=1.8)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([NICE[f] for f in feats], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{a}\n(n={counts[a]})" for a in order], fontsize=8.5)
    for i in range(len(order)):
        for j in range(len(feats)):
            v = zmat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 1.1 else "black")
    ax.set_title("Perfil de los 6 arquetipos de jugador\n"
                 "z-score entre arquetipos: rojo = resalta por ARRIBA, azul = por debajo",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z-score (vs. promedio de arquetipos)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "archetype_profiles_heatmap.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("Arquetipos y estadistica en la que resaltan:")
    print(desc[["arquetipo", "jugadores", "stat_destacada"]].to_string(index=False))
    print(f"\n[viz] -> outputs/figures/archetype_profiles_heatmap.png")
    print(f"[viz] -> outputs/tables/archetype_descriptions.csv")


if __name__ == "__main__":
    main()
