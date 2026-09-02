"""
MODO PREDICCION 2026-27 (temporada por empezar, sin estadisticas todavia).

Proyecta la fuerza de cada plantilla desde el roster + el historial de cada
jugador, corre la cascada M1 (victorias) -> M2 (seeding) -> M3 (playoffs) ->
M4 (campeon), y ademas M1 sobre el calendario ya publicado.

  python -m src.predict.predict_2026

Salidas en outputs/prediccion_2026-27/:
  tabla_2026-27.csv                 M1-M4 por equipo (wins, seed, playoffs, titulo)
  standings_este.png / _oeste.png   clasificacion proyectada por conferencia
  odds_titulo.png                    probabilidad de campeon
  calendario/game_predictions.csv   probabilidad de cada partido publicado
  calendario/<equipo>.png           calendario de cada uno de los 30 equipos
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
from src.features import player_projection as pp  # noqa: E402
from src.models import m1_wins, m2_seeding, m3_playoffs  # noqa: E402
from src.data.conferences import conference  # noqa: E402

TARGET = "2026-27"
PREV = "2025-26"
OUT = os.path.join(config.OUTPUTS_DIR, "prediccion_2026-27")
CAL = os.path.join(OUT, "calendario")
ROTATION = 10   # jugadores de rotacion proyectada por equipo
HCA = 0.06


def nick(name):
    return {"Portland Trail Blazers": "Blazers"}.get(name, str(name).split()[-1])


# ---------------------------------------------------------------- proyeccion
def project_players():
    roster = pd.read_csv(os.path.join(config.RAW_DIR, "rosters", TARGET, "roster.csv"))
    pc = pd.read_csv(os.path.join(config.PROCESSED_DIR, "players", "combined",
                                  "player_clean.csv"))
    pc["yr"] = pc["SEASON"].map(pp.year_of)
    hist = pc[pc["yr"] < pp.year_of(TARGET)]
    curve = pp.build_data_curve(hist)
    prior = float(hist["PIE"].mean())

    rows = []
    for r in roster.itertuples():
        past = hist[hist["PLAYER_ID"] == r.PLAYER_ID]
        age_t = float(r.AGE)
        if len(past) == 0:                      # rookie / sin historia NBA
            rows.append(dict(TEAM_ID=r.TEAM_ID, PLAYER_ID=r.PLAYER_ID,
                             PLAYER_NAME=r.PLAYER_NAME, age=age_t,
                             pie=prior, min_proj=13.0, rookie=1))
            continue
        base_pie, base_min = pp.recency_baseline(past, prior)
        ps = past.sort_values("yr")
        last_age = int(ps["AGE"].iloc[-1])
        pie_last = float(ps["PIE"].iloc[-1])
        pie_std = base_pie * pp.std_age_mult(age_t) / max(pp.std_age_mult(last_age), 1e-6)
        pies = ps["PIE"].values
        yoy = float(np.clip(pies[-1] - pies[-2], -pp.TRAJ_CAP, pp.TRAJ_CAP)) if len(pies) >= 2 else 0.0
        alpha = float(np.clip((pp.TRAJ_AGE - age_t) / pp.TRAJ_SPAN, 0.0, pp.TRAJ_MAXW))
        pie_traj = alpha * (pie_last + yoy) + (1 - alpha) * pie_std
        min_proj = float(np.clip(base_min * (0.97 if age_t > 30 else 1.0), 0, pp.CAP_MIN))
        rows.append(dict(TEAM_ID=r.TEAM_ID, PLAYER_ID=r.PLAYER_ID,
                         PLAYER_NAME=r.PLAYER_NAME, age=age_t,
                         pie=round(pie_traj, 4), min_proj=round(min_proj, 1), rookie=0))
    return pd.DataFrame(rows), pc


def team_features(players, pc):
    prev_team = (pc[pc["SEASON"] == PREV].set_index("PLAYER_ID")["TEAM_ID"].to_dict())
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))
    names = tm[tm.SEASON == PREV].set_index("TEAM_ID")["TEAM_NAME"].to_dict()
    prev_nr = tm[tm.SEASON == PREV].set_index("TEAM_ID")["NET_RATING"].to_dict()

    rows = []
    for team, g in players.groupby("TEAM_ID"):
        g = g.sort_values("min_proj", ascending=False)
        rot = g.head(ROTATION)
        w = rot["min_proj"] / rot["min_proj"].sum()
        core = g.head(8)
        wc = core["min_proj"] / core["min_proj"].sum()
        ret_mask = np.array([prev_team.get(p) == team for p in rot["PLAYER_ID"]])
        rows.append(dict(
            TEAM_ID=team, TEAM_NAME=names.get(team, str(team)),
            conf=conference(names.get(team, "")),
            pie_wmean_proj=float((rot["pie"] * w).sum()),
            best_pie_proj=float(rot["pie"].max()),
            avg_age_core=float((core["age"] * wc).sum()),
            continuity=float(rot["min_proj"][ret_mask].sum() / rot["min_proj"].sum()),
            prior_net_rating=float(prev_nr.get(team, 0.0)),
        ))
    df = pd.DataFrame(rows)
    df["squad_strength_proj"] = (df["pie_wmean_proj"] - df["pie_wmean_proj"].mean()) / \
        df["pie_wmean_proj"].std(ddof=0)
    return df


# ---------------------------------------------------------------- cascada
def run_cascade(feat):
    F = m1_wins.FEATURES
    hist = m1_wins.assemble().dropna(subset=F + ["wins"])
    sc = StandardScaler().fit(hist[F])
    model = Ridge(alpha=1.0).fit(sc.transform(hist[F]), hist["wins"])
    feat = feat.copy()
    raw = model.predict(sc.transform(feat[F]))
    # Normalizacion de liga: cada partido tiene un ganador, asi que el total de
    # victorias debe ser 30*41=1230 (promedio 41). M1 predice cada equipo por
    # separado y se infla; se desplaza al promedio correcto (conserva las
    # diferencias entre equipos, arregla la suma).
    raw = raw - raw.mean() + 41.0
    feat["wins_pred"] = np.clip(raw, 15, 70).round(1)

    # M2 seeding por conferencia
    feat["seed_pred"] = (feat.groupby("conf")["wins_pred"]
                         .rank(ascending=False, method="first").astype(int))
    feat["tramo"] = feat["seed_pred"].map(m2_seeding.bucket)

    # M3 playoffs (Monte Carlo)
    feat["q"] = np.clip(feat["wins_pred"] / 82.0, 0.05, 0.95)
    probs = m3_playoffs.sim_season(feat[["TEAM_ID", "TEAM_NAME", "conf", "seed_pred", "q"]])
    feat = feat.merge(probs, on="TEAM_ID", how="left").fillna({"p_champion": 0,
                      "p_finals": 0, "p_conf_final": 0, "p_round2": 0})
    # M4 = M3 (la calibracion eligio w=0)
    feat["p_titulo"] = feat["p_champion"]
    return feat


# ---------------------------------------------------------------- figuras
def fig_standings(feat, conf, path):
    d = feat[feat.conf == conf].sort_values("wins_pred", ascending=True)
    colors = {"Playoffs": "#2FB380", "Play-in": "#E0A100", "Loteria": "#D65A7A"}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(d["TEAM_NAME"], d["wins_pred"], color=[colors[t] for t in d["tramo"]])
    for i, (w, s) in enumerate(zip(d["wins_pred"], d["seed_pred"])):
        ax.text(w + 0.4, i, f"#{s}  {w:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Victorias proyectadas")
    ax.set_title(f"Clasificacion proyectada {TARGET} — {conf}\n"
                 f"verde=Playoffs, amarillo=Play-in, rojo=Loteria",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 70)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_odds(feat, path):
    d = feat.nlargest(12, "p_titulo")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["TEAM_NAME"], d["p_titulo"] * 100, color="#E0A100")
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad de campeon (%)")
    ax.set_title(f"Odds de titulo {TARGET} (preseason)", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- calendario
def log5(qa, qb):
    d = qa + qb - 2 * qa * qb
    return 0.5 if d <= 0 else (qa - qa * qb) / d


def calendar(feat):
    path = os.path.join(config.RAW_DIR, "schedule", TARGET, "schedule.csv")
    if not os.path.exists(path):
        print("[pred] no hay calendario 2026-27 aun; se omite la parte de partidos.")
        return None
    sch = pd.read_csv(path)
    q = feat.set_index("TEAM_ID")["q"].to_dict()
    nm = feat.set_index("TEAM_ID")["TEAM_NAME"].to_dict()
    rows = []
    for r in sch.itertuples():
        qh, qa = q.get(r.home_team_id), q.get(r.away_team_id)
        if qh is None or qa is None:
            continue
        p_home = float(np.clip(log5(qh, qa) + HCA, 0.02, 0.98))
        rows.append(dict(GAME_DATE=r.GAME_DATE,
                         home=nm.get(r.home_team_id), away=nm.get(r.away_team_id),
                         home_id=r.home_team_id, away_id=r.away_team_id,
                         p_home_win=round(p_home, 3)))
    games = pd.DataFrame(rows)
    os.makedirs(CAL, exist_ok=True)
    games.to_csv(os.path.join(CAL, "game_predictions.csv"), index=False)

    # una figura de calendario por equipo (los 30)
    for team, name in nm.items():
        g = games[(games.home_id == team) | (games.away_id == team)].sort_values("GAME_DATE")
        if g.empty:
            continue
        probs, labels = [], []
        for r in g.itertuples():
            fecha = str(r.GAME_DATE)[:10]  # MM/DD/YYYY
            fecha = "/".join(fecha.split("/")[:2]) if "/" in fecha else fecha
            if r.home_id == team:
                probs.append(r.p_home_win); labels.append(f"{fecha}\nvs {nick(r.away)}")
            else:
                probs.append(1 - r.p_home_win); labels.append(f"{fecha}\n@ {nick(r.home)}")
        n = len(probs)
        exp_wins = sum(probs)
        fig, ax = plt.subplots(figsize=(max(0.55 * n + 2, 4.5), 3.6))
        colors = ["#2FB380" if p >= 0.5 else "#D65A7A" for p in probs]
        ax.bar(range(n), probs, color=colors)
        ax.axhline(0.5, color="gray", ls="--", lw=0.7)
        for i, p in enumerate(probs):
            ax.text(i, p + 0.02, f"{p*100:.0f}", ha="center", fontsize=6)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=6.5)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Prob. de ganar")
        ax.set_title(f"{name} — {n} de 82 partidos publicados  (verde=favorito; "
                     f"esperadas: {exp_wins:.1f} de estos {n})",
                     fontsize=9.5, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(CAL, f"{nick(name)}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
    return games


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[pred] proyectando {TARGET} desde rosters...")
    players, pc = project_players()
    feat = team_features(players, pc)
    feat = run_cascade(feat)

    cols = ["TEAM_NAME", "conf", "wins_pred", "seed_pred", "tramo",
            "p_titulo", "p_finals", "squad_strength_proj", "best_pie_proj",
            "avg_age_core", "continuity", "prior_net_rating"]
    out = feat[cols].sort_values(["conf", "wins_pred"], ascending=[True, False])
    out.to_csv(os.path.join(OUT, "tabla_2026-27.csv"), index=False)

    fig_standings(feat, "East", os.path.join(OUT, "standings_este.png"))
    fig_standings(feat, "West", os.path.join(OUT, "standings_oeste.png"))
    fig_odds(feat, os.path.join(OUT, "odds_titulo.png"))
    games = calendar(feat)

    print("\n=== PREDICCION 2026-27: favoritos al titulo ===")
    print(feat.nlargest(8, "p_titulo")[["TEAM_NAME", "wins_pred", "seed_pred",
          "p_titulo"]].to_string(index=False))
    print(f"\n[pred] tabla -> {os.path.join(OUT, 'tabla_2026-27.csv')}")
    if games is not None:
        print(f"[pred] calendario: {len(games)} partidos, figuras por equipo en {CAL}")


if __name__ == "__main__":
    main()
