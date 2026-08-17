"""
Backtest de M1 SOBRE EL CALENDARIO (game by game), sin sesgo.

Idea: en vez de predecir el total de victorias de golpe, se predice la
probabilidad de ganar CADA partido del calendario y se suman. La fuerza de cada
equipo sale de M1 (wins_pred), que es leakage-free (entrenado solo con
temporadas anteriores). Del calendario solo se usa quien juega contra quien y
donde (local/visitante) — informacion conocida antes de empezar, sin trampa.

  python -m src.models.m1_schedule

Requiere game logs:  python -m src.ingest.download_gamelogs  (en tu maquina)

Prob de un partido:  log5(q_local, q_visita) +/- ventaja de local.
q = wins_pred / 82  (recortado).

Crea:
  outputs/tables/m1_schedule_backtest.csv     por temporada: MAE y acierto
  outputs/tables/m1_game_predictions.csv       cada partido con su probabilidad
  outputs/figures/m1_schedule_expected_vs_actual.png
  outputs/figures/m1_calendar_<equipo>_<season>.png   calendario de un equipo
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

HCA = 0.06
N_SIMS = 10000            # simulaciones de temporada completa (Monte Carlo)
SIGMA_STRENGTH = 0.09     # incertidumbre de la fuerza del equipo (calibrado: ~79% cobertura)


def log5(qa, qb):
    d = qa + qb - 2 * qa * qb
    return 0.5 if d <= 0 else (qa - qa * qb) / d


def load_gamelogs() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(config.RAW_DIR, "gamelogs", "*", "team_gamelog.csv")))
    if not paths:
        raise FileNotFoundError(
            "No hay game logs. Corre en tu maquina:\n"
            "  python -m src.ingest.download_gamelogs --start 2021 --end 2025")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def build_games(gl: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Empareja las 2 filas de cada partido y calcula prob de victoria del local."""
    gl = gl.copy()
    gl["home"] = gl["MATCHUP"].str.contains("vs.", regex=False)
    pairs = gl.merge(gl, on=["GAME_ID", "SEASON"], suffixes=("", "_o"))
    pairs = pairs[pairs["TEAM_ID"] != pairs["TEAM_ID_o"]]
    home = pairs[pairs["home"]].copy()  # una fila por partido (la del local)

    home = home.merge(q.rename(columns={"TEAM_ID": "TEAM_ID", "q": "q_home"}),
                      on=["SEASON", "TEAM_ID"], how="inner")
    home = home.merge(q.rename(columns={"TEAM_ID": "TEAM_ID_o", "q": "q_away"}),
                      on=["SEASON", "TEAM_ID_o"], how="inner")

    base = home.apply(lambda r: log5(r["q_home"], r["q_away"]), axis=1)
    home["p_home_win"] = np.clip(base + HCA, 0.02, 0.98)
    home["home_won"] = (home["WL"] == "W").astype(int)
    return home.rename(columns={"TEAM_NAME": "home_team", "TEAM_NAME_o": "away_team"})


def backtest(games: pd.DataFrame, q: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    # victorias esperadas por equipo = suma de prob de ganar sus partidos
    rows = []
    for season, g in games.groupby("SEASON"):
        # perspectiva de cada equipo: local (p_home_win) y visita (1-p_home_win)
        exp = {}
        act = {}
        for _, r in g.iterrows():
            exp[r["TEAM_ID"]] = exp.get(r["TEAM_ID"], 0) + r["p_home_win"]
            exp[r["TEAM_ID_o"]] = exp.get(r["TEAM_ID_o"], 0) + (1 - r["p_home_win"])
            act[r["TEAM_ID"]] = act.get(r["TEAM_ID"], 0) + r["home_won"]
            act[r["TEAM_ID_o"]] = act.get(r["TEAM_ID_o"], 0) + (1 - r["home_won"])
        e = pd.DataFrame({"TEAM_ID": list(exp), "exp_wins": list(exp.values())})
        e["act_wins"] = e["TEAM_ID"].map(act)
        e = e.merge(m1[m1.SEASON == season][["TEAM_ID", "wins_pred"]], on="TEAM_ID", how="left")
        mae_sched = (e["exp_wins"] - e["act_wins"]).abs().mean()
        mae_m1 = (e["wins_pred"] - e["act_wins"]).abs().mean()
        # acierto por partido (predice gana local si p>0.5)
        acc = ((g["p_home_win"] > 0.5).astype(int) == g["home_won"]).mean()
        rows.append(dict(season=season, n_juegos=len(g),
                         MAE_calendario=round(mae_sched, 2),
                         MAE_M1_directo=round(mae_m1, 2),
                         acierto_partido=round(acc, 3)))
    return pd.DataFrame(rows)


def monte_carlo(games: pd.DataFrame, n_sims: int = N_SIMS,
                sigma_strength: float = SIGMA_STRENGTH) -> pd.DataFrame:
    """Simula la temporada completa n_sims veces. Dos fuentes de incertidumbre:
      1) suerte de cada partido  (Bernoulli(p))
      2) incertidumbre de la FUERZA: en cada simulacion, cada equipo puede ser
         'secretamente' mejor o peor que lo proyectado (shift ~ Normal(0, sigma)),
         lo que corre TODOS sus partidos a la vez -> captura las sorpresas.
    Devuelve por equipo-temporada: victorias media, sd y banda 10-90%."""
    rng = np.random.default_rng(config.SEED)
    rows = []
    for season, g in games.groupby("SEASON"):
        teams = sorted(set(g["TEAM_ID"]) | set(g["TEAM_ID_o"]))
        idx = {t: i for i, t in enumerate(teams)}
        shift = rng.normal(0, sigma_strength, size=(n_sims, len(teams)))
        wins = np.zeros((n_sims, len(teams)))
        for _, r in g.iterrows():
            hi, ai = idx[r["TEAM_ID"]], idx[r["TEAM_ID_o"]]
            p = np.clip(r["p_home_win"] + shift[:, hi] - shift[:, ai], 0.02, 0.98)
            home_win = rng.random(n_sims) < p
            wins[:, hi] += home_win
            wins[:, ai] += ~home_win
        for t, i in idx.items():
            col = wins[:, i]
            rows.append(dict(SEASON=season, TEAM_ID=t,
                             sim_mean=col.mean(), sim_sd=col.std(),
                             sim_p10=np.percentile(col, 10),
                             sim_p90=np.percentile(col, 90)))
    return pd.DataFrame(rows)


def montecarlo_fig(mc, games, season, path):
    """Barras de victorias proyectadas con banda de incertidumbre vs reales."""
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))[["SEASON", "TEAM_ID", "TEAM_NAME", "W"]]
    d = mc[mc.SEASON == season].merge(tm, on=["SEASON", "TEAM_ID"]).sort_values("sim_mean")
    fig, ax = plt.subplots(figsize=(9, 10))
    y = range(len(d))
    err = [d["sim_mean"] - d["sim_p10"], d["sim_p90"] - d["sim_mean"]]
    ax.barh(y, d["sim_mean"], color="#5B8DEF", alpha=0.75,
            xerr=err, error_kw=dict(ecolor="#888", lw=1))
    ax.scatter(d["W"], y, color="#E0763A", zorder=5, s=45, label="Victorias reales")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["TEAM_NAME"], fontsize=8)
    ax.set_xlabel("Victorias proyectadas (media Monte Carlo, banda 10-90%)")
    ax.set_title(f"Simulacion de temporada — {season}\n"
                 f"{N_SIMS} simulaciones sobre el calendario   (naranja = real)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def calendar_fig(games, season, team_name, path):
    g = games[(games.SEASON == season) &
              ((games.home_team == team_name) | (games.away_team == team_name))].copy()
    g = g.sort_values("GAME_DATE")
    probs, results, labels = [], [], []
    for _, r in g.iterrows():
        if r["home_team"] == team_name:
            p = r["p_home_win"]; won = r["home_won"]; opp = r["away_team"]; loc = "vs"
        else:
            p = 1 - r["p_home_win"]; won = 1 - r["home_won"]; opp = r["home_team"]; loc = "@"
        probs.append(p); results.append(won); labels.append(f"{loc} {opp.split()[-1]}")
    n = len(probs)
    fig, ax = plt.subplots(figsize=(min(0.28 * n + 2, 22), 3.2))
    colors = ["#2FB380" if p >= 0.5 else "#D65A7A" for p in probs]
    ax.bar(range(n), probs, color=colors)
    ax.axhline(0.5, color="gray", ls="--", lw=0.7)
    for i, won in enumerate(results):
        ax.text(i, 1.02, "W" if won else "L", ha="center", fontsize=6,
                color="#1a7a4c" if won else "#a83246")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Prob. de victoria")
    ax.set_title(f"Calendario predicho — {team_name} {season}  "
                 f"(verde=victoria probable; W/L arriba = real)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] -> {os.path.basename(path)}")


def main():
    gl = load_gamelogs()
    m1 = pd.read_csv(config.find_table("m1_predictions.csv"))
    q = m1[["SEASON", "TEAM_ID", "wins_pred"]].copy()
    q["q"] = np.clip(q["wins_pred"] / 82.0, 0.05, 0.95)

    games = build_games(gl, q[["SEASON", "TEAM_ID", "q"]])
    if games.empty:
        print("No hay temporadas con game logs Y prediccion de M1 en comun. "
              "Descarga game logs de 2021-22 en adelante.")
        return
    games[["SEASON", "GAME_DATE", "home_team", "away_team", "p_home_win", "home_won"]] \
        .to_csv(os.path.join(config.TABLES_DIR, "m1_game_predictions.csv"), index=False)

    bt = backtest(games, q, m1)
    bt.to_csv(os.path.join(config.TABLES_DIR, "m1_schedule_backtest.csv"), index=False)

    print("=" * 66)
    print("M1 SOBRE EL CALENDARIO (backtest sin sesgo)")
    print("=" * 66)
    print(bt.to_string(index=False))
    print("-" * 66)
    print(f"GLOBAL  MAE_calendario={bt['MAE_calendario'].mean():.2f}   "
          f"MAE_M1_directo={bt['MAE_M1_directo'].mean():.2f}   "
          f"acierto_partido={ (games['p_home_win']>0.5).eq(games['home_won']).mean():.3f}")

    # figura: esperadas vs reales (todas las temporadas)
    allteams = []
    for season, g in games.groupby("SEASON"):
        exp, act = {}, {}
        for _, r in g.iterrows():
            exp[r["TEAM_ID"]] = exp.get(r["TEAM_ID"], 0) + r["p_home_win"]
            exp[r["TEAM_ID_o"]] = exp.get(r["TEAM_ID_o"], 0) + (1 - r["p_home_win"])
            act[r["TEAM_ID"]] = act.get(r["TEAM_ID"], 0) + r["home_won"]
            act[r["TEAM_ID_o"]] = act.get(r["TEAM_ID_o"], 0) + (1 - r["home_won"])
        for tid in exp:
            allteams.append((exp[tid], act[tid]))
    ex, ac = zip(*allteams)
    fig, axx = plt.subplots(figsize=(7, 7))
    axx.scatter(ac, ex, s=35, alpha=0.5, color="#5B8DEF")
    axx.plot([10, 70], [10, 70], "--", color="gray")
    axx.set_xlabel("Victorias reales"); axx.set_ylabel("Victorias esperadas (calendario)")
    axx.set_title("M1 calendario: esperadas vs reales")
    axx.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "m1_schedule_expected_vs_actual.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Monte Carlo: simular la temporada completa muchas veces ---
    mc = monte_carlo(games)
    mc.to_csv(os.path.join(config.TABLES_DIR, "m1_montecarlo_wins.csv"), index=False)
    tmw = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                   "team_clean.csv"))[["SEASON", "TEAM_ID", "W"]]
    mcv = mc.merge(tmw, on=["SEASON", "TEAM_ID"])
    cover = ((mcv["W"] >= mcv["sim_p10"]) & (mcv["W"] <= mcv["sim_p90"])).mean()
    band = (mcv["sim_p90"] - mcv["sim_p10"]).mean()
    print("-" * 66)
    print(f"MONTE CARLO ({N_SIMS} sims/temporada sobre el calendario):")
    print(f"  banda media 10-90% = {band:.1f} victorias   "
          f"(el resultado real cae dentro el {cover*100:.0f}% de las veces; ideal ~80%)")

    last = sorted(games["SEASON"].unique())[-1]
    montecarlo_fig(mc, games, last,
                   os.path.join(config.FIGURES_DIR, f"m1_montecarlo_{last}.png"))

    # calendario de un equipo de ejemplo (el de mas victorias esperadas en la ultima temporada)
    gl_last = games[games.SEASON == last]
    teams = pd.concat([gl_last["home_team"], gl_last["away_team"]]).unique()
    # elige un equipo con nombre valido
    team = sorted(teams)[0]
    safe = team.split()[-1]
    calendar_fig(games, last, team,
                 os.path.join(config.FIGURES_DIR, f"m1_calendar_{safe}_{last}.png"))
    print(f"[m1-sched] -> outputs/tables/m1_schedule_backtest.csv, m1_game_predictions.csv")


if __name__ == "__main__":
    main()
