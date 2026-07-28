"""
Features de ENTRENADOR (leakage-safe).

Para cada equipo-temporada T calcula, usando SOLO la historia del coach en
temporadas < T:
  - huella de estilo: pace / lean ofensivo / lean defensivo tipicos del coach
  - residual de rendimiento: victorias por encima de lo que predice el talento
    (wins - wins_esperadas_por_talento), promediado y con shrinkage
  - experiencia y si es coach nuevo (sin historia previa -> prior neutro)

  python -m src.features.coach_features

Lee : data/raw/coaches/coaches.csv
      data/processed/teams/combined/team_clean.csv
      data/processed/players/combined/squad_strength.csv
Crea: data/processed/teams/combined/coach_features.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

SHRINK = 1.5  # temporadas-equivalentes de regresion a 0 para el residual


def year(season: str) -> int:
    return int(season[:4])


def main() -> None:
    coaches = pd.read_csv(os.path.join(config.RAW_DIR, "coaches", "coaches.csv"))
    tm = pd.read_csv(os.path.join(config.PROCESSED_DIR, "teams", "combined",
                                  "team_clean.csv"))
    ss = pd.read_csv(os.path.join(config.PROCESSED_DIR, "players", "combined",
                                  "squad_strength.csv"))[["SEASON", "TEAM_ID", "pie_wmean"]]

    # panel base: cada equipo-temporada con su coach, wins, estilo y talento realizado
    base = (tm[["SEASON", "TEAM_ID", "W", "OFF_RATING", "DEF_RATING", "PACE"]]
            .merge(ss, on=["SEASON", "TEAM_ID"], how="left")
            .merge(coaches[["SEASON", "TEAM_ID", "coach_name"]],
                   on=["SEASON", "TEAM_ID"], how="left"))
    base["yr"] = base["SEASON"].map(year)
    # lean = desviacion respecto a la media de la liga esa temporada
    for c in ["OFF_RATING", "DEF_RATING", "PACE"]:
        base[f"{c}_lean"] = base.groupby("SEASON")[c].transform(lambda s: s - s.mean())

    rows = []
    for _, r in base.iterrows():
        T = r["yr"]
        coach = r["coach_name"]
        # historia del coach ESTRICTAMENTE anterior a T
        hist = base[(base["coach_name"] == coach) & (base["yr"] < T)] \
            if pd.notna(coach) else base.iloc[0:0]
        n = len(hist)
        if n == 0:
            rows.append(dict(SEASON=r["SEASON"], TEAM_ID=r["TEAM_ID"],
                             coach_name=coach, coach_exp=0, is_new_coach=1,
                             coach_resid=0.0, coach_pace_lean=0.0,
                             coach_off_lean=0.0, coach_def_lean=0.0))
            continue
        # residual de rendimiento: fit wins ~ pie_wmean con datos < T
        train = base[(base["yr"] < T)].dropna(subset=["pie_wmean", "W"])
        if len(train) >= 10:
            b1, b0 = np.polyfit(train["pie_wmean"], train["W"], 1)
            hist_pred = b0 + b1 * hist["pie_wmean"]
            resid = (hist["W"] - hist_pred).mean()
        else:
            resid = 0.0
        resid_shrunk = resid * n / (n + SHRINK)
        rows.append(dict(
            SEASON=r["SEASON"], TEAM_ID=r["TEAM_ID"], coach_name=coach,
            coach_exp=n, is_new_coach=0,
            coach_resid=round(float(resid_shrunk), 3),
            coach_pace_lean=round(float(hist["PACE_lean"].mean()), 3),
            coach_off_lean=round(float(hist["OFF_RATING_lean"].mean()), 3),
            coach_def_lean=round(float(hist["DEF_RATING_lean"].mean()), 3),
        ))

    out = pd.DataFrame(rows)
    dst = os.path.join(config.PROCESSED_DIR, "teams", "combined", "coach_features.csv")
    out.to_csv(dst, index=False)
    print(f"[coach] {len(out)} equipos-temporada -> {dst}")
    print(f"[coach] coaches nuevos (sin historia previa): {out['is_new_coach'].sum()}")

    # validacion: ¿el residual del coach (solo historia previa) anticipa wins?
    v = out.merge(tm[["SEASON", "TEAM_ID", "W"]], on=["SEASON", "TEAM_ID"])
    v = v[v["is_new_coach"] == 0]
    print(f"[coach] corr coach_resid vs wins (coaches con historia, n={len(v)}): "
          f"{v['coach_resid'].corr(v['W']):.3f}")
    print("[coach] top-5 coaches por residual medio (wins sobre talento):")
    top = (out.groupby("coach_name")["coach_resid"].mean()
              .sort_values(ascending=False).head(5))
    print(top.round(2).to_string())


if __name__ == "__main__":
    main()
