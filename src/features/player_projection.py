"""
FASE A del predictor: proyeccion de jugador.

Para cada temporada objetivo T, proyecta el PIE y los minutos de cada jugador
usando SOLO su historia hasta T-1 (sin fuga). Cubre el caso de jugadores que aun
no tienen stats de T: se predice su aporte desde su historia + curva de edad, y
si no hay historia (rookies) se usa un prior.

Prueba DOS curvas de edad:
  - 'data'  : estimada de tus propios datos (cambio de PIE ano a ano por edad,
              usando solo temporadas < T para no filtrar futuro)
  - 'std'   : curva estandar parametrica (pico ~26)

  python -m src.features.player_projection

Lee : data/processed/players/combined/player_clean.csv
Crea: data/processed/players/combined/player_projections.csv
      (SEASON, PLAYER_ID, TEAM_ID, AGE, n_prior, pie_proj_data, pie_proj_std,
       min_proj, pie_actual, min_actual, is_rookie)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

RECENCY_W = [0.5, 0.3, 0.2]   # pesos ultimas 3 temporadas (mas reciente primero)
SHRINK_K = 40                  # fuerza de regresion a la media (en "partidos")
CAP_MIN = 38.0

# --- Termino de trayectoria (para jovenes en ascenso, tipo Wembanyama) ---
TRAJ_CAP = 0.035   # tope al cambio ano-a-ano que se extrapola (evita ruido)
TRAJ_AGE = 25      # edad hasta la que aplica el empujon de trayectoria
TRAJ_SPAN = 8.0    # que tan rapido baja el peso con la edad
TRAJ_MAXW = 0.6    # peso maximo que puede tener la trayectoria (los mas jovenes)


def year_of(season: str) -> int:
    return int(str(season)[:4])


def std_age_mult(age: float) -> float:
    """Curva estandar: multiplicador de valor por edad, pico en 26.
    Cuadratica suave, normalizada a 1.0 en el pico."""
    peak = 26.0
    # penalizacion asimetrica: declive un poco mas fuerte despues del pico
    if age <= peak:
        return 1.0 - 0.006 * (peak - age) ** 2 / 4
    return 1.0 - 0.010 * (age - peak) ** 2 / 4


def build_data_curve(hist: pd.DataFrame) -> dict:
    """delta[age] = cambio medio de PIE de esa edad a la siguiente (year-over-year)."""
    h = hist.sort_values(["PLAYER_ID", "yr"]).copy()
    h["pie_next"] = h.groupby("PLAYER_ID")["PIE"].shift(-1)
    h["yr_next"] = h.groupby("PLAYER_ID")["yr"].shift(-1)
    consec = h[h["yr_next"] == h["yr"] + 1].copy()
    consec["dpie"] = consec["pie_next"] - consec["PIE"]
    g = consec.groupby("AGE")["dpie"].mean()
    # suaviza: rellena edades faltantes con 0 e interpola
    ages = range(18, 42)
    return {a: float(g.get(a, np.nan)) for a in ages}


def data_age_delta(curve: dict, from_age: int, to_age: int) -> float:
    """Suma de deltas de from_age hasta to_age (puede ser negativa)."""
    if to_age == from_age:
        return 0.0
    step = 1 if to_age > from_age else -1
    total, a = 0.0, int(from_age)
    while a != int(to_age):
        d = curve.get(a, 0.0)
        if np.isnan(d):
            d = 0.0
        total += d * step
        a += step
    return total


def recency_baseline(past: pd.DataFrame, prior: float) -> tuple:
    """Media ponderada por recencia + regresion a la media segun tamano de muestra."""
    past = past.sort_values("yr", ascending=False).head(3)
    w = np.array(RECENCY_W[:len(past)])
    w = w / w.sum()
    base_pie = float((past["PIE"].values * w).sum())
    base_min = float((past["MIN"].values * w).sum())
    n = float(past["GP"].sum()) if "GP" in past else 40.0 * len(past)
    shrunk = (n * base_pie + SHRINK_K * prior) / (n + SHRINK_K)
    return shrunk, base_min


def project_for_season(target: str, hist_all: pd.DataFrame, actual: pd.DataFrame,
                       prior: float) -> pd.DataFrame:
    ty = year_of(target)
    hist = hist_all[hist_all["yr"] < ty]
    curve = build_data_curve(hist)

    rows = []
    # jugadores presentes en T (roster real, para backtest) + su fila real
    for _, r in actual.iterrows():
        pid = r["PLAYER_ID"]
        past = hist[hist["PLAYER_ID"] == pid]
        age_T = r["AGE"] if not np.isnan(r.get("AGE", np.nan)) else np.nan
        if len(past) == 0:
            # rookie / sin historia -> prior por edad
            rows.append(dict(
                SEASON=target, PLAYER_ID=pid, TEAM_ID=r["TEAM_ID"], AGE=age_T,
                n_prior=0, is_rookie=1,
                pie_proj_data=prior, pie_proj_std=prior, pie_proj_traj=prior,
                min_proj=float(min(r.get("MIN", 15.0), CAP_MIN)),  # placeholder
                pie_actual=r["PIE"], min_actual=r.get("MIN", np.nan)))
            continue
        base_pie, base_min = recency_baseline(past, prior)
        past_sorted = past.sort_values("yr")
        last_row = past_sorted.iloc[-1]
        last_age = int(last_row["AGE"])
        pie_last = float(last_row["PIE"])  # baseline naive (ultimo anio, sin edad)
        tgt_age = int(age_T) if not np.isnan(age_T) else last_age + (ty - past["yr"].max())
        # curva data (aditiva) y std (multiplicativa)
        pie_data = base_pie + data_age_delta(curve, last_age, tgt_age)
        pie_std = base_pie * std_age_mult(tgt_age) / max(std_age_mult(last_age), 1e-6)
        # trayectoria: si un jugador JOVEN viene subiendo, proyecta que sigue subiendo.
        pies = past_sorted["PIE"].values
        yoy = float(np.clip(pies[-1] - pies[-2], -TRAJ_CAP, TRAJ_CAP)) if len(pies) >= 2 else 0.0
        traj_extrap = pie_last + yoy
        alpha = float(np.clip((TRAJ_AGE - tgt_age) / TRAJ_SPAN, 0.0, TRAJ_MAXW))
        pie_traj = alpha * traj_extrap + (1 - alpha) * pie_std
        min_proj = float(np.clip(base_min * (0.97 if tgt_age > 30 else 1.0), 0, CAP_MIN))
        rows.append(dict(
            SEASON=target, PLAYER_ID=pid, TEAM_ID=r["TEAM_ID"], AGE=age_T,
            n_prior=len(past), is_rookie=0, pie_last=round(pie_last, 4),
            pie_proj_data=round(pie_data, 4), pie_proj_std=round(pie_std, 4),
            pie_proj_traj=round(pie_traj, 4), min_proj=round(min_proj, 1),
            pie_actual=r["PIE"], min_actual=r.get("MIN", np.nan)))
    return pd.DataFrame(rows)


def validate(proj: pd.DataFrame) -> None:
    v = proj[proj["is_rookie"] == 0].dropna(subset=["pie_actual"])

    def mae(col):
        return (v["pie_actual"] - v[col]).abs().mean()

    def corr(col):
        return v[col].corr(v["pie_actual"])

    print("=" * 60)
    print(f"VALIDACION proyeccion de PIE ({len(v)} jugador-temporadas con historia)")
    print("=" * 60)
    print(f"  {'pie_last (naive)':18s}  MAE={mae('pie_last'):.4f}   corr={corr('pie_last'):.3f}")
    for col in ["pie_proj_data", "pie_proj_std", "pie_proj_traj"]:
        if col in v.columns:
            print(f"  {col:18s}  MAE={mae(col):.4f}   corr={corr(col):.3f}")
    # subgrupo jovenes (donde deberia notarse la trayectoria)
    yng = v[v["AGE"] <= 23]
    if len(yng) and "pie_proj_traj" in v.columns:
        def m(col, d):
            return (d["pie_actual"] - d[col]).abs().mean()
        print(f"  -- jovenes <=23 (n={len(yng)}): "
              f"std MAE={m('pie_proj_std', yng):.4f}  traj MAE={m('pie_proj_traj', yng):.4f}")


def main() -> None:
    path = os.path.join(config.PROCESSED_DIR, "players", "combined", "player_clean.csv")
    pl = pd.read_csv(path)
    pl["yr"] = pl["SEASON"].map(year_of)
    prior = float(pl["PIE"].mean())

    seasons = sorted(pl["SEASON"].unique())
    out = []
    for target in seasons[1:]:  # el primero no tiene historia previa
        actual = pl[pl["SEASON"] == target]
        out.append(project_for_season(target, pl, actual, prior))
    proj = pd.concat(out, ignore_index=True)

    dst = os.path.join(config.PROCESSED_DIR, "players", "combined", "player_projections.csv")
    proj.to_csv(dst, index=False)
    print(f"[proj] {len(proj)} proyecciones ({proj['is_rookie'].sum()} rookies) -> {dst}")

    # baseline naive: PIE del ultimo anio (sin ajuste de edad)
    validate(proj)


if __name__ == "__main__":
    main()
