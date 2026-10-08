
"""
Modelo M1 - XGBoost
Predicción de victorias de equipos NBA.

Utiliza las mismas variables del modelo Ridge original.
La evaluación se realiza mediante validación temporal walk-forward.

Ejecutar desde la raíz del repositorio:
    python -m src.models.m1_xgboost
"""

from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.m1_wins import assemble


# ==========================================
# CONFIGURACIÓN
# ==========================================

FEATURES = [
    "squad_strength_proj",
    "continuity",
    "prior_net_rating",
    "best_pie_proj",
    "avg_age_core",
]

PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


# ==========================================
# CREAR MODELO
# ==========================================

def create_model():
    """Crea un modelo XGBoost para regresión."""
    return XGBRegressor(**PARAMS)


# ==========================================
# PREPARAR DATOS
# ==========================================

def load_data():
    """Carga las variables preparadas por el proyecto."""
    df = assemble().copy()

    required = ["season", "wins"] + FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias: {missing}"
        )

    df["wins"] = pd.to_numeric(
        df["wins"], errors="coerce"
    )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(subset=required).copy()
    df = df.sort_values("season").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No hay datos utilizables para entrenar XGBoost."
        )

    return df


# ==========================================
# VALIDACIÓN TEMPORAL
# ==========================================

def walk_forward_validation(df):
    """
    Entrena con temporadas anteriores y predice
    únicamente temporadas posteriores.
    """

    predictions = []
    metrics = []

    seasons = sorted(df["season"].unique())

    for season in seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]

        if len(train) < 30 or test.empty:
            continue

        model = create_model()

        model.fit(
            train[FEATURES],
            train["wins"]
        )

        y_true = test["wins"].to_numpy()

        y_pred = np.clip(
            model.predict(test[FEATURES]),
            0,
            82
        )

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
        r2 = r2_score(y_true, y_pred)

        metrics.append({
            "season": season,
            "Modelo": "XGBoost",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        for i, (index, row) in enumerate(test.iterrows()):
            predictions.append({
                "season": season,
                "team": row.get("team", index),
                "wins_real": y_true[i],
                "wins_pred": y_pred[i],
                "Modelo": "XGBoost",
            })

        print(
            f"Temporada {season}: "
            f"MAE={mae:.3f} | "
            f"RMSE={rmse:.3f} | "
            f"R2={r2:.3f}"
        )

    if not predictions:
        raise ValueError(
            "No hubo suficientes temporadas para validar."
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(metrics),
    )


# ==========================================
# RESULTADOS GLOBALES
# ==========================================

def evaluate_global(predictions):
    y_true = predictions["wins_real"]
    y_pred = predictions["wins_pred"]

    return pd.DataFrame([{
        "Modelo": "XGBoost",
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
        "Correlacion": y_true.corr(y_pred),
    }])


# ==========================================
# EJECUTAR
# ==========================================

def main():
    print("\n=== MODELO XGBOOST NBA ===\n")

    df = load_data()

    print(f"Registros disponibles: {len(df)}")
    print(f"Temporadas: {sorted(df['season'].unique())}")

    predictions, metrics = walk_forward_validation(df)
    global_metrics = evaluate_global(predictions)

    print("\n=== RESULTADOS GLOBALES ===")
    print(
        global_metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        output_dir / "xgboost_predictions.csv",
        index=False
    )

    metrics.to_csv(
        output_dir / "xgboost_seasons.csv",
        index=False
    )

    global_metrics.to_csv(
        output_dir / "xgboost_global.csv",
        index=False
    )

    print("\nArchivos guardados en outputs/tables/")


if __name__ == "__main__":
    main()
