
"""
NBA Predictor - Random Forest Regressor
Modelo alternativo a M1 Ridge.

Predice victorias por equipo-temporada.
Usa las mismas features y datos preparados de M1.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.m1_wins import assemble, FEATURES, year


def create_model():
    """Construye el modelo Random Forest."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def evaluate(y_true, y_pred):
    """Calcula métricas de error."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
    }


def run_random_forest():
    """Evalúa Random Forest mediante walk-forward."""

    df = assemble().copy()

    # Verificar que tenemos los datos necesarios
    required = ["SEASON", "wins"] + list(FEATURES)
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}"
        )

    # Misma lógica temporal que M1
    df["yr"] = df["SEASON"].apply(year)

    seasons = sorted(
        df["SEASON"].unique(),
        key=year,
    )

    results = []
    predictions = []

    for season in seasons:

        train = df[df["yr"] < year(season)].copy()
        test = df[df["SEASON"] == season].copy()

        if len(train) < 20 or test.empty:
            continue

        # No entrenar con valores faltantes
        train = train.dropna(
            subset=list(FEATURES) + ["wins"]
        )
        test = test.dropna(
            subset=list(FEATURES) + ["wins"]
        )

        if len(train) < 20 or test.empty:
            continue

        model = create_model()

        model.fit(
            train[FEATURES],
            train["wins"],
        )

        pred = model.predict(test[FEATURES])
        pred = np.clip(pred, 0, 82)

        metrics = evaluate(test["wins"], pred)

        results.append({
            "season": season,
            "n": len(test),
            **metrics,
        })

        for actual, predicted in zip(
            test["wins"].to_numpy(),
            pred,
        ):
            predictions.append({
                "season": season,
                "wins_real": float(actual),
                "wins_pred": float(predicted),
            })

        print(
            f"{season} | "
            f"MAE: {metrics['MAE']:.2f} | "
            f"RMSE: {metrics['RMSE']:.2f} | "
            f"R2: {metrics['R2']:.3f}"
        )

    if not predictions:
        raise ValueError(
            "No se generaron predicciones. "
            "Revisa los datos y temporadas disponibles."
        )

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)

    global_metrics = evaluate(
        predictions_df["wins_real"],
        predictions_df["wins_pred"],
    )

    print("\nRESULTADOS GLOBALES - RANDOM FOREST")
    for metric, value in global_metrics.items():
        print(f"{metric}: {value:.4f}")

    return results_df, predictions_df, global_metrics


if __name__ == "__main__":
    run_random_forest()

