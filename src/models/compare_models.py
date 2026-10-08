
"""
NBA Predictor - Comparacion Ridge vs Random Forest.

Ambos modelos utilizan:
- Las mismas 5 variables predictoras.
- Los mismos datos.
- Las mismas temporadas de entrenamiento y prueba.
- Validacion temporal walk-forward.

Ejecutar desde la raiz del repositorio:
    python -m src.models.compare_models

Genera:
    outputs/tables/model_comparison.csv
    outputs/tables/model_comparison_by_season.csv
    outputs/tables/model_predictions_comparison.csv
    outputs/tables/random_forest_feature_importance.csv
"""

import os

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config

from src.models.m1_wins import assemble, FEATURES, year
from src.models.m1_random_forest import (
    create_model,
    optimize_hyperparameters,
)


def calculate_metrics(y_true, y_pred):
    """Calcula metricas de regresion."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correlation = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
        else np.nan
    )

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
        "Correlation": correlation,
    }


def compare_models(df):
    """Backtesting walk-forward para ambos modelos."""

    seasons = sorted(df["SEASON"].unique(), key=year)

    metrics_rows = []
    predictions_rows = []

    for season in seasons:

        train = df[df["yr"] < year(season)]
        test = df[df["SEASON"] == season]

        # Misma condicion del Ridge original
        if len(train) < 20 or len(test) == 0:
            continue

        X_train = train[FEATURES]
        y_train = train["wins"]

        X_test = test[FEATURES]
        y_test = test["wins"]

        # MODELO 1: RIDGE ORIGINAL
        scaler = StandardScaler().fit(X_train)

        ridge = Ridge(alpha=1.0)
        ridge.fit(scaler.transform(X_train), y_train)

        ridge_pred = ridge.predict(
            scaler.transform(X_test)
        )
        ridge_pred = np.clip(ridge_pred, 0, 82)

        # MODELO 2: RANDOM FOREST
        forest = create_model()
        forest.fit(X_train, y_train)

        forest_pred = forest.predict(X_test)
        forest_pred = np.clip(forest_pred, 0, 82)

        # Metricas por temporada
        for model_name, predictions in [
            ("Ridge", ridge_pred),
            ("Random Forest", forest_pred),
        ]:
            metrics_rows.append({
                "season": season,
                "model": model_name,
                "n": len(test),
                **calculate_metrics(y_test, predictions),
            })

        # Predicciones individuales
        for team_id, actual, ridge_wins, forest_wins in zip(
            test["TEAM_ID"],
            y_test,
            ridge_pred,
            forest_pred,
        ):
            predictions_rows.append({
                "SEASON": season,
                "TEAM_ID": team_id,
                "wins_real": float(actual),
                "wins_pred_ridge": float(ridge_wins),
                "wins_pred_random_forest": float(forest_wins),
            })

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(predictions_rows),
    )


def main():

    df = assemble()

    metrics, predictions = compare_models(df)

    if predictions.empty:
        raise ValueError(
            "No hay temporadas suficientes para evaluar "
            "los modelos mediante walk-forward."
        )

    # Metricas globales
    comparison_rows = []

    for name, column in [
        ("Ridge", "wins_pred_ridge"),
        ("Random Forest", "wins_pred_random_forest"),
    ]:
        comparison_rows.append({
            "model": name,
            "n": len(predictions),
            **calculate_metrics(
                predictions["wins_real"],
                predictions[column],
            ),
        })

    comparison = pd.DataFrame(comparison_rows)

    # Importancia de variables del Random Forest
    forest = create_model()
    forest.fit(df[FEATURES], df["wins"])

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Guardar resultados
    os.makedirs(config.TABLES_DIR, exist_ok=True)

    comparison.to_csv(
        os.path.join(
            config.TABLES_DIR,
            "model_comparison.csv",
        ),
        index=False,
    )

    metrics.to_csv(
        os.path.join(
            config.TABLES_DIR,
            "model_comparison_by_season.csv",
        ),
        index=False,
    )

    predictions.to_csv(
        os.path.join(
            config.TABLES_DIR,
            "model_predictions_comparison.csv",
        ),
        index=False,
    )

    importance.to_csv(
        os.path.join(
            config.TABLES_DIR,
            "random_forest_feature_importance.csv",
        ),
        index=False,
    )

    print("\nCOMPARACION GLOBAL")
    print(comparison.round(3).to_string(index=False))

    print("\nRESULTADOS POR TEMPORADA")
    print(metrics.round(3).to_string(index=False))

    print("\nIMPORTANCIA DE VARIABLES")
    print(importance.round(3).to_string(index=False))

    winner = comparison.loc[
        comparison["MAE"].idxmin(), "model"
    ]

    print(f"\nModelo con menor MAE: {winner}")


if __name__ == "__main__":
    main()

