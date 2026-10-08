
"""
NBA Predictor - Random Forest Regressor optimizado.

Predice victorias de equipos NBA usando las mismas
variables del modelo M1 Ridge.

Mejoras:
- Busqueda de hiperparametros
- Validacion temporal interna
- Backtesting walk-forward
- Control de sobreajuste
- Importancia de variables

Ejecutar:
    python -m src.models.m1_random_forest
"""

import numpy as np
import pandas as pd

from itertools import product

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.m1_wins import assemble, FEATURES, year


RANDOM_STATE = 42

# Configuracion original para mantener compatibilidad
BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_leaf": 3,
    "max_features": 1.0,
}

# Espacio de busqueda
PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [2, 3, 5],
    "max_features": [0.6, 1.0],
}


def create_model(params=None):
    """Crea un Random Forest con parametros configurables."""

    configuration = BASE_PARAMS.copy()

    if params is not None:
        configuration.update(params)

    return RandomForestRegressor(
        **configuration,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate(y_true, y_pred):
    """Calcula las metricas de regresion."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correlation = np.nan

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(
            mean_squared_error(y_true, y_pred)
        )),
        "R2": float(r2_score(y_true, y_pred)),
        "Correlation": float(correlation),
    }


def parameter_combinations():
    """Genera las configuraciones a evaluar."""

    keys = list(PARAM_GRID.keys())

    for values in product(
        *(PARAM_GRID[key] for key in keys)
    ):
        yield dict(zip(keys, values))


def optimize_hyperparameters(train):
    """
    Busca parametros usando exclusivamente datos
    anteriores a la temporada de prueba externa.

    Cada temporada interna se valida con modelos
    entrenados en temporadas anteriores.
    """

    seasons = sorted(
        train["SEASON"].unique(),
        key=year,
    )

    # Sin suficientes temporadas para validar,
    # conservamos la configuracion original.
    if len(seasons) < 3:
        return BASE_PARAMS.copy()

    best_params = BASE_PARAMS.copy()
    best_mae = float("inf")

    for params in parameter_combinations():

        errors = []

        for season in seasons:

            inner_train = train[
                train["yr"] < year(season)
            ]

            inner_test = train[
                train["SEASON"] == season
            ]

            if len(inner_train) < 20 or inner_test.empty:
                continue

            model = create_model(params)

            model.fit(
                inner_train[FEATURES],
                inner_train["wins"],
            )

            predictions = np.clip(
                model.predict(inner_test[FEATURES]),
                0,
                82,
            )

            # Guardamos errores individuales para
            # ponderar correctamente cada equipo.
            errors.extend(
                np.abs(
                    inner_test["wins"].to_numpy()
                    - predictions
                ).tolist()
            )

        if not errors:
            continue

        current_mae = float(np.mean(errors))

        if current_mae < best_mae:
            best_mae = current_mae
            best_params = params.copy()

    return best_params


def run_random_forest():
    """Ejecuta backtesting walk-forward optimizado."""

    df = assemble().copy()

    required = [
        "SEASON",
        "TEAM_ID",
        "wins",
    ] + list(FEATURES)

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}"
        )

    df["yr"] = df["SEASON"].apply(year)

    seasons = sorted(
        df["SEASON"].unique(),
        key=year,
    )

    results = []
    predictions = []
    importances = []

    for season in seasons:

        train = df[
            df["yr"] < year(season)
        ].copy()

        test = df[
            df["SEASON"] == season
        ].copy()

        if len(train) < 20 or test.empty:
            continue

        # Optimizacion solo con datos anteriores
        best_params = optimize_hyperparameters(train)

        model = create_model(best_params)

        model.fit(
            train[FEATURES],
            train["wins"],
        )

        pred = np.clip(
            model.predict(test[FEATURES]),
            0,
            82,
        )

        metrics = evaluate(test["wins"], pred)

        results.append({
            "season": season,
            "n": len(test),
            **metrics,
            **best_params,
        })

        for team_id, actual, predicted in zip(
            test["TEAM_ID"],
            test["wins"],
            pred,
        ):
            predictions.append({
                "SEASON": season,
                "TEAM_ID": team_id,
                "wins_real": float(actual),
                "wins_pred": float(predicted),
            })

        for feature, importance in zip(
            FEATURES,
            model.feature_importances_,
        ):
            importances.append({
                "season": season,
                "feature": feature,
                "importance": float(importance),
            })

        print(
            f"{season} | "
            f"MAE: {metrics['MAE']:.2f} | "
            f"RMSE: {metrics['RMSE']:.2f} | "
            f"R2: {metrics['R2']:.3f}"
        )

        print(f"Parametros: {best_params}")

    if not predictions:
        raise ValueError(
            "No hay suficientes datos historicos "
            "para realizar el backtesting."
        )

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)
    importance_df = pd.DataFrame(importances)

    global_metrics = evaluate(
        predictions_df["wins_real"],
        predictions_df["wins_pred"],
    )

    print("\nRESULTADOS GLOBALES - RANDOM FOREST")
    for metric, value in global_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nIMPORTANCIA PROMEDIO DE VARIABLES")

    if not importance_df.empty:
        summary = (
            importance_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        print(summary.to_string())

    return results_df, predictions_df, global_metrics


if __name__ == "__main__":
    run_random_forest()


