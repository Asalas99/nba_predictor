
"""
NBA Predictor - Comparacion de 4 modelos.

Modelos:
1. Ridge original
2. Random Forest base
3. Random Forest optimizado
4. XGBoost

Todos utilizan:
- Las mismas 5 variables predictoras.
- Los mismos datos de entrenamiento y prueba.
- Validacion temporal walk-forward.
- Predicciones limitadas al rango de 0 a 82 victorias.

Ejecutar desde la raiz del repositorio:
    python -m src.models.compare_models

Archivos generados:
    outputs/tables/model_comparison.csv
    outputs/tables/model_comparison_by_season.csv
    outputs/tables/model_predictions_comparison.csv
    outputs/tables/random_forest_feature_importance.csv
    outputs/tables/xgboost_feature_importance.csv
    outputs/tables/rf_optimized_parameters.csv
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
    create_model as create_random_forest,
    optimize_hyperparameters,
)

from src.models.m1_xgboost import (
    create_model as create_xgboost,
)


# ==========================================
# METRICAS
# ==========================================

def calculate_metrics(y_true, y_pred):
    """Calcula metricas de regresion."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        raise ValueError("No hay predicciones para evaluar.")

    correlation = np.nan

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        correlation = float(
            np.corrcoef(y_true, y_pred)[0, 1]
        )

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) >= 2
            else np.nan
        ),
        "Correlation": correlation,
    }


# ==========================================
# VALIDACION DE DATOS
# ==========================================

def validate_data(df):
    """Comprueba que existan las columnas necesarias."""

    required = [
        "SEASON",
        "TEAM_ID",
        "yr",
        "wins",
        *FEATURES,
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas en los datos: {missing}"
        )

    if df.empty:
        raise ValueError(
            "El conjunto de datos esta vacio."
        )

    df = df.copy()

    numeric_columns = [
        "yr",
        "wins",
        *FEATURES,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=["SEASON", "TEAM_ID", *numeric_columns]
    ).copy()

    if df.empty:
        raise ValueError(
            "No hay registros completos para comparar."
        )

    # Cada equipo debe aparecer una sola vez por temporada.
    duplicates = df.duplicated(
        subset=["SEASON", "TEAM_ID"]
    )

    if duplicates.any():
        raise ValueError(
            "Hay equipos duplicados dentro de una temporada."
        )

    return df


# ==========================================
# COMPARACION WALK-FORWARD
# ==========================================

def compare_models(df):
    """
    Compara cuatro modelos con las mismas
    temporadas de entrenamiento y prueba.
    """

    seasons = sorted(
        df["SEASON"].unique(),
        key=year,
    )

    metrics_rows = []
    predictions_rows = []
    parameters_rows = []

    for season in seasons:

        current_year = year(season)

        train = df[
            df["yr"] < current_year
        ].copy()

        test = df[
            df["SEASON"] == season
        ].copy()

        # Misma condicion que el comparador anterior.
        if len(train) < 20 or test.empty:
            continue

        X_train = train[FEATURES]
        y_train = train["wins"]

        X_test = test[FEATURES]
        y_test = test["wins"]

        print(f"\nEvaluando temporada: {season}")

        # ----------------------------------
        # MODELO 1: RIDGE
        # ----------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=1.0)

        ridge.fit(
            X_train_scaled,
            y_train,
        )

        ridge_pred = np.clip(
            ridge.predict(X_test_scaled),
            0,
            82,
        )

        # ----------------------------------
        # MODELO 2: RANDOM FOREST BASE
        # ----------------------------------

        forest = create_random_forest()

        forest.fit(
            X_train,
            y_train,
        )

        forest_pred = np.clip(
            forest.predict(X_test),
            0,
            82,
        )

        # ----------------------------------
        # MODELO 3: RANDOM FOREST OPTIMIZADO
        # ----------------------------------

        # La optimizacion recibe solamente
        # datos anteriores a la temporada test.
        best_params = optimize_hyperparameters(train)

        optimized_forest = create_random_forest(
            best_params
        )

        optimized_forest.fit(
            X_train,
            y_train,
        )

        optimized_pred = np.clip(
            optimized_forest.predict(X_test),
            0,
            82,
        )

        parameters_rows.append({
            "season": season,
            **best_params,
        })

        print(
            f"Parametros RF optimizado: {best_params}"
        )

        # ----------------------------------
        # MODELO 4: XGBOOST
        # ----------------------------------

        xgboost = create_xgboost()

        xgboost.fit(
            X_train,
            y_train,
        )

        xgboost_pred = np.clip(
            xgboost.predict(X_test),
            0,
            82,
        )

        # ----------------------------------
        # METRICAS POR TEMPORADA
        # ----------------------------------

        model_predictions = {
            "Ridge": ridge_pred,
            "Random Forest": forest_pred,
            "Random Forest Optimizado": optimized_pred,
            "XGBoost": xgboost_pred,
        }

        for model_name, predictions in model_predictions.items():

            metrics_rows.append({
                "season": season,
                "model": model_name,
                "n": len(test),
                **calculate_metrics(
                    y_test,
                    predictions,
                ),
            })

        # ----------------------------------
        # PREDICCIONES POR EQUIPO
        # ----------------------------------

        for (
            team_id,
            actual,
            ridge_wins,
            forest_wins,
            optimized_wins,
            xgboost_wins,
        ) in zip(
            test["TEAM_ID"],
            y_test,
            ridge_pred,
            forest_pred,
            optimized_pred,
            xgboost_pred,
        ):

            predictions_rows.append({
                "SEASON": season,
                "TEAM_ID": team_id,
                "wins_real": float(actual),
                "wins_pred_ridge": float(ridge_wins),
                "wins_pred_random_forest": float(forest_wins),
                "wins_pred_rf_optimized": float(optimized_wins),
                "wins_pred_xgboost": float(xgboost_wins),
            })

        print(
            f"Temporada {season}: "
            f"{len(test)} equipos evaluados."
        )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(predictions_rows),
        pd.DataFrame(parameters_rows),
    )


# ==========================================
# IMPORTANCIA DE VARIABLES
# ==========================================

def calculate_feature_importance(df, model_type):
    """
    Entrena el modelo con los datos disponibles
    para analizar la importancia de variables.

    Esta importancia es descriptiva y no
    corresponde a una evaluacion fuera de muestra.
    """

    if model_type == "random_forest":
        model = create_random_forest()

    elif model_type == "xgboost":
        model = create_xgboost()

    else:
        raise ValueError(
            f"Modelo no reconocido: {model_type}"
        )

    model.fit(
        df[FEATURES],
        df["wins"],
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


# ==========================================
# EJECUCION PRINCIPAL
# ==========================================

def main():

    print("\n====================================")
    print("NBA PREDICTOR - COMPARACION DE MODELOS")
    print("====================================")

    # Cargar datos del pipeline original.
    df = assemble()

    df = validate_data(df)

    print(f"\nRegistros disponibles: {len(df)}")
    print(f"Variables utilizadas: {FEATURES}")

    # Ejecutar los cuatro modelos.
    (
        metrics,
        predictions,
        optimized_parameters,
    ) = compare_models(df)

    if predictions.empty:
        raise ValueError(
            "No hay suficientes temporadas para "
            "realizar la validacion walk-forward."
        )

    # ----------------------------------
    # RESULTADOS GLOBALES
    # ----------------------------------

    comparison_rows = []

    model_columns = [
        ("Ridge", "wins_pred_ridge"),
        ("Random Forest", "wins_pred_random_forest"),
        (
            "Random Forest Optimizado",
            "wins_pred_rf_optimized",
        ),
        ("XGBoost", "wins_pred_xgboost"),
    ]

    for model_name, prediction_column in model_columns:

        comparison_rows.append({
            "model": model_name,
            "n": len(predictions),
            **calculate_metrics(
                predictions["wins_real"],
                predictions[prediction_column],
            ),
        })

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "MAE",
        ascending=True,
    ).reset_index(drop=True)

    # ----------------------------------
    # IMPORTANCIA DE VARIABLES
    # ----------------------------------

    forest_importance = calculate_feature_importance(
        df,
        "random_forest",
    )

    xgboost_importance = calculate_feature_importance(
        df,
        "xgboost",
    )

    # ----------------------------------
    # GUARDAR ARCHIVOS
    # ----------------------------------

    os.makedirs(
        config.TABLES_DIR,
        exist_ok=True,
    )

    def save_table(dataframe, filename):
        path = os.path.join(
            config.TABLES_DIR,
            filename,
        )

        dataframe.to_csv(
            path,
            index=False,
        )

        print(f"Guardado: {path}")

    save_table(
        comparison,
        "model_comparison.csv",
    )

    save_table(
        metrics,
        "model_comparison_by_season.csv",
    )

    save_table(
        predictions,
        "model_predictions_comparison.csv",
    )

    save_table(
        forest_importance,
        "random_forest_feature_importance.csv",
    )

    save_table(
        xgboost_importance,
        "xgboost_feature_importance.csv",
    )

    save_table(
        optimized_parameters,
        "rf_optimized_parameters.csv",
    )

    # ----------------------------------
    # MOSTRAR RESULTADOS
    # ----------------------------------

    print("\n=== COMPARACION GLOBAL ===")

    print(
        comparison.round(3).to_string(
            index=False
        )
    )

    print("\n=== RESULTADOS POR TEMPORADA ===")

    print(
        metrics.round(3).to_string(
            index=False
        )
    )

    print("\n=== IMPORTANCIA RANDOM FOREST ===")

    print(
        forest_importance.round(3).to_string(
            index=False
        )
    )

    print("\n=== IMPORTANCIA XGBOOST ===")

    print(
        xgboost_importance.round(3).to_string(
            index=False
        )
    )

    # ----------------------------------
    # GANADOR
    # ----------------------------------

    winner = comparison.iloc[0]["model"]
    best_mae = comparison.iloc[0]["MAE"]

    print("\n====================================")
    print(f"MODELO GANADOR: {winner}")
    print(f"MEJOR MAE: {best_mae:.3f}")
    print("====================================")


if __name__ == "__main__":
    main()


