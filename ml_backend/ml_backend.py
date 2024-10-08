import json

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Mapping of model names to their scikit-learn classes and their hyperparameters for grid search
MODEL_MAPPING = {
    "Linear Regression": (LinearRegression, {}),
    "Ridge Regression": (Ridge, {"alpha": [0.1, 1.0, 10.0]}),
    "Lasso Regression": (Lasso, {"alpha": [0.1, 1.0, 10.0]}),
    "ElasticNet Regression": (
        ElasticNet,
        {
            "alpha": [0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
    ),
    "Decision Trees": (DecisionTreeRegressor, {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 10, 20]}),
    "Random Forest": (
        RandomForestRegressor,
        {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 10, 20]},
    ),
    "Support Vector Machines": (
        SVR,
        {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
        },
    ),
    "Neural Networks": (
        MLPRegressor,
        {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": [0.0001, 0.001, 0.01],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor,
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    ),
    "AdaBoost": (
        AdaBoostRegressor,
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5],
        },
    ),
}


def run_ml_methods(df: pd.DataFrame, target_column: str, feature_columns: list, selected_models: list):
    X = df[feature_columns]
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    for model_name in selected_models:
        model_class, param_grid = MODEL_MAPPING.get(model_name, (None, None))
        if model_class is None:
            continue

        model = model_class()

        # Hyperparameter search with 5-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predict and evaluate
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, train_predictions),
            "test_mse": mean_squared_error(y_test, test_predictions),
            "train_r2": r2_score(y_train, train_predictions),
            "test_r2": r2_score(y_test, test_predictions),
        }

        results[model_name] = {
            "best_hyperparameters": grid_search.best_params_,
            "cv_mean_score": -grid_search.best_score_,
            "cv_std_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            "test_predictions": test_predictions.tolist(),
            "y_test": y_test.tolist(),
            **metrics,
        }

    return results


def download_results_as_json(results, filename="ml_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
