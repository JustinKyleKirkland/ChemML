import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# Mapping of model names to their scikit-learn classes
MODEL_MAPPING = {
    "Linear Regression": LinearRegression,
    "Decision Trees": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Neural Networks": MLPRegressor,
}


def run_ml_methods(df: pd.DataFrame, target_column: str, feature_columns: list, selected_models: list):
    X = df[feature_columns]
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    for model_name in selected_models:
        model_class = MODEL_MAPPING.get(model_name)
        if model_class is None:
            continue

        model = model_class()

        # Cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="neg_mean_squared_error",
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, train_predictions),
            "test_mse": mean_squared_error(y_test, test_predictions),
            "train_r2": r2_score(y_train, train_predictions),
            "test_r2": r2_score(y_test, test_predictions),
        }

        results[model_name] = {
            "cv_mean_score": -cv_scores.mean(),  # Convert negative MSE to positive for display
            "cv_std_score": cv_scores.std(),
            **metrics,
        }

    return results
