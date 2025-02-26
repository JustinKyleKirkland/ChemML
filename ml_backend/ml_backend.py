import json
import multiprocessing
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from .model_configs import MODEL_CONFIGS

if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # method already set


@dataclass
class ModelResults:
    """Results from training and evaluating a model."""

    best_hyperparameters: Dict
    cv_mean_score: float
    cv_std_score: float
    test_predictions: List[float]
    y_test: List[float]
    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float


def run_ml_methods(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    selected_models: List[str],
    custom_params: Optional[Dict[str, Dict]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> Dict[str, ModelResults]:
    """
    Train and evaluate multiple machine learning models.

    Args:
            df: Input DataFrame containing features and target
            target_column: Name of the target variable column
            feature_columns: List of feature column names
            selected_models: List of model names to train
            custom_params: Optional dictionary of custom parameters for each model
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            n_jobs: Number of jobs to run in parallel

    Returns:
            Dictionary mapping model names to their results

    Raises:
            ValueError: If invalid parameters are provided for a model
    """
    custom_params = custom_params or {}
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}

    for model_name in selected_models:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not found in configurations")

        config = MODEL_CONFIGS[model_name]
        model = config.model_class()

        # Use custom parameters if provided, otherwise use defaults
        param_grid = custom_params.get(model_name, config.param_grid)

        try:
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=1 if "pytest" in sys.modules else n_jobs,
            )
            grid_search.fit(X_train, y_train)
        except ValueError as e:
            # Re-raise parameter validation errors
            if "Invalid parameter" in str(e):
                raise ValueError(f"Invalid parameters for {model_name}: {str(e)}")
            raise

        best_model = grid_search.best_estimator_
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        results[model_name] = ModelResults(
            best_hyperparameters=grid_search.best_params_,
            cv_mean_score=-grid_search.best_score_,
            cv_std_score=grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            test_predictions=test_predictions.tolist(),
            y_test=y_test.tolist(),
            train_mse=mean_squared_error(y_train, train_predictions),
            test_mse=mean_squared_error(y_test, test_predictions),
            train_r2=r2_score(y_train, train_predictions),
            test_r2=r2_score(y_test, test_predictions),
        )

    return results


def download_results_as_json(
    results: Dict[str, ModelResults], filename: str = "ml_results.json", output_dir: Optional[str] = None
) -> None:
    """
    Save model results to a JSON file.

    Args:
            results: Dictionary of model results
            filename: Name of the output file
            output_dir: Optional directory path for the output file
    """
    output_path = Path(output_dir or ".") / filename

    # Convert dataclass instances to dictionaries
    serializable_results = {model_name: asdict(model_results) for model_name, model_results in results.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
