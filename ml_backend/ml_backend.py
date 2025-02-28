import json
import multiprocessing
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
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

	best_hyperparameters: Dict[str, Any]
	cv_mean_score: float
	cv_std_score: float
	test_predictions: np.ndarray
	y_test: np.ndarray
	train_mse: float
	test_mse: float
	train_r2: float
	test_r2: float


def run_ml_methods(
	df: pd.DataFrame,
	target_column: str,
	feature_columns: List[str],
	selected_models: List[str],
	custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
	test_size: float = 0.2,
	random_state: int = 42,
	cv_folds: int = 5,
	n_jobs: int = -1,
	pre_dispatch: str = "2*n_jobs",
	error_score: Union[str, float] = "raise",
	refit: bool = True,
) -> Dict[str, ModelResults]:
	"""
	Train and evaluate multiple machine learning models with optimized performance.

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
		pre_dispatch: Number of predispatched jobs for parallel execution
		error_score: Value to assign to the score if an error occurs in fitting
		refit: Whether to refit the best model on the entire dataset

	Returns:
		Dictionary mapping model names to their results

	Raises:
		ValueError: If invalid parameters are provided for a model
	"""
	custom_params = custom_params or {}
	X = df[feature_columns].values  # Convert to numpy array for better performance
	y = df[target_column].values

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
				pre_dispatch=pre_dispatch,
				error_score=error_score,
				refit=refit,
			)
			grid_search.fit(X_train, y_train)
		except ValueError as e:
			if "Invalid parameter" in str(e):
				raise ValueError(f"Invalid parameters for {model_name}: {str(e)}")
			raise

		best_model = grid_search.best_estimator_
		train_predictions = best_model.predict(X_train)
		test_predictions = best_model.predict(X_test)

		results[model_name] = ModelResults(
			best_hyperparameters=grid_search.best_params_,
			cv_mean_score=-grid_search.best_score_,  # Convert back from negative MSE
			cv_std_score=grid_search.cv_results_["std_test_score"][grid_search.best_index_],
			test_predictions=test_predictions,
			y_test=y_test,
			train_mse=mean_squared_error(y_train, train_predictions),
			test_mse=mean_squared_error(y_test, test_predictions),
			train_r2=r2_score(y_train, train_predictions),
			test_r2=r2_score(y_test, test_predictions),
		)

	return results


@lru_cache(maxsize=32)
def _serialize_results(results_dict: Dict[str, Any]) -> str:
	"""Cache and serialize results to JSON string."""
	return json.dumps(results_dict, indent=4)


def download_results_as_json(
	results: Dict[str, ModelResults],
	filename: str = "ml_results.json",
	output_dir: Optional[str] = None,
	ensure_numpy: bool = True,
) -> None:
	"""
	Save model results to a JSON file with optimized serialization.

	Args:
		results: Dictionary of model results
		filename: Name of the output file
		output_dir: Optional directory path for the output file
		ensure_numpy: Convert numpy arrays to lists for JSON serialization
	"""
	output_path = Path(output_dir or ".") / filename

	# Convert dataclass instances to dictionaries with numpy array handling
	serializable_results = {}
	for model_name, model_results in results.items():
		result_dict = asdict(model_results)
		if ensure_numpy:
			result_dict["test_predictions"] = result_dict["test_predictions"].tolist()
			result_dict["y_test"] = result_dict["y_test"].tolist()
		serializable_results[model_name] = result_dict

	# Create directory if it doesn't exist
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Use cached serialization for better performance
	json_str = _serialize_results(serializable_results)

	with open(output_path, "w") as f:
		f.write(json_str)
