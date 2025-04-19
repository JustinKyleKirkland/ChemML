import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json

# Default available ML methods
AVAILABLE_ML_METHODS: List[str] = [
	"Linear Regression",
	"Ridge Regression",
	"Lasso Regression",
	"Elastic Net Regression",
	"Decision Tree Regression",
	"Random Forest Regression",
	"Support Vector Regression",
	"Neural Network Regression",
	"Gradient Boosting Regression",
	"AdaBoost Regression",
	"K-Nearest Neighbors Regression",
	"Gaussian Process Regression",
]

# Default hyperparameter grid for each model
MODEL_CONFIGS = {
	"Linear Regression": {
		"model": LinearRegression,
		"param_grid": {"fit_intercept": [True, False]},
	},
	"Ridge Regression": {
		"model": Ridge,
		"param_grid": {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]},
	},
	"Lasso Regression": {
		"model": Lasso,
		"param_grid": {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]},
	},
	"Elastic Net Regression": {
		"model": ElasticNet,
		"param_grid": {
			"alpha": [0.1, 1.0, 10.0],
			"l1_ratio": [0.1, 0.5, 0.9],
			"fit_intercept": [True, False],
		},
	},
	"Decision Tree Regression": {
		"model": DecisionTreeRegressor,
		"param_grid": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
	},
	"Random Forest Regression": {
		"model": RandomForestRegressor,
		"param_grid": {
			"n_estimators": [10, 50, 100],
			"max_depth": [3, 5, 10, None],
			"min_samples_split": [2, 5, 10],
		},
	},
	"Support Vector Regression": {
		"model": SVR,
		"param_grid": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
	},
	"Neural Network Regression": {
		"model": MLPRegressor,
		"param_grid": {
			"hidden_layer_sizes": [(50,), (100,), (50, 50)],
			"alpha": [0.0001, 0.001, 0.01],
			"learning_rate": ["constant", "adaptive"],
		},
	},
	"Gradient Boosting Regression": {
		"model": GradientBoostingRegressor,
		"param_grid": {
			"n_estimators": [50, 100, 200],
			"learning_rate": [0.01, 0.1, 0.2],
			"max_depth": [3, 5, 7],
		},
	},
	"AdaBoost Regression": {
		"model": AdaBoostRegressor,
		"param_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
	},
	"K-Nearest Neighbors Regression": {
		"model": KNeighborsRegressor,
		"param_grid": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
	},
	"Gaussian Process Regression": {
		"model": GaussianProcessRegressor,
		"param_grid": {"alpha": [1e-10, 1e-5, 1e-1], "normalize_y": [True, False]},
	},
}


@dataclass
class MLResult:
	"""Stores the results of ML model execution."""

	method_name: str
	cv_mean_score: float
	cv_std_score: float
	train_mse: float
	test_mse: float
	train_r2: float
	test_r2: float
	best_hyperparameters: Dict[str, Any]
	y_test: List[float]
	test_predictions: List[float]

	def __post_init__(self) -> None:
		"""Validate the data after initialization."""
		if len(self.y_test) != len(self.test_predictions):
			raise ValueError("Length of y_test and test_predictions must match")


class MLController:
	"""
	Controller for machine learning operations, separating business logic from UI.

	This class handles all ML-related operations, including:
	- Feature selection
	- Model training and evaluation
	- Results storage and retrieval
	- Model configuration
	"""

	def __init__(self) -> None:
		"""Initialize the ML controller."""
		self.logger = logging.getLogger("MLController")
		self.df = pd.DataFrame()
		self.available_methods = AVAILABLE_ML_METHODS.copy()
		self.custom_params = {}
		self.results_cache = {}

	def set_dataframe(self, df: pd.DataFrame) -> None:
		"""
		Set the DataFrame to use for ML operations.

		Args:
		    df: Input DataFrame
		"""
		self.df = df.copy()

	def get_available_models(self) -> List[str]:
		"""
		Get list of available ML models.

		Returns:
		    List of model names
		"""
		return self.available_methods

	def set_custom_parameters(self, method: str, params: Dict[str, Any]) -> None:
		"""
		Set custom hyperparameters for a method.

		Args:
		    method: Name of the ML method
		    params: Dictionary of parameter grid
		"""
		self.custom_params[method] = params

	def get_custom_parameters(self, method: str) -> Dict[str, Any]:
		"""
		Get custom hyperparameters for a method.

		Args:
		    method: Name of the ML method

		Returns:
		    Dictionary of custom parameters if set, otherwise default parameters
		"""
		if method in self.custom_params:
			return self.custom_params[method]
		elif method in MODEL_CONFIGS:
			return MODEL_CONFIGS[method]["param_grid"]
		else:
			return {}

	def validate_selections(
		self, target_column: str, feature_columns: List[str], selected_methods: List[str]
	) -> Tuple[bool, str]:
		"""
		Validate selections for ML training.

		Args:
		    target_column: Target variable name
		    feature_columns: List of feature column names
		    selected_methods: List of selected ML methods

		Returns:
		    Tuple of (is_valid, error_message)
		"""
		if self.df.empty:
			return False, "No data loaded. Please load a CSV file first."

		if not target_column or target_column == "Select Target Column":
			return False, "Please select a target column."

		if not feature_columns:
			return False, "Please select at least one feature column."

		if not selected_methods:
			return False, "Please select at least one ML method."

		if target_column not in self.df.columns:
			return False, f"Target column '{target_column}' not found in data."

		for col in feature_columns:
			if col not in self.df.columns:
				return False, f"Feature column '{col}' not found in data."

		# Check if target column has numeric data
		if not pd.api.types.is_numeric_dtype(self.df[target_column]):
			return False, f"Target column '{target_column}' must contain numeric data."

		# Check if feature columns have valid data types
		for col in feature_columns:
			if not (pd.api.types.is_numeric_dtype(self.df[col]) or pd.api.types.is_bool_dtype(self.df[col])):
				return False, f"Feature column '{col}' must contain numeric data."

		return True, ""

	def run_ml_methods(
		self,
		target_column: str,
		feature_columns: List[str],
		selected_methods: List[str],
		test_size: float = 0.2,
		random_state: int = 42,
		cv_folds: int = 5,
	) -> Dict[str, MLResult]:
		"""
		Run selected ML methods on specified data.

		Args:
		    target_column: Target variable name
		    feature_columns: List of feature column names
		    selected_methods: List of selected ML methods
		    test_size: Proportion of data to use for testing
		    random_state: Random seed for reproducibility
		    cv_folds: Number of cross-validation folds

		Returns:
		    Dictionary of ML results by method name

		Raises:
		    ValueError: If validation fails
		"""
		is_valid, error_message = self.validate_selections(target_column, feature_columns, selected_methods)

		if not is_valid:
			raise ValueError(error_message)

		self.logger.info(f"Running ML methods: {selected_methods}")
		self.logger.info(f"Target: {target_column}, Features: {feature_columns}")

		try:
			# Prepare data
			X = self.df[feature_columns].copy()
			y = self.df[target_column].copy()

			# Handle missing values
			X = X.fillna(X.mean())

			# Split data
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

			# Scale features
			scaler = StandardScaler()
			X_train_scaled = scaler.fit_transform(X_train)
			X_test_scaled = scaler.transform(X_test)

			results = {}

			# Train and evaluate each model
			for method in selected_methods:
				self.logger.info(f"Training {method}...")

				if method not in MODEL_CONFIGS:
					self.logger.warning(f"Unknown method: {method}, skipping")
					continue

				# Get model configuration
				model_class = MODEL_CONFIGS[method]["model"]
				param_grid = self.get_custom_parameters(method)

				# Create grid search
				grid_search = GridSearchCV(
					model_class(), param_grid, cv=cv_folds, scoring="neg_mean_squared_error", n_jobs=-1
				)

				# Fit model
				grid_search.fit(X_train_scaled, y_train)

				# Get best model
				best_model = grid_search.best_estimator_

				# Make predictions
				train_pred = best_model.predict(X_train_scaled)
				test_pred = best_model.predict(X_test_scaled)

				# Calculate metrics
				train_mse = mean_squared_error(y_train, train_pred)
				test_mse = mean_squared_error(y_test, test_pred)
				train_r2 = r2_score(y_train, train_pred)
				test_r2 = r2_score(y_test, test_pred)

				# Create result object
				result = MLResult(
					method_name=method,
					cv_mean_score=-grid_search.best_score_,  # Negate to get MSE
					cv_std_score=grid_search.cv_results_["std_test_score"][grid_search.best_index_],
					train_mse=train_mse,
					test_mse=test_mse,
					train_r2=train_r2,
					test_r2=test_r2,
					best_hyperparameters=grid_search.best_params_,
					y_test=y_test.tolist(),
					test_predictions=test_pred.tolist(),
				)

				# Store result
				results[method] = result

				# Cache result
				cache_key = f"{method}_{target_column}_{','.join(sorted(feature_columns))}"
				self.results_cache[cache_key] = result

				self.logger.info(f"Completed {method} (Test R²: {test_r2:.4f})")

			return results

		except Exception as e:
			self.logger.error(f"Error running ML methods: {str(e)}")
			raise

	def get_cached_result(self, method: str, target_column: str, feature_columns: List[str]) -> Optional[MLResult]:
		"""
		Get cached result for a method and feature set.

		Args:
		    method: Name of the ML method
		    target_column: Target column name
		    feature_columns: List of feature column names

		Returns:
		    MLResult or None if not cached
		"""
		cache_key = f"{method}_{target_column}_{','.join(sorted(feature_columns))}"
		return self.results_cache.get(cache_key)

	def save_results_to_json(self, results: Dict[str, MLResult], filename: str) -> None:
		"""
		Save ML results to a JSON file.

		Args:
		    results: Dictionary of results by method name
		    filename: Path to save the JSON file
		"""
		try:
			# Convert results to serializable format
			serializable_results = {}

			for method, result in results.items():
				serializable_results[method] = {
					"method_name": result.method_name,
					"cv_mean_score": result.cv_mean_score,
					"cv_std_score": result.cv_std_score,
					"train_mse": result.train_mse,
					"test_mse": result.test_mse,
					"train_r2": result.train_r2,
					"test_r2": result.test_r2,
					"best_hyperparameters": result.best_hyperparameters,
					# Only store first 100 predictions for file size
					"y_test_sample": result.y_test[:100],
					"test_predictions_sample": result.test_predictions[:100],
				}

			with open(filename, "w") as f:
				json.dump(serializable_results, f, indent=4)

			self.logger.info(f"Results saved to {filename}")

		except Exception as e:
			self.logger.error(f"Error saving results to {filename}: {str(e)}")
			raise
