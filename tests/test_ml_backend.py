import numpy as np
import pandas as pd
import pytest

from ml_backend.ml_backend import run_ml_methods
from ml_backend.model_configs import MODEL_CONFIGS


@pytest.fixture
def sample_data():
	"""Create sample data for testing"""
	np.random.seed(42)
	X = np.random.rand(100, 2)
	y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
	df = pd.DataFrame(X, columns=["feature1", "feature2"])
	df["target"] = y
	return df


def test_run_ml_methods_with_custom_params(sample_data):
	"""Test ML methods with custom parameters"""
	custom_params = {"Random Forest Regression": {"max_depth": [3, 5], "min_samples_split": [2, 3]}}

	results = run_ml_methods(
		df=sample_data,
		target_column="target",
		feature_columns=["feature1", "feature2"],
		selected_models=["Random Forest Regression"],
		custom_params=custom_params,
		cv_folds=2,  # Reduce folds for testing
	)

	# Check if results contain the model
	assert "Random Forest Regression" in results

	# Verify the best parameters are from our custom grid
	best_params = results["Random Forest Regression"].best_hyperparameters
	assert best_params["max_depth"] in [3, 5]
	assert best_params["min_samples_split"] in [2, 3]


def test_run_ml_methods_default_params(sample_data):
	"""Test ML methods with default parameters"""
	try:
		results = run_ml_methods(
			df=sample_data,
			target_column="target",
			feature_columns=["feature1", "feature2"],
			selected_models=["Random Forest Regression"],
		)

		# Check if results contain the model
		assert "Random Forest Regression" in results

		# Verify the best parameters are from the default grid
		default_params = MODEL_CONFIGS["Random Forest Regression"].param_grid
		best_params = results["Random Forest Regression"].best_hyperparameters

		# Check that best parameters are within default parameter ranges
		assert best_params["max_depth"] in default_params["max_depth"]
		assert best_params["min_samples_split"] in default_params["min_samples_split"]
	except Exception as e:
		pytest.fail(f"Test failed with exception: {str(e)}")


def test_run_ml_methods_invalid_custom_params(sample_data):
	"""Test ML methods with invalid custom parameters"""
	custom_params = {
		"Random Forest Regression": {
			"invalid_param": [1, 2]  # Invalid parameter name
		}
	}

	with pytest.raises(ValueError, match="Invalid parameters for Random Forest Regression"):
		run_ml_methods(
			df=sample_data,
			target_column="target",
			feature_columns=["feature1", "feature2"],
			selected_models=["Random Forest Regression"],
			custom_params=custom_params,
		)
