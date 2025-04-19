import pytest
import pandas as pd
import numpy as np
from chemml.ui.controllers.ml_controller import MLController, MLResult


@pytest.fixture
def sample_dataframe():
	"""Create a sample DataFrame for testing."""
	# Create a predictable dataset with linear relationship
	np.random.seed(42)  # For reproducibility
	n_samples = 50
	X = np.random.rand(n_samples, 3)
	y = 2 * X[:, 0] + 0.5 * X[:, 1] - 1.0 * X[:, 2] + 0.1 * np.random.randn(n_samples)

	df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
	df["target"] = y
	return df


@pytest.fixture
def ml_controller():
	"""Create an MLController instance for testing."""
	return MLController()


def test_ml_controller_initialization(ml_controller):
	"""Test that the ML controller initializes correctly."""
	assert ml_controller is not None
	assert ml_controller.df.empty
	assert len(ml_controller.available_methods) > 0
	assert len(ml_controller.custom_params) == 0
	assert len(ml_controller.results_cache) == 0


def test_set_dataframe(ml_controller, sample_dataframe):
	"""Test setting DataFrame in the controller."""
	ml_controller.set_dataframe(sample_dataframe)
	assert ml_controller.df.equals(sample_dataframe)
	assert list(ml_controller.df.columns) == ["feature1", "feature2", "feature3", "target"]


def test_get_available_models(ml_controller):
	"""Test getting available ML models."""
	models = ml_controller.get_available_models()
	assert isinstance(models, list)
	assert "Linear Regression" in models
	assert "Random Forest Regression" in models


def test_set_get_custom_parameters(ml_controller):
	"""Test setting and getting custom parameters."""
	method = "Linear Regression"
	params = {"fit_intercept": [False]}

	ml_controller.set_custom_parameters(method, params)
	retrieved_params = ml_controller.get_custom_parameters(method)

	assert retrieved_params == params


def test_validate_selections(ml_controller, sample_dataframe):
	"""Test validation of ML selections."""
	ml_controller.set_dataframe(sample_dataframe)

	# Valid selections
	is_valid, _ = ml_controller.validate_selections(
		target_column="target", feature_columns=["feature1", "feature2"], selected_methods=["Linear Regression"]
	)
	assert is_valid

	# Invalid - missing target
	is_valid, error_msg = ml_controller.validate_selections(
		target_column="", feature_columns=["feature1", "feature2"], selected_methods=["Linear Regression"]
	)
	assert not is_valid
	assert "target column" in error_msg.lower()

	# Invalid - missing features
	is_valid, error_msg = ml_controller.validate_selections(
		target_column="target", feature_columns=[], selected_methods=["Linear Regression"]
	)
	assert not is_valid
	assert "feature" in error_msg.lower()

	# Invalid - missing methods
	is_valid, error_msg = ml_controller.validate_selections(
		target_column="target", feature_columns=["feature1", "feature2"], selected_methods=[]
	)
	assert not is_valid
	assert "method" in error_msg.lower()


def test_run_ml_methods(ml_controller, sample_dataframe):
	"""Test running ML methods."""
	ml_controller.set_dataframe(sample_dataframe)

	# Run Linear Regression - which should work well on our linear data
	results = ml_controller.run_ml_methods(
		target_column="target",
		feature_columns=["feature1", "feature2", "feature3"],
		selected_methods=["Linear Regression"],
	)

	# Check that results contains expected method
	assert "Linear Regression" in results

	# Check that result has expected structure
	result = results["Linear Regression"]
	assert isinstance(result, MLResult)
	assert result.method_name == "Linear Regression"
	assert isinstance(result.cv_mean_score, float)
	assert isinstance(result.train_r2, float)
	assert isinstance(result.test_r2, float)

	# Since our data is linear, Linear Regression should perform well
	assert result.train_r2 > 0.9

	# Test that caching works
	cached_result = ml_controller.get_cached_result("Linear Regression", "target", ["feature1", "feature2", "feature3"])
	assert cached_result is not None
	assert cached_result.method_name == "Linear Regression"


@pytest.mark.parametrize("method", ["Linear Regression", "Ridge Regression"])
def test_multiple_ml_methods(ml_controller, sample_dataframe, method):
	"""Test that multiple ML methods can be run successfully."""
	ml_controller.set_dataframe(sample_dataframe)

	results = ml_controller.run_ml_methods(
		target_column="target", feature_columns=["feature1", "feature2", "feature3"], selected_methods=[method]
	)

	assert method in results
	assert results[method].method_name == method
	assert results[method].train_r2 > 0.9  # Should perform well on linear data
