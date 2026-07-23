"""Tests for the Settings tab and its integration with the ML run config."""

import numpy as np
import pandas as pd
import pytest

from chemml.ui.controllers.ml_controller import MLController
from chemml.ui.views.settings_view import DEFAULT_CONFIG, SettingsView


@pytest.fixture
def settings(qtbot):
	view = SettingsView()
	qtbot.addWidget(view)
	return view


@pytest.fixture
def sample_dataframe():
	np.random.seed(0)
	X = np.random.rand(40, 3)
	y = 2 * X[:, 0] - X[:, 1]
	df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
	df["target"] = y
	return df


def test_defaults(settings):
	assert settings.get_config() == DEFAULT_CONFIG


def test_reset_defaults(settings):
	settings.test_size_spin.setValue(0.4)
	settings.cv_folds_spin.setValue(8)
	settings.reset_defaults()
	assert settings.get_config() == DEFAULT_CONFIG


def test_settings_changed_signal(settings, qtbot):
	with qtbot.waitSignal(settings.settings_changed, timeout=1000) as blocker:
		settings.cv_folds_spin.setValue(7)
	assert blocker.args[0]["cv_folds"] == 7


def test_controller_respects_no_scaling(sample_dataframe):
	controller = MLController()
	controller.set_dataframe(sample_dataframe)
	results = controller.run_ml_methods(
		target_column="target",
		feature_columns=["f1", "f2", "f3"],
		selected_methods=["Linear Regression"],
		scale_features=False,
		cv_folds=3,
	)
	assert "Linear Regression" in results


def test_cv_folds_capped_to_sample_count(sample_dataframe):
	"""Requesting more folds than samples must not raise; it is capped instead."""
	controller = MLController()
	controller.set_dataframe(sample_dataframe.head(4))
	results = controller.run_ml_methods(
		target_column="target",
		feature_columns=["f1", "f2", "f3"],
		selected_methods=["Linear Regression"],
		cv_folds=10,
	)
	assert "Linear Regression" in results


def test_cache_is_config_aware(sample_dataframe):
	controller = MLController()
	controller.set_dataframe(sample_dataframe)
	cfg_a = {"test_size": 0.2, "random_state": 42, "cv_folds": 3, "scale_features": True}
	controller.run_ml_methods(
		target_column="target",
		feature_columns=["f1", "f2"],
		selected_methods=["Linear Regression"],
		**cfg_a,
	)
	# Same columns but a different config should be a cache miss.
	cfg_b = dict(cfg_a, random_state=7)
	assert controller.get_cached_result("Linear Regression", "target", ["f1", "f2"], cfg_a) is not None
	assert controller.get_cached_result("Linear Regression", "target", ["f1", "f2"], cfg_b) is None
