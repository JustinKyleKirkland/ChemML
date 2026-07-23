"""Tests for the plotting business logic (PlotController) and PlotView wiring."""

import matplotlib

matplotlib.use("Agg")  # Must be set before importing pyplot

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from chemml.ui.controllers.plot_controller import PlotController, PlotSettings
from chemml.ui.views.plot_view import PlotView


@pytest.fixture
def sample_data():
	return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})


@pytest.fixture
def controller(sample_data):
	ctrl = PlotController()
	ctrl.update_data(sample_data)
	return ctrl


class TestPlotController:
	def test_settings_defaults(self):
		settings = PlotSettings()
		assert settings.x_title == "X Axis"
		assert settings.y_title == "Y Axis"

	def test_create_scatter_plot_returns_data(self, controller):
		x, y = controller.create_scatter_plot("x", "y", marker_color="blue")
		np.testing.assert_array_equal(x, [1, 2, 3, 4, 5])
		np.testing.assert_array_equal(y, [2, 4, 6, 8, 10])

	def test_create_scatter_plot_missing_column(self, controller):
		with pytest.raises(ValueError):
			controller.create_scatter_plot("x", "missing")

	def test_calculate_r_squared_perfect_fit(self, controller):
		r2 = controller.calculate_r_squared(np.array([1, 2, 3]), np.array([2, 4, 6]))
		assert 0.99 < r2 <= 1.0

	def test_calculate_r_squared_constant_y(self, controller):
		# No variance in y -> R^2 defined as 0 by the controller
		r2 = controller.calculate_r_squared(np.array([1, 2, 3]), np.array([5, 5, 5]))
		assert r2 == 0

	def test_add_trend_line_requires_two_points(self, controller):
		with pytest.raises(ValueError):
			controller.add_trend_line(np.array([1]), np.array([2]))


class TestPlotView:
	def test_view_initializes_with_data(self, qtbot, sample_data):
		view = PlotView(df=sample_data)
		qtbot.addWidget(view)
		assert view.x_combo.count() == 2
		assert view.y_combo.count() == 2

	def test_update_data_refreshes_combos(self, qtbot, sample_data):
		view = PlotView()
		qtbot.addWidget(view)
		view.update_data(sample_data)
		assert {view.x_combo.itemText(i) for i in range(view.x_combo.count())} == {"x", "y"}

	def test_plot_data_rejects_identical_axes(self, qtbot, sample_data):
		view = PlotView(df=sample_data)
		qtbot.addWidget(view)
		view.x_combo.setCurrentText("x")
		view.y_combo.setCurrentText("x")
		with patch.object(view, "_show_warning") as warn:
			view.plot_data()
			warn.assert_called_once()
