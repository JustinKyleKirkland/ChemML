import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from gui.plot_view import DataPlotter, PlotSettings, PlottingWidget


@pytest.fixture
def sample_dataframe():
	return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})


@pytest.fixture
def plot_widget(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)
	return widget


class TestDataPlotter:
	def test_create_scatter_plot(self, sample_dataframe):
		plotter = DataPlotter(sample_dataframe)
		x_data, y_data = plotter.create_scatter_plot("x", "y", "o", 10, QColor(Qt.black))

		assert np.array_equal(x_data, sample_dataframe["x"].values)
		assert np.array_equal(y_data, sample_dataframe["y"].values)
		plt.close()

	def test_calculate_r_squared(self, sample_dataframe):
		plotter = DataPlotter(sample_dataframe)
		x_data = sample_dataframe["x"].values
		y_data = sample_dataframe["y"].values

		r_squared = plotter.calculate_r_squared(x_data, y_data)
		assert r_squared == pytest.approx(1.0)  # Perfect linear relationship

		# Test with single point
		plotter.data = pd.DataFrame({"x": [1], "y": [2]})
		assert plotter.calculate_r_squared(np.array([1]), np.array([2])) == 0

	def test_add_trend_line(self, sample_dataframe):
		plotter = DataPlotter(sample_dataframe)
		x_data = sample_dataframe["x"].values
		y_data = sample_dataframe["y"].values

		plotter.add_trend_line(x_data, y_data, QColor(Qt.black), 1, "-")
		# Verify trend line was added (matplotlib doesn't provide easy way to check)
		assert len(plt.gca().lines) == 1
		plt.close()


class TestPlottingWidget:
	def test_initialization(self, plot_widget, sample_dataframe):
		assert plot_widget.df.equals(sample_dataframe)
		assert isinstance(plot_widget.plotter, DataPlotter)
		assert isinstance(plot_widget.settings, PlotSettings)

	def test_plot_data_same_columns(self, plot_widget, qtbot):
		plot_widget.x_combo.setCurrentText("x")
		plot_widget.y_combo.setCurrentText("x")

		plot_widget.plot_data()
		assert plot_widget.plotter.figure is None

	def test_plot_data_different_columns(self, plot_widget, qtbot):
		plot_widget.x_combo.setCurrentText("x")
		plot_widget.y_combo.setCurrentText("y")

		plot_widget.plot_data()
		assert plot_widget.plotter.figure is not None
		plt.close()

	def test_update_data(self, plot_widget):
		new_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
		plot_widget.update_data(new_df)

		assert plot_widget.df.equals(new_df)
		assert plot_widget.plotter.data.equals(new_df)
		assert plot_widget.x_combo.count() == 2
		assert plot_widget.y_combo.count() == 2

	def test_get_line_style(self, plot_widget):
		styles = {"Dashed": "--", "Dotted": ":", "DashDot": "-.", "Solid": "-"}

		for style_name, expected in styles.items():
			plot_widget.line_fit_options_dialog.line_type_combo.setCurrentText(style_name)
			assert plot_widget._get_line_style() == expected

	def test_update_axes_options(self, plot_widget):
		plot_widget.axes_options_dialog.x_title = "New X"
		plot_widget.axes_options_dialog.y_title = "New Y"
		plot_widget.axes_options_dialog.title_size = 14
		plot_widget.axes_options_dialog.tick_size = 12

		plot_widget.update_axes_options()

		assert plot_widget.settings.x_title == "New X"
		assert plot_widget.settings.y_title == "New Y"
		assert plot_widget.settings.title_size == 14
		assert plot_widget.settings.tick_size == 12

	def test_handle_scroll(self, plot_widget, qtbot):
		plot_widget.x_combo.setCurrentText("x")
		plot_widget.y_combo.setCurrentText("y")
		plot_widget.plot_data()

		class MockEvent:
			def __init__(self, button):
				self.button = button

		# Test zoom out (scroll up)
		original_xlim = plt.gca().get_xlim()
		plot_widget._handle_scroll(MockEvent("up"))
		new_xlim = plt.gca().get_xlim()
		# For zoom out, the new range should be larger
		assert (new_xlim[1] - new_xlim[0]) > (original_xlim[1] - original_xlim[0])
		plt.close()

	def test_handle_button_press(self, plot_widget):
		plot_widget.x_combo.setCurrentText("x")
		plot_widget.y_combo.setCurrentText("y")
		plot_widget.plot_data()

		class MockEvent:
			def __init__(self, button):
				self.button = button

		# Store original view
		plot_widget._handle_button_press(MockEvent(1))
		original_xlim = plt.gca().get_xlim()

		# Change view
		plt.gca().set_xlim(0, 10)

		# Reset view
		plot_widget._handle_button_press(MockEvent(3))
		assert plt.gca().get_xlim() == original_xlim

		plt.close()
