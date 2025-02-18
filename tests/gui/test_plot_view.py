import pandas as pd
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from gui.plot_view import PlottingWidget


@pytest.fixture
def sample_dataframe():
	return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})


def test_plot_data(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	assert widget.current_plot is None

	widget.plot_data()

	assert widget.current_plot is not None


def test_update_plot(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	widget.plot_data()

	assert widget.current_plot is not None

	widget.update_plot()

	assert widget.current_plot is not None


def test_start_timer(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	assert not widget.timer.isActive()

	widget.start_timer()

	assert widget.timer.isActive()


def test_update_data(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	new_dataframe = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})

	widget.update_data(new_dataframe)

	assert widget.df.equals(new_dataframe)


def test_select_marker_color(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	assert widget.marker_color == Qt.black

	widget.select_marker_color()

	assert isinstance(widget.marker_color, QColor)


def test_select_line_fit_color(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	assert widget.line_fit_color == Qt.red

	widget.select_line_fit_color()

	assert isinstance(widget.line_fit_color, QColor)


def test_get_line_style(sample_dataframe, qtbot):
	widget = PlottingWidget(sample_dataframe)
	qtbot.addWidget(widget)

	widget.line_type_combo.setCurrentText("Dashed")
	assert widget.get_line_style() == "--"

	widget.line_type_combo.setCurrentText("Dotted")
	assert widget.get_line_style() == ":"

	widget.line_type_combo.setCurrentText("DashDot")
	assert widget.get_line_style() == "-."

	widget.line_type_combo.setCurrentText("Solid")
	assert widget.get_line_style() == "-"
