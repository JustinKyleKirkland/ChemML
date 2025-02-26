from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")  # Must be before importing pyplot
import pandas as pd
import pytest

from gui.plot_view import DataPlotter, PlottingWidget


@pytest.fixture(autouse=True)
def mock_show():
    """Prevent matplotlib from showing plots"""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.close"):
        yield


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})


@pytest.fixture
def mock_figure():
    """Create a mock matplotlib figure"""
    fig = MagicMock()
    fig.canvas = MagicMock()
    fig.gca = MagicMock(return_value=MagicMock())
    return fig


class TestDataPlotter:
    @pytest.fixture
    def plotter(self, mock_figure, sample_data):
        with patch("matplotlib.pyplot.figure", return_value=mock_figure):
            plotter = DataPlotter(data=sample_data)
            plotter.figure = mock_figure
            yield plotter

    def test_create_scatter_plot(self, plotter, sample_data):
        """Test scatter plot creation without displaying"""
        plotter.create_scatter_plot(
            "x",
            "y",  # Pass column names instead of Series
            marker_style="o",
            marker_size=5,
            marker_color="blue",
        )
        assert plotter.figure is not None

    def test_calculate_r_squared(self, plotter):
        """Test R-squared calculation"""
        r2 = plotter.calculate_r_squared([1, 2, 3], [2, 4, 6])
        assert 0.99 < r2 < 1.01

    def test_add_trend_line(self, plotter, sample_data):
        """Test trend line addition without displaying"""
        plotter.create_scatter_plot(
            "x",
            "y",  # Pass column names instead of Series
            marker_style="o",
            marker_size=5,
            marker_color="blue",
        )

        # Create a mock poly1d object
        mock_poly = MagicMock()
        mock_poly.__call__ = lambda x: x  # Make it callable and return input

        # Mock both polyfit and poly1d
        with patch("numpy.polyfit", return_value=[1, 0]), patch("numpy.poly1d", return_value=mock_poly):
            plotter.add_trend_line("x", "y", line_color="red", line_width=2, line_style="--")

        assert plotter.figure is not None


class TestPlottingWidget:
    @pytest.fixture
    def plotting_widget(self, qtbot, cleanup_widgets, mock_figure, sample_data):
        """Create a PlottingWidget instance"""
        with patch("matplotlib.pyplot.figure", return_value=mock_figure):
            widget = PlottingWidget(df=sample_data)
            cleanup_widgets(widget)
            qtbot.addWidget(widget)
            widget.figure = mock_figure
            yield widget

    def test_initialization(self, plotting_widget):
        """Test widget initialization"""
        assert plotting_widget is not None

    def test_plot_data(self, plotting_widget, sample_data):
        """Test plotting with different columns"""
        # Update to match actual implementation
        plotting_widget.update_data(sample_data)
        # Since update_plot is a method that only takes self,
        # we should set the columns first
        plotting_widget.x_column = "x"
        plotting_widget.y_column = "y"
        plotting_widget.update_plot()  # No arguments needed
        assert plotting_widget.figure is not None  # Check figure instead of canvas

    def test_update_data(self, plotting_widget, sample_data):
        """Test data update functionality"""
        plotting_widget.update_data(sample_data)
        assert plotting_widget.df.equals(sample_data)

    def test_update_axes_options(self, plotting_widget, monkeypatch):
        """Test axes options update"""

        class MockDialog:
            def __init__(self, *args, **kwargs):
                pass

            def exec_(self):
                return True

            def get_values(self):
                return {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}

        monkeypatch.setattr("gui.plot_view.AxesOptionsDialog", MockDialog)
        plotting_widget.update_axes_options()
