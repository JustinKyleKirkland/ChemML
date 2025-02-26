from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
)

from utils.plotting_utils import AxesOptionsDialog, LineFitOptionsDialog, MarkerOptionsDialog


@dataclass
class PlotSettings:
    """Stores plot appearance settings."""

    x_title: str = "X Axis"
    y_title: str = "Y Axis"
    title_size: int = 12
    tick_size: int = 10
    zoom_factor: float = 1.2


class DataPlotter:
    """Handles the actual plotting logic separate from the GUI."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.figure: plt.Figure | None = None
        self.original_xlim: tuple[float, float] | None = None
        self.original_ylim: tuple[float, float] | None = None

    def create_scatter_plot(
        self,
        x_column: str,
        y_column: str,
        marker_style: str,
        marker_size: int,
        marker_color: QColor,
    ) -> None:
        """Creates a scatter plot with the given parameters."""
        x_data = self.data[x_column].values
        y_data = self.data[y_column].values

        plt.scatter(
            x_data,
            y_data,
            marker=marker_style,
            s=marker_size,
            color=marker_color.name() if isinstance(marker_color, QColor) else "red",
        )
        return x_data, y_data

    def add_trend_line(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        line_color: QColor,
        line_width: int,
        line_style: str,
    ) -> None:
        """Adds a trend line to the current plot."""
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        plt.plot(
            x_data,
            p(x_data),
            color=line_color.name() if isinstance(line_color, QColor) else "black",
            linewidth=line_width,
            linestyle=line_style,
        )

    def calculate_r_squared(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """Calculates the R-squared value for the current plot."""
        if len(x_data) <= 1:
            return 0

        mean_y = np.mean(y_data)
        ss_tot = np.sum((y_data - mean_y) ** 2)

        if ss_tot == 0:
            return 0

        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ss_res = np.sum((y_data - p(x_data)) ** 2)

        return 1 - (ss_res / ss_tot)


class PlottingWidget(QDialog):
    """Main plotting widget that handles the GUI and coordinates with DataPlotter."""

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.plotter = DataPlotter(df)
        self.settings = PlotSettings()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Initialize all UI elements."""
        self.setWindowTitle("Plot Data")
        self.setGeometry(100, 100, 500, 350)

        self.layout = QVBoxLayout()
        self._setup_data_selectors()
        self._setup_option_buttons()
        self._setup_plot_button()
        self.setLayout(self.layout)

        # Initialize option dialogs
        self.marker_options_dialog = MarkerOptionsDialog(self)
        self.line_fit_options_dialog = LineFitOptionsDialog(self)
        self.axes_options_dialog = AxesOptionsDialog(self)

    def _setup_data_selectors(self) -> None:
        """Setup the X and Y data selection combos."""
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()

        for combo in (self.x_combo, self.y_combo):
            combo.addItems(self.df.columns)

        for label, combo in [("Select X Data:", self.x_combo), ("Select Y Data:", self.y_combo)]:
            self.layout.addWidget(QLabel(label))
            self.layout.addWidget(combo)

    def _setup_option_buttons(self) -> None:
        """Setup the options buttons section."""
        plot_options_label = QLabel("----- Plot Options -----")
        plot_options_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(plot_options_label)

        buttons_layout = QHBoxLayout()

        option_buttons = {
            "Marker Options": self.open_marker_options_dialog,
            "Line of Best Fit Options": self.open_line_fit_options_dialog,
            "Axes Options": self.open_axes_options_dialog,
        }

        for text, callback in option_buttons.items():
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons_layout.addWidget(button)

        self.layout.addLayout(buttons_layout)

    def _setup_plot_button(self) -> None:
        """Setup the main plot button."""
        plot_button_layout = QHBoxLayout()
        plot_button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)
        self.plot_button.setStyleSheet(
            "background-color: #007AFF; color: white; " "border-radius: 10px; padding: 10px; border: none;"
        )

        plot_button_layout.addWidget(self.plot_button)
        self.layout.addLayout(plot_button_layout)

    def _connect_signals(self) -> None:
        """Connect all signal handlers."""
        self.axes_options_dialog.options_applied.connect(self.update_axes_options)
        self.marker_options_dialog.options_applied.connect(self.update_plot)
        self.line_fit_options_dialog.options_applied.connect(self.update_plot)

    def plot_data(self) -> None:
        """Create and display the plot."""
        if self.plotter.figure is not None:
            plt.close(self.plotter.figure)

        if self.x_combo.currentText() == self.y_combo.currentText():
            self._show_warning("Invalid Data", "X and Y data cannot be the same.")
            return

        self.plotter.figure = plt.figure()
        self.update_plot()
        plt.show()

        # Connect matplotlib event handlers
        fig = plt.gcf()
        fig.canvas.mpl_connect("scroll_event", self._handle_scroll)
        fig.canvas.mpl_connect("button_press_event", self._handle_button_press)

    def update_plot(self) -> None:
        """Update the current plot with new settings."""
        plt.clf()

        x_data, y_data = self.plotter.create_scatter_plot(
            self.x_combo.currentText(),
            self.y_combo.currentText(),
            self.marker_options_dialog.get_marker_symbol(),
            self.marker_options_dialog.marker_size_slider.value(),
            self.marker_options_dialog.marker_color,
        )

        if self.line_fit_options_dialog.line_fit_checkbox.isChecked():
            self.plotter.add_trend_line(
                x_data,
                y_data,
                self.line_fit_options_dialog.line_fit_color,
                self.line_fit_options_dialog.line_thickness_slider.value(),
                self._get_line_style(),
            )

        if self.line_fit_options_dialog.r_squared_checkbox.isChecked():
            r_squared = self.plotter.calculate_r_squared(x_data, y_data)
            plt.text(0.1, 0.9, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes)

        self._update_plot_appearance()

    def _update_plot_appearance(self) -> None:
        """Update plot labels and appearance settings."""
        plt.xlabel(self.settings.x_title, fontsize=self.settings.title_size)
        plt.ylabel(self.settings.y_title, fontsize=self.settings.title_size)
        plt.tick_params(axis="both", which="major", labelsize=self.settings.tick_size)
        plt.draw()

    def update_data(self, df: pd.DataFrame) -> None:
        """Update the widget with new data."""
        self.df = df
        self.plotter.data = df
        for combo in (self.x_combo, self.y_combo):
            combo.clear()
            combo.addItems(self.df.columns)

    def _get_line_style(self) -> str:
        """Get the line style based on the current selection."""
        styles = {
            "Dashed": "--",
            "Dotted": ":",
            "DashDot": "-.",
        }
        return styles.get(self.line_fit_options_dialog.line_type_combo.currentText(), "-")

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _handle_scroll(self, event) -> None:
        """Handle mouse scroll events for zooming."""
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        scale_factor = (
            1 / self.settings.zoom_factor
            if event.button == "up"
            else self.settings.zoom_factor
            if event.button == "down"
            else 1
        )

        new_xlim = [x + (x - xlim[0]) * (1 - scale_factor) for x in xlim]
        new_ylim = [y + (y - ylim[0]) * (1 - scale_factor) for y in ylim]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        plt.draw()

    def _handle_button_press(self, event) -> None:
        """Handle mouse button press events."""
        if event.button == 3:  # Right click
            if self.plotter.original_xlim and self.plotter.original_ylim:
                plt.gca().set_xlim(self.plotter.original_xlim)
                plt.gca().set_ylim(self.plotter.original_ylim)
                plt.draw()
        else:
            self.plotter.original_xlim = plt.gca().get_xlim()
            self.plotter.original_ylim = plt.gca().get_ylim()

    # Dialog openers
    def open_marker_options_dialog(self) -> None:
        self.marker_options_dialog.exec_()

    def open_line_fit_options_dialog(self) -> None:
        self.line_fit_options_dialog.exec_()

    def open_axes_options_dialog(self) -> None:
        self.axes_options_dialog.exec_()

    def update_axes_options(self) -> None:
        """Update plot settings from axes options dialog."""
        self.settings.x_title = self.axes_options_dialog.x_title
        self.settings.y_title = self.axes_options_dialog.y_title
        self.settings.title_size = self.axes_options_dialog.title_size
        self.settings.tick_size = self.axes_options_dialog.tick_size
        self.update_plot()
