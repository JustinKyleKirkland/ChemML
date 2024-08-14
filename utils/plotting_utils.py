import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class PlottingWidget(QDialog):
    current_plot: plt.Figure | None = None

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the PlotDataWindow class.

        Parameters:
        - df: pandas.DataFrame
            The DataFrame containing the data to be plotted.

        Attributes:
        - df: pandas.DataFrame
            The DataFrame containing the data to be plotted.
        - marker_color: QColor
            The color of the markers.
        - line_fit_color: QColor
            The color of the line of best fit.
        - timer: QTimer
            The timer used to update the plot.
        - layout: QVBoxLayout
            The layout of the window.
        - plot_button: QPushButton
            The button used to plot the data.
        - x_combo: QComboBox
            The combo box used to select the X data.
        - y_combo: QComboBox
            The combo box used to select the Y data.
        - marker_type_combo: QComboBox
            The combo box used to select the marker type.
        - marker_color_button: QPushButton
            The button used to select the marker color.
        - marker_size_slider: QSlider
            The slider used to select the marker size.
        - line_fit_checkbox: QCheckBox
            The checkbox used to add a line of best fit.
        - r_squared_checkbox: QCheckBox
            The checkbox used to calculate R².
        - line_fit_color_button: QPushButton
            The button used to select the line of best fit color.
        - line_thickness_slider: QSlider
            The slider used to select the line thickness.
        - line_type_combo: QComboBox
            The combo box used to select the line type.
        """

        super().__init__()

        self.df: pd.DataFrame = df
        self.setWindowTitle("Plot Data")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)
        self.layout.addWidget(self.plot_button)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()

        self.x_combo.addItems(self.df.columns)
        self.y_combo.addItems(self.df.columns)

        self.layout.addWidget(QLabel("Select X Data:"))
        self.layout.addWidget(self.x_combo)
        self.layout.addWidget(QLabel("Select Y Data:"))
        self.layout.addWidget(self.y_combo)

        self.marker_type_combo = QComboBox()
        self.marker_type_combo.addItems(["o", "s", "^", "D"])

        self.layout.addWidget(QLabel("Marker Type:"))
        self.layout.addWidget(self.marker_type_combo)

        self.marker_color_button = QPushButton("Select Marker Color")
        self.marker_color_button.clicked.connect(self.select_marker_color)
        self.layout.addWidget(self.marker_color_button)

        self.marker_size_slider = QSlider(Qt.Horizontal)
        self.marker_size_slider.setRange(1, 20)
        self.marker_size_slider.setValue(5)

        self.layout.addWidget(QLabel("Marker Size:"))
        self.layout.addWidget(self.marker_size_slider)

        self.line_fit_checkbox = QCheckBox("Add Line of Best Fit")
        self.r_squared_checkbox = QCheckBox("Calculate R²")

        self.layout.addWidget(self.line_fit_checkbox)
        self.layout.addWidget(self.r_squared_checkbox)

        self.line_fit_color_button = QPushButton("Select Line of Best Fit Color")
        self.line_fit_color_button.clicked.connect(self.select_line_fit_color)
        self.layout.addWidget(self.line_fit_color_button)

        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setRange(1, 10)
        self.line_thickness_slider.setValue(2)
        self.layout.addWidget(QLabel("Line Thickness:"))
        self.layout.addWidget(self.line_thickness_slider)

        self.line_type_combo = QComboBox()
        self.line_type_combo.addItems(["Solid", "Dashed", "Dotted", "DashDot"])
        self.layout.addWidget(QLabel("Line Type:"))
        self.layout.addWidget(self.line_type_combo)

        self.setLayout(self.layout)

        self.marker_color: QColor = Qt.black
        self.line_fit_color: QColor = Qt.red

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

        self.marker_size_slider.valueChanged.connect(self.start_timer)
        self.line_thickness_slider.valueChanged.connect(self.start_timer)

    def plot_data(self) -> None:
        """
        Plot the data.

        This method creates a new plot and updates it with the current data.
        If there is an existing plot, it is closed before creating a new one.

        Parameters:
            None

        Returns:
            None
        """

        if PlottingWidget.current_plot is not None:
            plt.close(PlottingWidget.current_plot)

        PlottingWidget.current_plot = plt.figure()
        self.update_plot()

        plt.show()

        self.timer.start(200)

    def update_plot(self) -> None:
        """
        Update the plot with the selected data columns and options.

        This method clears the current plot and creates a new scatter plot based on the selected x and y columns.
        The marker type, size, and color are determined by the corresponding combo boxes and sliders.
        If the line fit checkbox is checked, a linear regression line is fitted to the data and plotted.
        The line color, thickness, and style are determined by the corresponding options.
        If the R-squared checkbox is checked, the coefficient of determination (R-squared) is calculated and displayed
        on the plot.

        Parameters:
        - self: The instance of the class.

        Returns:
        - None
        """
        plt.clf()

        x_column: str = self.x_combo.currentText()
        y_column: str = self.y_combo.currentText()

        plt.scatter(
            self.df[x_column],
            self.df[y_column],
            marker=self.marker_type_combo.currentText(),
            s=self.marker_size_slider.value(),
            color=self.marker_color.name()
            if isinstance(self.marker_color, QColor)
            else "black",
        )

        if self.line_fit_checkbox.isChecked():
            z = np.polyfit(self.df[x_column], self.df[y_column], 1)
            p = np.poly1d(z)
            plt.plot(
                self.df[x_column],
                p(self.df[x_column]),
                color=self.line_fit_color.name()
                if isinstance(self.line_fit_color, QColor)
                else "red",
                linewidth=self.line_thickness_slider.value(),
                linestyle=self.get_line_style(),
            )

            if self.r_squared_checkbox.isChecked():
                r_squared = 1 - (
                    np.sum((self.df[y_column] - p(self.df[x_column])) ** 2)
                    / np.sum((self.df[y_column] - np.mean(self.df[y_column])) ** 2)
                )
                plt.text(
                    0.1, 0.9, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes
                )

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title("Data Plot")
        plt.draw()

    def start_timer(self) -> None:
        """
        Starts the timer if it is not already active.

        Parameters:
            None

        Returns:
            None
        """
        if not self.timer.isActive():
            self.timer.start(100)

    def update_data(self, df: pd.DataFrame) -> None:
        """
        Update the data in the plotting utility.

        Parameters:
        - df: pandas.DataFrame
            The new data to be used for plotting.

        Returns:
        None
        """
        self.df = df
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(self.df.columns)
        self.y_combo.addItems(self.df.columns)

    def select_marker_color(self) -> None:
        """
        Opens a color dialog to allow the user to select a marker color.
        The selected color is stored in the `marker_color` attribute of the object.

        Returns:
            None
        """
        color: QColor = QColorDialog.getColor()
        if color.isValid():
            self.marker_color = color

    def select_line_fit_color(self) -> None:
        """
        Opens a color dialog to allow the user to select a color for the line fit.
        If a valid color is selected, it is stored in the `line_fit_color` attribute.
        """
        color: QColor = QColorDialog.getColor()
        if color.isValid():
            self.line_fit_color = color

    def get_line_style(self) -> str:
        """
        Get the line style based on the selected line type.

        Returns:
            str: The line style for plotting.
        """
        line_type: str = self.line_type_combo.currentText()
        if line_type == "Dashed":
            return "--"
        elif line_type == "Dotted":
            return ":"
        elif line_type == "DashDot":
            return "-."
        else:
            return "-"
