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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class MarkerOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Marker Options")
        self.setGeometry(100, 100, 200, 200)

        layout = QVBoxLayout()

        self.marker_type_combo = QComboBox()
        self.marker_type_combo.addItems(["o", "s", "^", "D"])
        layout.addWidget(QLabel("Marker Type:"))
        layout.addWidget(self.marker_type_combo)

        self.marker_color_button = QPushButton("Select Marker Color")
        self.marker_color_button.clicked.connect(self.select_marker_color)
        layout.addWidget(self.marker_color_button)

        self.marker_size_slider = QSlider(Qt.Horizontal)
        self.marker_size_slider.setRange(1, 20)
        self.marker_size_slider.setValue(5)
        layout.addWidget(QLabel("Marker Size:"))
        layout.addWidget(self.marker_size_slider)

        self.marker_color: QColor = Qt.red

        self.setLayout(layout)

    def select_marker_color(self) -> None:
        color: QColor = QColorDialog.getColor()
        if color.isValid():
            self.marker_color = color


class LineFitOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Line of Best Fit Options")
        self.setGeometry(100, 100, 200, 200)

        layout = QVBoxLayout()

        self.line_fit_checkbox = QCheckBox("Add Line of Best Fit")
        layout.addWidget(self.line_fit_checkbox)

        self.r_squared_checkbox = QCheckBox("Calculate R²")
        layout.addWidget(self.r_squared_checkbox)

        self.line_fit_color_button = QPushButton("Select Line of Best Fit Color")
        self.line_fit_color_button.clicked.connect(self.select_line_fit_color)
        layout.addWidget(self.line_fit_color_button)

        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setRange(1, 10)
        self.line_thickness_slider.setValue(2)
        layout.addWidget(QLabel("Line Thickness:"))
        layout.addWidget(self.line_thickness_slider)

        self.line_type_combo = QComboBox()
        self.line_type_combo.addItems(["Solid", "Dashed", "Dotted", "DashDot"])
        layout.addWidget(QLabel("Line Type:"))
        layout.addWidget(self.line_type_combo)

        self.line_fit_color: QColor = Qt.black

        self.setLayout(layout)

    def select_line_fit_color(self) -> None:
        color: QColor = QColorDialog.getColor()
        if color.isValid():
            self.line_fit_color = color


class PlottingWidget(QDialog):
    current_plot: plt.Figure | None = None

    def __init__(self, df: pd.DataFrame) -> None:
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

        # Add the plot options label
        plot_options_label = QLabel("----- Plot Options -----")
        plot_options_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(plot_options_label)

        # Create a horizontal layout for the buttons
        buttons_layout = QHBoxLayout()

        # Marker Options Button
        self.marker_options_button = QPushButton("Marker Options")
        self.marker_options_button.clicked.connect(self.open_marker_options_dialog)
        buttons_layout.addWidget(self.marker_options_button)

        # Line of Best Fit Options Button
        self.line_fit_options_button = QPushButton("Line of Best Fit Options")
        self.line_fit_options_button.clicked.connect(self.open_line_fit_options_dialog)
        buttons_layout.addWidget(self.line_fit_options_button)

        # Add the horizontal layout to the main layout
        self.layout.addLayout(buttons_layout)

        self.setLayout(self.layout)

        self.timer: QTimer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

        # Initialize dialogs
        self.marker_options_dialog = MarkerOptionsDialog(self)
        self.line_fit_options_dialog = LineFitOptionsDialog(self)

    def open_marker_options_dialog(self) -> None:
        self.marker_options_dialog.exec_()

    def open_line_fit_options_dialog(self) -> None:
        self.line_fit_options_dialog.exec_()

    def plot_data(self) -> None:
        if PlottingWidget.current_plot is not None:
            plt.close(PlottingWidget.current_plot)

        PlottingWidget.current_plot = plt.figure()
        self.update_plot()

        plt.show()

        self.timer.start(200)

    def update_plot(self) -> None:
        plt.clf()

        x_column: str = self.x_combo.currentText()
        y_column: str = self.y_combo.currentText()

        plt.scatter(
            self.df[x_column],
            self.df[y_column],
            marker=self.marker_options_dialog.marker_type_combo.currentText(),
            s=self.marker_options_dialog.marker_size_slider.value(),
            color=self.marker_options_dialog.marker_color.name()
            if isinstance(self.marker_options_dialog.marker_color, QColor)
            else "black",
        )

        if self.line_fit_options_dialog.line_fit_checkbox.isChecked():
            z = np.polyfit(self.df[x_column], self.df[y_column], 1)
            p = np.poly1d(z)
            plt.plot(
                self.df[x_column],
                p(self.df[x_column]),
                color=self.line_fit_options_dialog.line_fit_color.name()
                if isinstance(self.line_fit_options_dialog.line_fit_color, QColor)
                else "red",
                linewidth=self.line_fit_options_dialog.line_thickness_slider.value(),
                linestyle=self.get_line_style(),
            )

            if self.line_fit_options_dialog.r_squared_checkbox.isChecked():
                r_squared = 1 - (
                    np.sum((self.df[y_column] - p(self.df[x_column])) ** 2)
                    / np.sum((self.df[y_column] - np.mean(self.df[y_column])) ** 2)
                )
                plt.text(0.1, 0.9, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes)

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title("Data Plot")
        plt.draw()

    def start_timer(self) -> None:
        if not self.timer.isActive():
            self.timer.start(100)

    def update_data(self, df: pd.DataFrame) -> None:
        self.df = df
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(self.df.columns)
        self.y_combo.addItems(self.df.columns)

    def get_line_style(self) -> str:
        line_type: str = self.line_fit_options_dialog.line_type_combo.currentText()
        if line_type == "Dashed":
            return "--"
        elif line_type == "Dotted":
            return ":"
        elif line_type == "DashDot":
            return "-."
        else:
            return "-"
