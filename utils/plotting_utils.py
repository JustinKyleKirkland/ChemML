# utils/plotting_utils.py
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class PlottingWidget(QDialog):
    current_plot = None  # Class variable to keep track of the current plot

    def __init__(self, df):
        super().__init__()

        self.df = df
        self.setWindowTitle("Plot Data")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()

        # Populate the X and Y data dropdowns with DataFrame columns
        self.x_combo.addItems(self.df.columns)
        self.y_combo.addItems(self.df.columns)

        self.layout.addWidget(QLabel("Select X Data:"))
        self.layout.addWidget(self.x_combo)
        self.layout.addWidget(QLabel("Select Y Data:"))
        self.layout.addWidget(self.y_combo)

        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_data)

        self.layout.addWidget(self.plot_button)

        # Additional options for plotting
        self.marker_type_combo = QComboBox()
        self.marker_type_combo.addItems(["o", "s", "^", "D"])  # Example markers

        self.layout.addWidget(QLabel("Marker Type:"))
        self.layout.addWidget(self.marker_type_combo)

        self.marker_size_slider = QSlider(Qt.Horizontal)
        self.marker_size_slider.setRange(1, 20)
        self.marker_size_slider.setValue(5)

        self.layout.addWidget(QLabel("Marker Size:"))
        self.layout.addWidget(self.marker_size_slider)

        self.line_fit_checkbox = QCheckBox("Add Line of Best Fit")
        self.r_squared_checkbox = QCheckBox("Calculate R²")

        self.layout.addWidget(self.line_fit_checkbox)
        self.layout.addWidget(self.r_squared_checkbox)

        self.setLayout(self.layout)

        # Timer for dynamic updating
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

        # Connect the slider to start the timer for dynamic updates
        self.marker_size_slider.valueChanged.connect(self.start_timer)

    def plot_data(self):
        # Close the existing plot if it's open
        if PlottingWidget.current_plot is not None:
            plt.close(PlottingWidget.current_plot)

        PlottingWidget.current_plot = (
            plt.figure()
        )  # Create a new plot and store the reference
        self.update_plot()  # Initial plot

        plt.show()

        # Start the timer to update the plot dynamically
        self.timer.start(100)  # Update every 100 milliseconds

    def update_plot(self):
        # Clear the previous plot
        plt.clf()  # Clear the current figure

        x_column = self.x_combo.currentText()
        y_column = self.y_combo.currentText()

        # Plotting the data
        plt.scatter(
            self.df[x_column],
            self.df[y_column],
            marker=self.marker_type_combo.currentText(),
            s=self.marker_size_slider.value(),
        )

        if self.line_fit_checkbox.isChecked():
            z = np.polyfit(self.df[x_column], self.df[y_column], 1)
            p = np.poly1d(z)
            plt.plot(self.df[x_column], p(self.df[x_column]), color="red")

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
        plt.draw()  # Redraw the updated plot

    def start_timer(self):
        # Start the timer to update the plot when the slider changes
        if not self.timer.isActive():
            self.timer.start(100)  # Update every 100 milliseconds

    def update_data(self, df):
        self.df = df
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(self.df.columns)
        self.y_combo.addItems(self.df.columns)
