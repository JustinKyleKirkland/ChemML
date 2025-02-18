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


class PlottingWidget(QDialog):
	current_plot: plt.Figure | None = None

	def __init__(self, df: pd.DataFrame) -> None:
		super().__init__()

		self.df: pd.DataFrame = df
		self.setWindowTitle("Plot Data")
		self.setGeometry(100, 100, 500, 350)
		self.zoom_factor = 1.2
		self.original_xlim = None
		self.original_ylim = None

		self.layout = QVBoxLayout()

		self.x_combo = QComboBox()
		self.y_combo = QComboBox()

		self.x_combo.addItems(self.df.columns)
		self.y_combo.addItems(self.df.columns)

		self.layout.addWidget(QLabel("Select X Data:"))
		self.layout.addWidget(self.x_combo)
		self.layout.addWidget(QLabel("Select Y Data:"))
		self.layout.addWidget(self.y_combo)

		plot_options_label = QLabel("----- Plot Options -----")
		plot_options_label.setAlignment(Qt.AlignCenter)
		self.layout.addWidget(plot_options_label)

		buttons_layout = QHBoxLayout()

		self.marker_options_button = QPushButton("Marker Options")
		self.marker_options_button.clicked.connect(self.open_marker_options_dialog)
		buttons_layout.addWidget(self.marker_options_button)

		self.line_fit_options_button = QPushButton("Line of Best Fit Options")
		self.line_fit_options_button.clicked.connect(self.open_line_fit_options_dialog)
		buttons_layout.addWidget(self.line_fit_options_button)

		self.axes_options_button = QPushButton("Axes Options")
		self.axes_options_button.clicked.connect(self.open_axes_options_dialog)
		buttons_layout.addWidget(self.axes_options_button)

		self.layout.addLayout(buttons_layout)

		plot_button_layout = QHBoxLayout()
		plot_button_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

		self.plot_button = QPushButton("Plot")
		self.plot_button.clicked.connect(self.plot_data)
		self.plot_button.setStyleSheet(
			"background-color: #007AFF; " "color: white; " "border-radius: 10px; " "padding: 10px; " "border: none; "
		)

		plot_button_layout.addWidget(self.plot_button)

		self.layout.addLayout(plot_button_layout)

		self.setLayout(self.layout)

		self.marker_options_dialog = MarkerOptionsDialog(self)
		self.line_fit_options_dialog = LineFitOptionsDialog(self)
		self.axes_options_dialog = AxesOptionsDialog(self)

		self.axes_options_dialog.options_applied.connect(self.update_axes_options)
		self.marker_options_dialog.options_applied.connect(self.update_plot)
		self.line_fit_options_dialog.options_applied.connect(self.update_plot)

		self.x_title = "X Axis"
		self.y_title = "Y Axis"
		self.title_size = 12
		self.tick_size = 10

	def open_marker_options_dialog(self) -> None:
		self.marker_options_dialog.exec_()

	def open_line_fit_options_dialog(self) -> None:
		self.line_fit_options_dialog.exec_()

	def open_axes_options_dialog(self) -> None:
		self.axes_options_dialog.exec_()

	def update_axes_options(self) -> None:
		self.x_title = self.axes_options_dialog.x_title
		self.y_title = self.axes_options_dialog.y_title
		self.title_size = self.axes_options_dialog.title_size
		self.tick_size = self.axes_options_dialog.tick_size
		self.update_plot()

	def plot_data(self) -> None:
		if PlottingWidget.current_plot is not None:
			plt.close(PlottingWidget.current_plot)

		if self.x_combo.currentText() == self.y_combo.currentText():
			self.show_warning_message("Invalid Data", "X and Y data cannot be the same.")
			return

		PlottingWidget.current_plot = plt.figure()
		self.update_plot()

		plt.show()

		self.cid_scroll = plt.gcf().canvas.mpl_connect("scroll_event", self.on_scroll)
		self.cis_button_press = plt.gcf().canvas.mpl_connect("button_press_event", self.on_button_press)

	def update_plot(self) -> None:
		plt.clf()

		x_column: str = self.x_combo.currentText()
		y_column: str = self.y_combo.currentText()

		x = self.df[x_column].values
		y = self.df[y_column].values

		if len(x) > 1:
			mean_y = np.mean(y)
			ss_tot = np.sum((y - mean_y) ** 2)
			if self.line_fit_options_dialog.r_squared_checkbox.isChecked():
				z = np.polyfit(x, y, 1)
				p = np.poly1d(z)
				ss_res = np.sum((y - p(x)) ** 2)
				r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
			else:
				ss_res = np.sum((y - mean_y) ** 2)
				r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
		else:
			r_squared = 0

		plt.scatter(
			x,
			y,
			marker=self.marker_options_dialog.get_marker_symbol(),
			s=self.marker_options_dialog.marker_size_slider.value(),
			color=self.marker_options_dialog.marker_color.name()
			if isinstance(self.marker_options_dialog.marker_color, QColor)
			else "red",
		)

		if self.line_fit_options_dialog.line_fit_checkbox.isChecked():
			z = np.polyfit(x, y, 1)
			p = np.poly1d(z)
			plt.plot(
				x,
				p(x),
				color=self.line_fit_options_dialog.line_fit_color.name()
				if isinstance(self.line_fit_options_dialog.line_fit_color, QColor)
				else "black",
				linewidth=self.line_fit_options_dialog.line_thickness_slider.value(),
				linestyle=self.get_line_style(),
			)

		if self.line_fit_options_dialog.r_squared_checkbox.isChecked():
			plt.text(0.1, 0.9, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes)

		plt.xlabel(self.x_title, fontsize=self.title_size)
		plt.ylabel(self.y_title, fontsize=self.title_size)
		plt.tick_params(axis="both", which="major", labelsize=self.tick_size)
		plt.draw()

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

	def show_warning_message(self, title: str, message: str) -> None:
		msg_box = QMessageBox(self)
		msg_box.setIcon(QMessageBox.Warning)
		msg_box.setWindowTitle(title)
		msg_box.setText(message)
		msg_box.setStandardButtons(QMessageBox.Ok)
		msg_box.exec_()

	def on_scroll(self, event) -> None:
		ax = plt.gca()
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()

		if event.button == "up":
			scale_factor = 1 / self.zoom_factor
		elif event.button == "down":
			scale_factor = self.zoom_factor
		else:
			scale_factor = 1

		new_xlim = [x + (x - xlim[0]) * (1 - scale_factor) for x in xlim]
		new_ylim = [y + (y - ylim[0]) * (1 - scale_factor) for y in ylim]

		ax.set_xlim(new_xlim)
		ax.set_ylim(new_ylim)

		plt.draw()

	def on_button_press(self, event) -> None:
		if event.button == 3:
			if self.original_xlim is not None and self.original_ylim is not None:
				plt.gca().set_xlim(self.original_xlim)
				plt.gca().set_ylim(self.original_ylim)
				plt.draw()
		else:
			self.original_xlim = plt.gca().get_xlim()
			self.original_ylim = plt.gca().get_ylim()
