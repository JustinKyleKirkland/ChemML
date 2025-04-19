import logging

import matplotlib.pyplot as plt
import pandas as pd

# Import from PyQt5 packages with full paths to avoid import conflicts
from PyQt5.QtCore import Qt  # Core functionality
from PyQt5.QtCore import QSize  # Import QSize separately to avoid conflicts
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
	QComboBox,
	QDialog,
	QHBoxLayout,
	QLabel,
	QMessageBox,
	QPushButton,
	QVBoxLayout,
	QGroupBox,
	QFrame,
	QGridLayout,
)

from chemml.ui.controllers.plot_controller import PlotController
from chemml.ui.widgets.plot_dialogs import AxesOptionsDialog, LineFitOptionsDialog, MarkerOptionsDialog
from chemml.ui.assets.icons import create_plot_icon
from chemml.ui.assets.colors import COLORS


class PlotView(QDialog):
	"""
	View component for the plotting functionality that interacts with PlotController.

	This view provides a UI for:
	- Selecting data columns to plot
	- Customizing plot appearance
	- Displaying and interacting with plots
	"""

	def __init__(self, df: pd.DataFrame = None) -> None:
		"""
		Initialize the plot view.

		Args:
		    df: Initial DataFrame to use for plotting (optional)
		"""
		super().__init__()
		self.logger = logging.getLogger("PlotView")

		# Initialize the controller
		self.controller = PlotController()
		if df is not None:
			self.controller.update_data(df)

		self._setup_ui()
		self._connect_signals()

	def _setup_ui(self) -> None:
		"""Initialize all UI elements with modern appearance."""
		self.setWindowTitle("Data Visualization")
		self.setGeometry(100, 100, 800, 600)

		# Main layout
		self.layout = QVBoxLayout()
		self.layout.setSpacing(10)
		self.layout.setContentsMargins(15, 15, 15, 15)

		# Create header with icon and title
		header_layout = QHBoxLayout()

		# Add icon
		icon_label = QLabel()
		icon_pixmap = create_plot_icon().pixmap(32, 32)
		icon_label.setPixmap(icon_pixmap)
		header_layout.addWidget(icon_label)

		# Add title
		title_label = QLabel("Data Visualization")
		title_font = QFont()
		title_font.setPointSize(16)
		title_font.setBold(True)
		title_label.setFont(title_font)
		title_label.setStyleSheet(f"color: {COLORS['primary']};")
		header_layout.addWidget(title_label)

		header_layout.addStretch(1)
		self.layout.addLayout(header_layout)

		# Add separator line
		separator = QFrame()
		separator.setFrameShape(QFrame.HLine)
		separator.setFrameShadow(QFrame.Sunken)
		separator.setStyleSheet(f"background-color: {COLORS['light']}; max-height: 1px;")
		self.layout.addWidget(separator)

		# Create data selection panel
		self._create_data_selection_panel()

		# Create options panel
		self._create_options_panel()

		# Add plot button
		self._setup_plot_button()

		# Set layout
		self.setLayout(self.layout)

		# Initialize option dialogs
		self.marker_options_dialog = MarkerOptionsDialog(self)
		self.line_fit_options_dialog = LineFitOptionsDialog(self)
		self.axes_options_dialog = AxesOptionsDialog(self)

	def _create_data_selection_panel(self) -> None:
		"""Create the data selection panel with improved styling."""
		data_group = QGroupBox("Data Selection")
		data_group.setStyleSheet("QGroupBox { font-weight: bold; }")

		data_layout = QGridLayout()
		data_layout.setVerticalSpacing(10)
		data_layout.setHorizontalSpacing(15)

		# Add column selectors with labels
		self.x_combo = QComboBox()
		self.y_combo = QComboBox()

		# Ensure combos have a nice height
		self.x_combo.setMinimumHeight(30)
		self.y_combo.setMinimumHeight(30)

		# Default items if data exists
		if not self.controller.df.empty:
			for combo in (self.x_combo, self.y_combo):
				combo.addItems(self.controller.df.columns)

		# Add to grid layout with proper alignment
		data_layout.addWidget(QLabel("X-Axis Data:"), 0, 0, Qt.AlignRight)
		data_layout.addWidget(self.x_combo, 0, 1)
		data_layout.addWidget(QLabel("Y-Axis Data:"), 1, 0, Qt.AlignRight)
		data_layout.addWidget(self.y_combo, 1, 1)

		# Set layout for the group
		data_group.setLayout(data_layout)
		self.layout.addWidget(data_group)

	def _create_options_panel(self) -> None:
		"""Create the plot options panel with improved styling."""
		options_group = QGroupBox("Plot Options")
		options_group.setStyleSheet("QGroupBox { font-weight: bold; }")

		options_layout = QGridLayout()

		# Create option buttons with icons and better styling
		options = [
			("Marker Style", self.open_marker_options_dialog, "Configure point markers (shape, size, color)"),
			("Trend Line", self.open_line_fit_options_dialog, "Add and configure line of best fit"),
			("Axes Settings", self.open_axes_options_dialog, "Configure axes titles and scaling"),
		]

		# Create buttons in a cleaner grid layout
		for i, (text, callback, tooltip) in enumerate(options):
			button = QPushButton(text)
			button.setMinimumHeight(35)
			button.setToolTip(tooltip)
			button.clicked.connect(callback)
			options_layout.addWidget(button, i // 2, i % 2)

		options_group.setLayout(options_layout)
		self.layout.addWidget(options_group)

	def _setup_plot_button(self) -> None:
		"""Setup the main plot button with improved appearance."""
		plot_button_layout = QHBoxLayout()
		plot_button_layout.addStretch(1)

		# Instructions label
		instructions = QLabel("Select data columns and configure options, then click Plot to visualize")
		instructions.setStyleSheet(f"color: {COLORS['secondary']};")
		plot_button_layout.addWidget(instructions)

		plot_button_layout.addStretch(1)

		# Create an attractive plot button
		self.plot_button = QPushButton("  Generate Plot  ")
		self.plot_button.setIcon(create_plot_icon())
		self.plot_button.setIconSize(QSize(24, 24))
		self.plot_button.clicked.connect(self.plot_data)
		self.plot_button.setMinimumHeight(40)
		self.plot_button.setStyleSheet(
			f"background-color: {COLORS['primary']}; "
			f"color: white; "
			f"border-radius: 5px; "
			f"padding: 8px 16px; "
			f"font-weight: bold; "
			f"border: none;"
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
		if self.controller.figure is not None:
			plt.close(self.controller.figure)

		x_column = self.x_combo.currentText()
		y_column = self.y_combo.currentText()

		if not x_column or not y_column:
			self._show_warning("Missing Data", "Please select both X and Y data columns.")
			return

		if x_column == y_column:
			self._show_warning("Invalid Data", "X and Y data cannot be the same.")
			return

		try:
			self.controller.figure = plt.figure()
			self.update_plot()
			plt.show()

			# Connect matplotlib event handlers
			fig = plt.gcf()
			fig.canvas.mpl_connect("scroll_event", self._handle_scroll)
			fig.canvas.mpl_connect("button_press_event", self._handle_button_press)

		except Exception as e:
			self.logger.error(f"Error plotting data: {e}")
			self._show_warning("Plot Error", str(e))

	def update_plot(self) -> None:
		"""Update the current plot with new settings."""
		try:
			plt.clf()

			# Get marker settings
			marker_symbol = self.marker_options_dialog.get_marker_symbol()
			marker_size = self.marker_options_dialog.marker_size_slider.value()

			# Safely get marker color - handles both QColor and GlobalColor
			marker_color = self._get_color_string(self.marker_options_dialog.marker_color)

			# Create scatter plot
			x_data, y_data = self.controller.create_scatter_plot(
				self.x_combo.currentText(),
				self.y_combo.currentText(),
				marker_style=marker_symbol,
				marker_size=marker_size,
				marker_color=marker_color,
			)

			# Add trend line if requested
			if self.line_fit_options_dialog.line_fit_checkbox.isChecked():
				# Safely get line color - handles both QColor and GlobalColor
				line_color = self._get_color_string(self.line_fit_options_dialog.line_fit_color)
				line_width = self.line_fit_options_dialog.line_thickness_slider.value()
				line_style = self._get_line_style()

				self.controller.add_trend_line(
					x_data,
					y_data,
					line_color=line_color,
					line_width=line_width,
					line_style=line_style,
				)

			# Add R-squared if requested
			if self.line_fit_options_dialog.r_squared_checkbox.isChecked():
				r_squared = self.controller.calculate_r_squared(x_data, y_data)
				plt.text(0.1, 0.9, f"R² = {r_squared:.2f}", transform=plt.gca().transAxes)

			# Update plot appearance
			self.controller.update_plot_appearance()

		except Exception as e:
			self.logger.error(f"Error updating plot: {e}")
			self._show_warning("Plot Update Error", str(e))

	def _get_color_string(self, color) -> str:
		"""
		Safely convert a PyQt color object to a string representation.

		Args:
		    color: A QColor object or GlobalColor enum

		Returns:
		    String representation of the color (#RRGGBB or color name)
		"""
		try:
			# If it has a name() method (QColor)
			return color.name()
		except AttributeError:
			# If it's a GlobalColor enum, convert to QColor first
			from PyQt5.QtGui import QColor

			if hasattr(Qt, "GlobalColor") and isinstance(color, getattr(Qt, "GlobalColor")):
				return QColor(color).name()
			# For other types, try to convert to string
			return str(color)

	def update_axes_options(self) -> None:
		"""Update plot settings from axes options dialog."""
		self.controller.settings.x_title = self.axes_options_dialog.x_title
		self.controller.settings.y_title = self.axes_options_dialog.y_title
		self.controller.settings.title_size = self.axes_options_dialog.title_size
		self.controller.settings.tick_size = self.axes_options_dialog.tick_size
		self.update_plot()

	def update_data(self, df: pd.DataFrame) -> None:
		"""
		Update the view with new data.

		Args:
		    df: New DataFrame for plotting
		"""
		self.controller.update_data(df)

		# Update combo boxes
		for combo in (self.x_combo, self.y_combo):
			combo.clear()
			combo.addItems(df.columns)

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
		try:
			zoom_in = event.button == "up"
			self.controller.zoom_plot(zoom_in)

		except Exception as e:
			self.logger.error(f"Error handling scroll: {e}")

	def _handle_button_press(self, event) -> None:
		"""Handle mouse button press events."""
		try:
			if event.button == 3:  # Right click
				self.controller.reset_zoom()
			elif event.button == 1:  # Left click
				self.controller.store_original_limits()

		except Exception as e:
			self.logger.error(f"Error handling button press: {e}")

	# Dialog openers
	def open_marker_options_dialog(self) -> None:
		self.marker_options_dialog.exec_()

	def open_line_fit_options_dialog(self) -> None:
		self.line_fit_options_dialog.exec_()

	def open_axes_options_dialog(self) -> None:
		self.axes_options_dialog.exec_()
