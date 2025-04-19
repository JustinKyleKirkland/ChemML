from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
	QCheckBox,
	QColorDialog,
	QComboBox,
	QDialog,
	QLabel,
	QLineEdit,
	QPushButton,
	QSlider,
	QVBoxLayout,
	QHBoxLayout,
	QGroupBox,
	QFrame,
	QWidget,
	QFormLayout,
	QDialogButtonBox,
)

from chemml.ui.assets.colors import COLORS


class BaseOptionsDialog(QDialog):
	"""Base class for all plot option dialogs with improved styling."""

	options_applied = pyqtSignal()

	def __init__(self, title, geometry=(300, 100, 400, 300), parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setGeometry(*geometry)
		self.setMinimumWidth(350)

		# Main layout for the dialog
		self.layout = QVBoxLayout()
		self.layout.setSpacing(12)
		self.layout.setContentsMargins(15, 15, 15, 15)

		# Add title label with modern styling
		title_label = QLabel(title)
		title_font = QFont()
		title_font.setPointSize(14)
		title_font.setBold(True)
		title_label.setFont(title_font)
		title_label.setStyleSheet(f"color: {COLORS['primary']};")
		title_label.setAlignment(Qt.AlignCenter)
		self.layout.addWidget(title_label)

		# Add separator
		separator = QFrame()
		separator.setFrameShape(QFrame.HLine)
		separator.setFrameShadow(QFrame.Sunken)
		separator.setStyleSheet(f"background-color: {COLORS['light']}; max-height: 1px;")
		self.layout.addWidget(separator)

		# Content widget (all child classes will add their content here)
		self.content_widget = QWidget()
		self.content_layout = QVBoxLayout(self.content_widget)
		self.content_layout.setSpacing(10)
		self.layout.addWidget(self.content_widget)

		# Button box for dialog actions
		self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		self.button_box.accepted.connect(self._on_accept)
		self.button_box.rejected.connect(self.reject)

		# Style the button box
		self.button_box.button(QDialogButtonBox.Ok).setText("Apply")
		self.button_box.button(QDialogButtonBox.Ok).setStyleSheet(
			f"background-color: {COLORS['primary']}; color: white; "
			f"border-radius: 4px; padding: 6px 16px; font-weight: bold;"
		)
		self.button_box.button(QDialogButtonBox.Cancel).setStyleSheet(
			f"background-color: {COLORS['light']}; color: {COLORS['dark']}; "
			f"border: 1px solid {COLORS['secondary']}; border-radius: 4px; padding: 6px 16px;"
		)

		self.layout.addWidget(self.button_box)
		self.setLayout(self.layout)

	def _add_slider(self, label, min_val, max_val, default, parent_layout=None):
		"""Add a slider with improved visual styling."""
		target_layout = parent_layout or self.content_layout
		group = QGroupBox(label)
		group.setStyleSheet("QGroupBox { font-weight: bold; }")

		layout = QVBoxLayout()

		# Create slider with modern styling
		slider = QSlider(Qt.Horizontal)
		slider.setRange(min_val, max_val)
		slider.setValue(default)
		slider.setMinimumWidth(200)

		# Add value label that updates when slider changes
		value_label = QLabel(str(default))
		value_label.setAlignment(Qt.AlignCenter)

		def update_label(value):
			value_label.setText(str(value))

		slider.valueChanged.connect(update_label)

		# Add elements to layout
		slider_layout = QHBoxLayout()
		slider_layout.addWidget(QLabel(str(min_val)))
		slider_layout.addWidget(slider)
		slider_layout.addWidget(QLabel(str(max_val)))

		layout.addLayout(slider_layout)
		layout.addWidget(value_label)

		group.setLayout(layout)
		target_layout.addWidget(group)
		return slider

	def _add_color_button(self, label, slot, parent_layout=None):
		"""Add a color selection button with color preview."""
		target_layout = parent_layout or self.content_layout

		# Create layout for color selector
		color_layout = QHBoxLayout()

		# Add label
		color_layout.addWidget(QLabel(label))

		# Create color preview (will be updated when color changes)
		self.color_preview = QFrame()
		self.color_preview.setFixedSize(24, 24)
		self.color_preview.setFrameShape(QFrame.Box)
		self.color_preview.setFrameShadow(QFrame.Plain)
		self.color_preview.setStyleSheet("background-color: red; border: 1px solid black;")
		color_layout.addWidget(self.color_preview)

		# Create button
		button = QPushButton("Choose Color")
		button.setStyleSheet(
			f"background-color: {COLORS['light']}; color: {COLORS['dark']}; "
			f"border: 1px solid {COLORS['secondary']}; border-radius: 4px; padding: 4px 8px;"
		)
		button.clicked.connect(slot)
		color_layout.addWidget(button)
		color_layout.addStretch(1)

		target_layout.addLayout(color_layout)
		return button

	def _on_accept(self):
		"""Handle dialog acceptance."""
		self.apply_options()
		self.accept()


class AxesOptionsDialog(BaseOptionsDialog):
	"""Dialog for configuring plot axes with improved UI."""

	def __init__(self, parent=None):
		super().__init__("Axes Configuration", parent=parent)

		# Create form layout for axis title inputs
		titles_group = QGroupBox("Axis Titles")
		titles_group.setStyleSheet("QGroupBox { font-weight: bold; }")
		form_layout = QFormLayout()

		# Input fields for axis titles with better styling
		self.x_title_input = QLineEdit()
		self.x_title_input.setPlaceholderText("Enter X axis title")
		self.x_title_input.setMinimumHeight(28)

		self.y_title_input = QLineEdit()
		self.y_title_input.setPlaceholderText("Enter Y axis title")
		self.y_title_input.setMinimumHeight(28)

		form_layout.addRow("X Axis Title:", self.x_title_input)
		form_layout.addRow("Y Axis Title:", self.y_title_input)
		titles_group.setLayout(form_layout)
		self.content_layout.addWidget(titles_group)

		# Size sliders in their own group
		sizes_group = QGroupBox("Text Sizes")
		sizes_group.setStyleSheet("QGroupBox { font-weight: bold; }")
		sizes_layout = QVBoxLayout()

		# Add sliders to the sizes layout
		self.title_size_slider = self._add_slider("Title Font Size", 8, 24, 12, sizes_layout)
		self.tick_size_slider = self._add_slider("Tick Label Size", 6, 16, 10, sizes_layout)

		sizes_group.setLayout(sizes_layout)
		self.content_layout.addWidget(sizes_group)

		# Default values
		self.x_title = "X Axis"
		self.y_title = "Y Axis"
		self.title_size = 12
		self.tick_size = 10

	def apply_options(self) -> None:
		"""Apply the selected options."""
		self.x_title = self.x_title_input.text() or "X Axis"
		self.y_title = self.y_title_input.text() or "Y Axis"
		self.title_size = self.title_size_slider.value()
		self.tick_size = self.tick_size_slider.value()
		self.options_applied.emit()


class MarkerOptionsDialog(BaseOptionsDialog):
	"""Dialog for configuring plot markers with improved UI."""

	MARKER_SYMBOLS = {"Circle": "o", "Square": "s", "Triangle": "^", "Diamond": "D", "Star": "*", "Plus": "+", "X": "x"}

	def __init__(self, parent=None):
		super().__init__("Marker Configuration", geometry=(300, 100, 400, 350), parent=parent)

		# Marker type selection group
		type_group = QGroupBox("Marker Style")
		type_group.setStyleSheet("QGroupBox { font-weight: bold; }")
		type_layout = QVBoxLayout()

		# Marker type combo box with improved styling
		self.marker_type_combo = QComboBox()
		self.marker_type_combo.addItems(self.MARKER_SYMBOLS.keys())
		self.marker_type_combo.setMinimumHeight(28)

		type_layout.addWidget(self.marker_type_combo)
		type_group.setLayout(type_layout)
		self.content_layout.addWidget(type_group)

		# Color selection with improved display
		color_group = QGroupBox("Marker Color")
		color_group.setStyleSheet("QGroupBox { font-weight: bold; }")
		color_layout = QVBoxLayout()

		# Color selection
		self.marker_color = QColor(COLORS["primary"])
		self.color_preview = QFrame()
		self.color_preview.setFixedSize(40, 24)
		self.color_preview.setFrameShape(QFrame.Box)
		self.color_preview.setStyleSheet(f"background-color: {self.marker_color.name()}; border: 1px solid black;")

		color_button = QPushButton("Choose Color")
		color_button.setMinimumHeight(28)
		color_button.clicked.connect(self.select_marker_color)

		color_button_layout = QHBoxLayout()
		color_button_layout.addWidget(self.color_preview)
		color_button_layout.addWidget(color_button)
		color_layout.addLayout(color_button_layout)

		color_group.setLayout(color_layout)
		self.content_layout.addWidget(color_group)

		# Size slider with visual feedback
		self.marker_size_slider = self._add_slider("Marker Size", 1, 20, 8)

	def select_marker_color(self) -> None:
		"""Open color dialog and update color preview."""
		color = QColorDialog.getColor(self.marker_color)
		if color.isValid():
			self.marker_color = color
			self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

	def get_marker_symbol(self) -> str:
		"""Get the marker symbol based on the current selection."""
		return self.MARKER_SYMBOLS.get(self.marker_type_combo.currentText(), "o")

	def apply_options(self) -> None:
		"""Apply the selected options."""
		self.options_applied.emit()


class LineFitOptionsDialog(BaseOptionsDialog):
	"""Dialog for configuring trend lines with improved UI."""

	def __init__(self, parent=None):
		super().__init__("Trend Line Configuration", geometry=(300, 100, 450, 400), parent=parent)

		# Enable checkboxes group
		options_group = QGroupBox("Options")
		options_group.setStyleSheet("QGroupBox { font-weight: bold; }")
		options_layout = QVBoxLayout()

		# Checkboxes with modern styling
		self.line_fit_checkbox = QCheckBox("Add Line of Best Fit")
		self.line_fit_checkbox.setChecked(True)
		self.line_fit_checkbox.setStyleSheet("QCheckBox { font-size: 13px; }")

		self.r_squared_checkbox = QCheckBox("Show R² value on plot")
		self.r_squared_checkbox.setChecked(True)
		self.r_squared_checkbox.setStyleSheet("QCheckBox { font-size: 13px; }")

		options_layout.addWidget(self.line_fit_checkbox)
		options_layout.addWidget(self.r_squared_checkbox)
		options_group.setLayout(options_layout)
		self.content_layout.addWidget(options_group)

		# Line properties group
		props_group = QGroupBox("Line Properties")
		props_group.setStyleSheet("QGroupBox { font-weight: bold; }")
		props_layout = QVBoxLayout()

		# Line type with improved styling
		line_type_layout = QFormLayout()
		self.line_type_combo = QComboBox()
		self.line_type_combo.addItems(["Solid", "Dashed", "Dotted", "DashDot"])
		self.line_type_combo.setMinimumHeight(28)
		line_type_layout.addRow("Line Type:", self.line_type_combo)
		props_layout.addLayout(line_type_layout)

		# Color selection with preview
		self.line_fit_color = QColor(COLORS["secondary"])
		self.color_preview = QFrame()
		self.color_preview.setFixedSize(40, 24)
		self.color_preview.setFrameShape(QFrame.Box)
		self.color_preview.setStyleSheet(f"background-color: {self.line_fit_color.name()}; border: 1px solid black;")

		color_button = QPushButton("Choose Color")
		color_button.setMinimumHeight(28)
		color_button.clicked.connect(self.select_line_fit_color)

		color_layout = QHBoxLayout()
		color_layout.addWidget(QLabel("Line Color:"))
		color_layout.addWidget(self.color_preview)
		color_layout.addWidget(color_button)
		color_layout.addStretch(1)
		props_layout.addLayout(color_layout)

		# Thickness slider
		self.line_thickness_slider = self._add_slider("Line Thickness", 1, 10, 2, props_layout)

		props_group.setLayout(props_layout)
		self.content_layout.addWidget(props_group)

	def select_line_fit_color(self) -> None:
		"""Open color dialog and update color preview."""
		color = QColorDialog.getColor(self.line_fit_color)
		if color.isValid():
			self.line_fit_color = color
			self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

	def apply_options(self) -> None:
		"""Apply the selected options."""
		self.options_applied.emit()
