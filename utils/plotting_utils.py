from PyQt5.QtCore import Qt, pyqtSignal
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
)


class BaseOptionsDialog(QDialog):
	options_applied = pyqtSignal()

	def __init__(self, title, geometry=(100, 100, 300, 250), parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setGeometry(*geometry)
		self.layout = QVBoxLayout()
		self.setLayout(self.layout)

	def _add_slider(self, label, min_val, max_val, default):
		slider = QSlider(Qt.Horizontal)
		slider.setRange(min_val, max_val)
		slider.setValue(default)
		self.layout.addWidget(QLabel(label))
		self.layout.addWidget(slider)
		return slider

	def _add_color_button(self, label, slot):
		button = QPushButton(label)
		button.clicked.connect(slot)
		self.layout.addWidget(button)
		return button

	def _add_apply_button(self):
		apply_button = QPushButton("Apply")
		apply_button.clicked.connect(self.apply_options)
		self.layout.addWidget(apply_button)


class AxesOptionsDialog(BaseOptionsDialog):
	def __init__(self, parent=None):
		super().__init__("Axes Options", parent=parent)

		# Input fields for axis titles
		self.x_title_input = QLineEdit()
		self.y_title_input = QLineEdit()

		self.layout.addWidget(QLabel("X Axis Title:"))
		self.layout.addWidget(self.x_title_input)
		self.layout.addWidget(QLabel("Y Axis Title:"))
		self.layout.addWidget(self.y_title_input)

		# Sliders
		self.title_size_slider = self._add_slider("Title Size:", 8, 40, 12)
		self.tick_size_slider = self._add_slider("Tick Size:", 8, 20, 10)

		self._add_apply_button()

		# Default values
		self.x_title = "X Axis"
		self.y_title = "Y Axis"
		self.title_size = 12
		self.tick_size = 10

	def apply_options(self) -> None:
		self.x_title = self.x_title_input.text() or "X Axis"
		self.y_title = self.y_title_input.text() or "Y Axis"
		self.title_size = self.title_size_slider.value()
		self.tick_size = self.tick_size_slider.value()
		self.options_applied.emit()


class MarkerOptionsDialog(BaseOptionsDialog):
	MARKER_SYMBOLS = {"Circle": "o", "Square": "s", "Triangle": "^", "Diamond": "D"}

	def __init__(self, parent=None):
		super().__init__("Marker Options", geometry=(100, 100, 200, 200), parent=parent)

		# Marker type combo box
		self.marker_type_combo = QComboBox()
		self.marker_type_combo.addItems(self.MARKER_SYMBOLS.keys())
		self.layout.addWidget(QLabel("Marker Type:"))
		self.layout.addWidget(self.marker_type_combo)

		# Color selection
		self.marker_color = Qt.red
		self._add_color_button("Select Marker Color", self.select_marker_color)

		# Size slider
		self.marker_size_slider = self._add_slider("Marker Size:", 1, 20, 5)

		self._add_apply_button()

	def select_marker_color(self) -> None:
		color = QColorDialog.getColor()
		if color.isValid():
			self.marker_color = color

	def get_marker_symbol(self) -> str:
		return self.MARKER_SYMBOLS.get(self.marker_type_combo.currentText(), "o")

	def apply_options(self) -> None:
		self.options_applied.emit()


class LineFitOptionsDialog(BaseOptionsDialog):
	def __init__(self, parent=None):
		super().__init__("Line of Best Fit Options", geometry=(100, 100, 200, 200), parent=parent)

		# Checkboxes
		self.line_fit_checkbox = QCheckBox("Add Line of Best Fit")
		self.r_squared_checkbox = QCheckBox("Calculate R²")
		self.layout.addWidget(self.line_fit_checkbox)
		self.layout.addWidget(self.r_squared_checkbox)

		# Color selection
		self.line_fit_color = Qt.black
		self._add_color_button("Select Line of Best Fit Color", self.select_line_fit_color)

		# Line properties
		self.line_thickness_slider = self._add_slider("Line Thickness:", 1, 10, 2)

		self.line_type_combo = QComboBox()
		self.line_type_combo.addItems(["Solid", "Dashed", "Dotted", "DashDot"])
		self.layout.addWidget(QLabel("Line Type:"))
		self.layout.addWidget(self.line_type_combo)

		self._add_apply_button()

	def select_line_fit_color(self) -> None:
		color = QColorDialog.getColor()
		if color.isValid():
			self.line_fit_color = color

	def apply_options(self) -> None:
		self.options_applied.emit()
