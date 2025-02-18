from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
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


class AxesOptionsDialog(QDialog):
	options_applied = pyqtSignal()

	def __init__(self, parent=None):
		super().__init__(parent)

		self.setWindowTitle("Axes Options")
		self.setGeometry(100, 100, 300, 250)

		layout = QVBoxLayout()

		self.x_title_input = QLineEdit()
		self.y_title_input = QLineEdit()

		layout.addWidget(QLabel("X Axis Title:"))
		layout.addWidget(self.x_title_input)
		layout.addWidget(QLabel("Y Axis Title:"))
		layout.addWidget(self.y_title_input)

		self.title_size_slider = QSlider(Qt.Horizontal)
		self.title_size_slider.setRange(8, 40)
		self.title_size_slider.setValue(12)
		layout.addWidget(QLabel("Title Size:"))
		layout.addWidget(self.title_size_slider)

		self.tick_size_slider = QSlider(Qt.Horizontal)
		self.tick_size_slider.setRange(8, 20)
		self.tick_size_slider.setValue(10)
		layout.addWidget(QLabel("Tick Size:"))
		layout.addWidget(self.tick_size_slider)

		apply_button = QPushButton("Apply")
		apply_button.clicked.connect(self.apply_options)
		layout.addWidget(apply_button)

		self.setLayout(layout)

		self.x_title = "X Axis"
		self.y_title = "Y Axis"
		self.title_size = 12
		self.tick_size = 10

	def apply_options(self) -> None:
		self.x_title = self.x_title_input.text() if self.x_title_input.text() else "X Axis"
		self.y_title = self.y_title_input.text() if self.y_title_input.text() else "Y Axis"
		self.title_size = self.title_size_slider.value()
		self.tick_size = self.tick_size_slider.value()
		self.options_applied.emit()


class MarkerOptionsDialog(QDialog):
	options_applied = pyqtSignal()

	def __init__(self, parent=None):
		super().__init__(parent)

		self.setWindowTitle("Marker Options")
		self.setGeometry(100, 100, 200, 200)

		layout = QVBoxLayout()

		self.marker_type_combo = QComboBox()
		self.marker_type_combo.addItems(["Circle", "Square", "Triangle", "Diamond"])
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

		apply_button = QPushButton("Apply")
		apply_button.clicked.connect(self.apply_options)
		layout.addWidget(apply_button)

		self.setLayout(layout)

	def select_marker_color(self) -> None:
		color: QColor = QColorDialog.getColor()
		if color.isValid():
			self.marker_color = color

	def get_marker_symbol(self) -> str:
		symbol_mapping = {"Circle": "o", "Square": "s", "Triangle": "^", "Diamond": "D"}
		return symbol_mapping.get(self.marker_type_combo.currentText(), "o")

	def apply_options(self) -> None:
		self.options_applied.emit()


class LineFitOptionsDialog(QDialog):
	options_applied = pyqtSignal()

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

		apply_button = QPushButton("Apply")
		apply_button.clicked.connect(self.apply_options)
		layout.addWidget(apply_button)

		self.setLayout(layout)

	def select_line_fit_color(self) -> None:
		color: QColor = QColorDialog.getColor()
		if color.isValid():
			self.line_fit_color = color

	def apply_options(self) -> None:
		self.options_applied.emit()
