import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from utils.plotting_utils import AxesOptionsDialog, BaseOptionsDialog, LineFitOptionsDialog, MarkerOptionsDialog


@pytest.fixture(scope="session")
def qapp():
	"""Create the QApplication instance"""
	# Ensure we don't already have an instance running
	if not QApplication.instance():
		app = QApplication([""])
		app.setStyle("Fusion")  # Use Fusion style for better cross-platform compatibility
		yield app
		app.quit()
	else:
		yield QApplication.instance()


@pytest.fixture(autouse=True)
def setup_test(qapp):
	"""Setup for each test"""
	yield
	# Clean up any remaining widgets after each test
	for widget in QApplication.topLevelWidgets():
		widget.deleteLater()


class TestBaseOptionsDialog:
	def test_initialization(self, qapp):
		dialog = BaseOptionsDialog("Test Dialog")
		try:
			assert dialog.windowTitle() == "Test Dialog"
			assert dialog.layout is not None
		finally:
			dialog.deleteLater()

	def test_add_slider(self, qapp):
		dialog = BaseOptionsDialog("Test")
		try:
			slider = dialog._add_slider("Test Slider", 0, 10, 5)
			assert slider.minimum() == 0
			assert slider.maximum() == 10
			assert slider.value() == 5
		finally:
			dialog.deleteLater()

	def test_add_color_button(self, qapp):
		dialog = BaseOptionsDialog("Test")
		try:

			def dummy_slot():
				pass

			button = dialog._add_color_button("Color Button", dummy_slot)
			assert button.text() == "Color Button"
		finally:
			dialog.deleteLater()


class TestAxesOptionsDialog:
	def test_initialization(self, qapp):
		dialog = AxesOptionsDialog()
		try:
			assert dialog.windowTitle() == "Axes Options"
			assert dialog.x_title == "X Axis"
			assert dialog.y_title == "Y Axis"
			assert dialog.title_size == 12
			assert dialog.tick_size == 10
		finally:
			dialog.deleteLater()

	def test_apply_options(self, qapp):
		dialog = AxesOptionsDialog()
		try:
			# Test with empty inputs
			dialog.x_title_input.setText("")
			dialog.y_title_input.setText("")
			dialog.title_size_slider.setValue(20)
			dialog.tick_size_slider.setValue(15)

			# Create a flag to check if signal was emitted
			signal_emitted = False

			def slot():
				nonlocal signal_emitted
				signal_emitted = True

			dialog.options_applied.connect(slot)
			dialog.apply_options()

			assert dialog.x_title == "X Axis"  # Default value
			assert dialog.y_title == "Y Axis"  # Default value
			assert dialog.title_size == 20
			assert dialog.tick_size == 15
			assert signal_emitted
		finally:
			dialog.deleteLater()

	def test_custom_titles(self, qapp):
		dialog = AxesOptionsDialog()
		try:
			dialog.x_title_input.setText("Custom X")
			dialog.y_title_input.setText("Custom Y")
			dialog.apply_options()
			assert dialog.x_title == "Custom X"
			assert dialog.y_title == "Custom Y"
		finally:
			dialog.deleteLater()


class TestMarkerOptionsDialog:
	def test_initialization(self, qapp):
		dialog = MarkerOptionsDialog()
		try:
			assert dialog.windowTitle() == "Marker Options"
			assert dialog.marker_color == Qt.red
			assert dialog.marker_size_slider.value() == 5
		finally:
			dialog.deleteLater()

	def test_marker_symbols(self, qapp):
		dialog = MarkerOptionsDialog()
		try:
			for marker_type, symbol in dialog.MARKER_SYMBOLS.items():
				dialog.marker_type_combo.setCurrentText(marker_type)
				assert dialog.get_marker_symbol() == symbol
		finally:
			dialog.deleteLater()

	def test_invalid_marker_symbol(self, qapp):
		dialog = MarkerOptionsDialog()
		try:
			dialog.marker_type_combo.setCurrentText("Invalid")
			assert dialog.get_marker_symbol() == "o"  # Default symbol
		finally:
			dialog.deleteLater()


class TestLineFitOptionsDialog:
	def test_initialization(self, qapp):
		dialog = LineFitOptionsDialog()
		try:
			assert dialog.windowTitle() == "Line of Best Fit Options"
			assert dialog.line_fit_color == Qt.black
			assert not dialog.line_fit_checkbox.isChecked()
			assert not dialog.r_squared_checkbox.isChecked()
		finally:
			dialog.deleteLater()

	def test_line_properties(self, qapp):
		dialog = LineFitOptionsDialog()
		try:
			# Test line thickness
			assert dialog.line_thickness_slider.value() == 2
			dialog.line_thickness_slider.setValue(5)
			assert dialog.line_thickness_slider.value() == 5

			# Test line type combo
			line_types = ["Solid", "Dashed", "Dotted", "DashDot"]
			for line_type in line_types:
				dialog.line_type_combo.setCurrentText(line_type)
				assert dialog.line_type_combo.currentText() == line_type
		finally:
			dialog.deleteLater()

	def test_checkboxes(self, qapp):
		dialog = LineFitOptionsDialog()
		try:
			dialog.line_fit_checkbox.setChecked(True)
			assert dialog.line_fit_checkbox.isChecked()

			dialog.r_squared_checkbox.setChecked(True)
			assert dialog.r_squared_checkbox.isChecked()
		finally:
			dialog.deleteLater()
