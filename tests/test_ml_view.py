import pytest
from PyQt5.QtWidgets import QApplication

from gui.ml_view import CustomParamsDialog, MethodSelectionWidget
from ml_backend.model_configs import MODEL_CONFIGS


@pytest.fixture
def app():
	"""Create a Qt Application"""
	return QApplication([])


@pytest.fixture
def method_selection(app, request):
	"""Create a MethodSelectionWidget instance"""
	widget = MethodSelectionWidget()

	def cleanup():
		widget.hide()
		widget.deleteLater()
		app.processEvents()

	request.addfinalizer(cleanup)
	return widget


@pytest.mark.qt
def test_custom_params_dialog(app, request):
	"""Test the CustomParamsDialog functionality"""
	test_params = {"max_depth": [10, 20], "min_samples_split": [2, 5]}
	dialog = CustomParamsDialog("Test Method", test_params)

	def cleanup():
		dialog.hide()
		dialog.deleteLater()
		app.processEvents()

	request.addfinalizer(cleanup)

	# Check initial state
	assert dialog.model_name == "Test Method"
	assert dialog.current_params == test_params
	assert str(test_params) in dialog.param_input.toPlainText()

	# Test parameter parsing
	dialog.param_input.setText("{'test': [1, 2, 3]}")
	assert dialog.get_params() == {"test": [1, 2, 3]}

	# Test invalid input handling
	dialog.param_input.setText("invalid json")
	assert dialog.get_params() == test_params  # Should return original params


@pytest.mark.qt
def test_method_selection_custom_params(method_selection):
	"""Test custom parameter handling in MethodSelectionWidget"""
	# Add a method to selected list
	method_selection.available_list.addItem("Random Forest Regression")
	method_selection._move_to_selected()

	# Verify no custom params initially
	assert len(method_selection.custom_params) == 0

	# Select the method
	method_selection.selected_list.item(0).setSelected(True)

	# Get default params
	default_params = MODEL_CONFIGS["Random Forest Regression"].param_grid
	assert "max_depth" in default_params

	# Add custom params
	custom_params = {"max_depth": [5, 10], "min_samples_split": [3, 6]}
	method_selection.custom_params["Random Forest Regression"] = custom_params

	# Verify custom params are stored
	assert method_selection.get_custom_params()["Random Forest Regression"] == custom_params
