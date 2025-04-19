import pandas as pd
import pytest
from PyQt5.QtWidgets import QMessageBox, QApplication

from gui.ml_view import (
	AVAILABLE_ML_METHODS,  # Import the constant from the module
	FeatureSelectionWidget,
	MethodSelectionWidget,
	MLView,
)


@pytest.fixture
def sample_dataframe():
	return pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10], "target": [3, 6, 9, 12, 15]})


@pytest.fixture
def method_selection(qtbot):
	widget = MethodSelectionWidget()
	qtbot.addWidget(widget)
	return widget


@pytest.fixture
def feature_selection(qtbot):
	widget = FeatureSelectionWidget()
	qtbot.addWidget(widget)
	return widget


@pytest.fixture
def ml_view(qtbot):
	widget = MLView()
	qtbot.addWidget(widget)
	return widget


@pytest.fixture
def app():
	"""Create a Qt application for testing."""
	return QApplication([])


class TestMethodSelectionWidget:
	def test_initialization(self, method_selection):
		"""Test initial state of the widget."""
		assert method_selection.available_list.count() == len(AVAILABLE_ML_METHODS)
		assert method_selection.selected_list.count() == 0

	def test_move_to_selected(self, method_selection):
		"""Test moving methods to selected list."""
		# Select first item
		method_selection.available_list.item(0).setSelected(True)
		initial_count = method_selection.available_list.count()

		method_selection._move_to_selected()

		assert method_selection.available_list.count() == initial_count - 1
		assert method_selection.selected_list.count() == 1

	def test_move_to_available(self, method_selection):
		"""Test moving methods back to available list."""
		# First move an item to selected
		method_selection.available_list.item(0).setSelected(True)
		method_selection._move_to_selected()

		# Then move it back
		method_selection.selected_list.item(0).setSelected(True)
		initial_selected_count = method_selection.selected_list.count()

		method_selection._move_to_available()

		assert method_selection.selected_list.count() == initial_selected_count - 1
		assert method_selection.available_list.count() == len(AVAILABLE_ML_METHODS)

	def test_get_selected_methods(self, method_selection):
		"""Test getting list of selected methods."""
		# Move two methods to selected
		method_selection.available_list.item(0).setSelected(True)
		method_selection.available_list.item(1).setSelected(True)
		method_selection._move_to_selected()

		selected = method_selection.get_selected_methods()
		assert len(selected) == 2
		assert all(isinstance(method, str) for method in selected)


class TestFeatureSelectionWidget:
	def test_initialization(self, feature_selection):
		"""Test initial state of the widget."""
		assert feature_selection.target_combo.currentText() == "Select Target Column"
		assert feature_selection.available_list.count() == 0
		assert feature_selection.selected_list.count() == 0

	def test_update_columns(self, feature_selection, sample_dataframe):
		"""Test updating columns from dataframe."""
		feature_selection.update_columns(sample_dataframe)

		assert (
			feature_selection.target_combo.count() == len(sample_dataframe.columns) + 1
		)  # +1 for "Select Target Column"
		assert feature_selection.available_list.count() == len(sample_dataframe.columns)

	def test_move_features(self, feature_selection, sample_dataframe):
		"""Test moving features between lists."""
		feature_selection.update_columns(sample_dataframe)

		# Select and move a feature
		feature_selection.available_list.item(0).setSelected(True)
		feature_selection._move_to_selected()

		assert feature_selection.selected_list.count() == 1
		assert feature_selection.available_list.count() == len(sample_dataframe.columns) - 1

	def test_get_target_column(self, feature_selection, sample_dataframe):
		"""Test getting selected target column."""
		feature_selection.update_columns(sample_dataframe)
		feature_selection.target_combo.setCurrentText("target")

		assert feature_selection.get_target_column() == "target"

	def test_get_feature_columns(self, feature_selection, sample_dataframe):
		"""Test getting selected feature columns."""
		feature_selection.update_columns(sample_dataframe)

		# Move two features to selected
		feature_selection.available_list.item(0).setSelected(True)
		feature_selection._move_to_selected()

		features = feature_selection.get_feature_columns()
		assert len(features) == 1
		assert isinstance(features[0], str)


class TestMLView:
	def test_initialization(self, ml_view):
		"""Test initial state of the view."""
		assert ml_view.df.empty
		assert isinstance(ml_view.method_selection, MethodSelectionWidget)
		assert isinstance(ml_view.feature_selection, FeatureSelectionWidget)

	def test_set_dataframe(self, ml_view, sample_dataframe):
		"""Test setting dataframe updates all components."""
		ml_view.set_dataframe(sample_dataframe)

		assert ml_view.df.equals(sample_dataframe)
		assert ml_view.feature_selection.target_combo.count() == len(sample_dataframe.columns) + 1
		assert ml_view.feature_selection.available_list.count() == len(sample_dataframe.columns)

	def test_handle_target_change(self, ml_view, sample_dataframe):
		"""Test handling target column changes."""
		ml_view.set_dataframe(sample_dataframe)
		ml_view.feature_selection.target_combo.setCurrentText("target")

		# Target should be removed from available features
		available_features = [
			ml_view.feature_selection.available_list.item(i).text()
			for i in range(ml_view.feature_selection.available_list.count())
		]
		assert "target" not in available_features

	def test_validate_selections_empty(self, ml_view, qtbot, monkeypatch):
		"""Test validation with no selections."""
		# Mock QMessageBox to prevent popups
		warning_shown = False

		def mock_warning(*args, **kwargs):
			nonlocal warning_shown
			warning_shown = True
			return QMessageBox.Ok

		monkeypatch.setattr(QMessageBox, "warning", mock_warning)

		assert not ml_view._validate_selections()
		assert warning_shown  # Verify the warning would have been shown

	def test_validate_selections_complete(self, ml_view, sample_dataframe, qtbot):
		"""Test validation with all selections made."""
		ml_view.set_dataframe(sample_dataframe)

		# Set target
		ml_view.feature_selection.target_combo.setCurrentText("target")

		# Select features
		ml_view.feature_selection.available_list.item(0).setSelected(True)
		ml_view.feature_selection._move_to_selected()

		# Select method
		ml_view.method_selection.available_list.item(0).setSelected(True)
		ml_view.method_selection._move_to_selected()

		assert ml_view._validate_selections()

	@pytest.mark.parametrize("button_name", ["run_button", "plot_button"])
	def test_action_buttons_exist(self, ml_view, button_name):
		"""Test that action buttons exist and are properly configured."""
		button = getattr(ml_view, button_name)
		assert button is not None

		style = button.styleSheet()
		assert "background-color:#007bff" in "".join(style.split())
		assert "color:white" in "".join(style.split())
		assert "font-weight:bold" in "".join(style.split())

		assert button.height() == 40

	def test_show_warning(self, ml_view, monkeypatch, qtbot):
		"""Test warning message display."""
		# Mock QMessageBox.warning
		warning_shown = False

		def mock_warning(*args, **kwargs):
			nonlocal warning_shown
			warning_shown = True
			return QMessageBox.Ok

		monkeypatch.setattr(QMessageBox, "warning", mock_warning)

		ml_view._show_warning("Test warning")
		assert warning_shown


def test_ml_view_initialization(app):
	"""Test ML view initializes correctly."""
	view = MLView()
	assert view is not None
	assert hasattr(view, "controller")
	assert hasattr(view, "method_selection")
	assert hasattr(view, "feature_selection")
	assert hasattr(view, "run_button")
	assert hasattr(view, "plot_button")


def test_method_selection_widget_initialization(app):
	"""Test method selection widget initializes correctly."""
	widget = MethodSelectionWidget()
	assert widget is not None
	assert hasattr(widget, "available_list")
	assert hasattr(widget, "selected_list")
	assert widget.available_list.count() > 0  # Should have some available methods


def test_feature_selection_widget_initialization(app):
	"""Test feature selection widget initializes correctly."""
	widget = FeatureSelectionWidget()
	assert widget is not None
	assert hasattr(widget, "target_combo")
	assert hasattr(widget, "available_list")
	assert hasattr(widget, "selected_list")
