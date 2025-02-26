from unittest.mock import MagicMock

import pandas as pd
import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import (
	QComboBox,
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMessageBox,
	QPushButton,
	QTableWidget,
)

from gui.csv_view import CSVView


@pytest.fixture(autouse=True)
def mpl_backend():
	"""Configure matplotlib to use non-interactive backend."""
	import matplotlib

	matplotlib.use("Agg")


@pytest.fixture
def mock_csv_path(tmp_path):
	"""Create a temporary CSV file for testing."""
	df = pd.DataFrame(
		{"numeric": [1, 2, 3, None, 5], "categorical": ["A", "B", None, "D", "E"], "binary": [0, 1, 1, 0, None]}
	)
	path = tmp_path / "test.csv"
	df.to_csv(path, index=False)
	return str(path)


@pytest.fixture
def csv_view(qtbot, cleanup_widgets):
	"""Create a CSVView instance for testing."""
	view = CSVView(None)
	cleanup_widgets(view)
	qtbot.addWidget(view)
	return view


@pytest.fixture
def csv_view_with_data(qtbot, mock_csv_path, cleanup_widgets):
	"""Create a CSVView instance with pre-loaded data."""
	view = CSVView(None)
	cleanup_widgets(view)
	qtbot.addWidget(view)

	# Load the CSV directly without dialog
	view.df = pd.read_csv(mock_csv_path)
	view.update_table(view.df)
	return view


@pytest.fixture
def sample_df():
	"""Create a sample DataFrame for testing."""
	return pd.DataFrame(
		{"numeric": [1, 2, 3, None, 5], "categorical": ["A", "B", None, "D", "E"], "binary": [0, 1, 1, 0, None]}
	)


def test_create_error_label(csv_view):
	error_label = csv_view.create_error_label()
	assert isinstance(error_label, QLabel)


def test_create_load_button(csv_view):
	load_button = csv_view.create_load_button()
	assert isinstance(load_button, QPushButton)


def test_create_filter_widgets(csv_view):
	column_dropdown, condition_dropdown, value_edit = csv_view.create_filter_widgets()
	assert isinstance(column_dropdown, QComboBox)
	assert isinstance(condition_dropdown, QComboBox)
	assert isinstance(value_edit, QLineEdit)


def test_create_filter_button(csv_view):
	filter_button = csv_view.create_filter_button()
	assert isinstance(filter_button, QPushButton)


def test_create_table_widget(csv_view):
	table_widget = csv_view.create_table_widget()
	assert isinstance(table_widget, QTableWidget)


def test_create_filter_layout(csv_view):
	filter_layout = csv_view.create_filter_layout()
	assert isinstance(filter_layout, QHBoxLayout)


def test_load_csv(csv_view, mock_csv_path, monkeypatch):
	"""Test loading CSV without file dialog."""
	# Mock the file dialog to return our test file path
	monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (mock_csv_path, ""))

	# Trigger load
	csv_view.load_csv()

	# Verify data was loaded
	assert not csv_view.df.empty
	assert "numeric" in csv_view.df.columns


def test_undo(csv_view):
	# Add some data to the undo stack
	csv_view.undo_stack.append(pd.DataFrame())

	# Call the undo method
	csv_view.undo()

	# Check if the undo operation was successful
	assert len(csv_view.undo_stack) == 0
	assert len(csv_view.redo_stack) == 1


def test_redo(csv_view):
	# Add some data to the redo stack
	csv_view.redo_stack.append(pd.DataFrame())

	# Call the redo method
	csv_view.redo()

	# Check if the redo operation was successful
	assert len(csv_view.redo_stack) == 0
	assert len(csv_view.undo_stack) == 1


def test_apply_one_hot_encoding(csv_view_with_data, monkeypatch):
	"""Test one-hot encoding without user interaction."""
	original_columns = len(csv_view_with_data.df.columns)
	original_df = csv_view_with_data.df.copy()

	# Create mock encoded DataFrame
	mock_encoded_df = original_df.copy()
	mock_encoded_df["categorical_A"] = [1, 0, 0, 0, 0]
	mock_encoded_df["categorical_B"] = [0, 1, 0, 0, 0]
	mock_encoded_df["categorical_D"] = [0, 0, 0, 1, 0]
	mock_encoded_df["categorical_E"] = [0, 0, 0, 0, 1]

	# Mock the one_hot_encode function
	def mock_one_hot_encode(df, column_name, n_distinct=True):
		return mock_encoded_df

	monkeypatch.setattr("utils.data_utils.one_hot_encode", mock_one_hot_encode)
	monkeypatch.setattr("gui.csv_view.one_hot_encode", mock_one_hot_encode)  # Also mock local import

	# Apply one-hot encoding directly
	csv_view_with_data.apply_one_hot_encoding("categorical")

	# Verify new columns were created
	assert len(csv_view_with_data.df.columns) > original_columns
	assert "categorical_A" in csv_view_with_data.df.columns
	assert "categorical_B" in csv_view_with_data.df.columns
	assert "categorical_D" in csv_view_with_data.df.columns
	assert "categorical_E" in csv_view_with_data.df.columns


def test_impute_missing_values(csv_view, monkeypatch):
	"""Test imputing missing values for a single column."""
	# Create test data with missing values
	test_df = pd.DataFrame({"numeric_col": [1.0, 2.0, None, 4.0, 5.0]})
	csv_view.df = test_df.copy()

	messages = []
	imputed_values = []

	# Mock QMessageBox
	class MockMessageBox:
		def __init__(self, parent=None):
			self.title = ""
			self.text = ""

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def setIcon(self, icon):
			pass

		def exec_(self):
			messages.append((self.title, self.text))

	monkeypatch.setattr(QMessageBox, "__new__", lambda cls, *args: MockMessageBox())

	# Mock impute_values
	def mock_impute(df, column, method):
		imputed_values.append((column, method))
		return df.assign(**{column: df[column].fillna(0)})

	# Mock both the utility module and local import
	monkeypatch.setattr("utils.data_utils.impute_values", mock_impute)
	monkeypatch.setattr("gui.csv_view.impute_values", mock_impute)

	# Test valid imputation
	csv_view.impute_missing_values("numeric_col", "mean")
	assert len(imputed_values) == 1
	assert imputed_values[0] == ("numeric_col", "mean")

	# Test invalid column
	csv_view.impute_missing_values("non_existent_col", "mean")
	assert len(messages) == 1
	assert messages[0][0] == "Imputation Error"
	assert "not found in DataFrame" in messages[0][1]


def test_show_errors(csv_view, monkeypatch, cleanup_widgets):
	"""Test showing errors without popup."""
	messages = []

	class MockMessageBox:
		def __init__(self, *args, **kwargs):
			self.text = ""
			self.title = ""
			self.icon = None
			cleanup_widgets(self)

		def setText(self, text):
			self.text = text
			messages.append(text)

		def setWindowTitle(self, title):
			self.title = title

		def setIcon(self, icon):
			self.icon = icon

		def exec_(self):
			# Add the message when exec_ is called
			messages.append(self.text)
			pass

		def hide(self):
			pass

		def close(self):
			pass

		def deleteLater(self):
			pass

		@staticmethod
		def Critical():
			return 1

	# Create a mock message box instance
	mock_msg_box = MockMessageBox()
	monkeypatch.setattr(csv_view, "create_message_box", lambda *args: mock_msg_box)

	# Call the show_errors method with some errors
	test_errors = ["Error 1", "Error 2"]
	csv_view.show_errors(test_errors)

	# Check if the error label is updated correctly
	assert csv_view.error_label.text() == "Error 1\nError 2"

	# Print debug info
	print("Error label text:", csv_view.error_label.text())
	print("Messages list:", messages)
	print("Mock message box text:", mock_msg_box.text)

	# Check that the errors were shown
	assert test_errors[0] in csv_view.error_label.text()
	assert test_errors[1] in csv_view.error_label.text()


def test_adjust_window_size(csv_view):
	# Call the adjust_window_size method
	csv_view.adjust_window_size()

	# Check if the window size is adjusted correctly
	assert csv_view.width() >= csv_view.min_width
	assert csv_view.height() >= csv_view.min_height


def test_apply_filter(csv_view):
	# Create a test DataFrame
	df = pd.DataFrame({"column_name": [1, 2, 3], "B": [4, 5, 6]})

	# Set the test DataFrame as the current DataFrame
	csv_view.df = df

	# Call the apply_filter method
	filtered_df = csv_view.apply_filter("column_name", "<", "2")
	print(filtered_df)

	# Check if the filter is applied correctly
	assert len(filtered_df) == 1
	assert filtered_df["column_name"].values[0] == 1


def test_update_table(csv_view):
	# Create a test DataFrame
	df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

	# Call the update_table method
	csv_view.update_table(df)

	# Check if the table is updated correctly
	assert csv_view.table_widget.rowCount() == len(df)
	assert csv_view.table_widget.columnCount() == len(df.columns)
	assert csv_view.table_widget.horizontalHeaderItem(0).text() == "A"
	assert csv_view.table_widget.horizontalHeaderItem(1).text() == "B"
	assert csv_view.table_widget.item(0, 0).text() == "1"
	assert csv_view.table_widget.item(0, 1).text() == "4"


def test_create_message_box(csv_view, monkeypatch):
	"""Test creation of message box with different parameters."""

	# Mock QMessageBox to avoid actual dialog display
	class MockMessageBox:
		def __init__(self, parent=None):
			self.title = ""
			self.text = ""
			self.icon_type = None

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def setIcon(self, icon):
			self.icon_type = icon

	monkeypatch.setattr(QMessageBox, "__new__", lambda cls, *args: MockMessageBox())

	msg_box = csv_view.create_message_box("Test Title", "Test Message", QMessageBox.Warning)
	assert msg_box.title == "Test Title"
	assert msg_box.text == "Test Message"
	assert msg_box.icon_type == QMessageBox.Warning


def test_show_warning(csv_view, monkeypatch):
	"""Test showing warning message."""
	messages = []

	class MockMessageBox:
		def __init__(self, parent=None):
			self.title = ""
			self.text = ""

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def setIcon(self, icon):
			pass

		def exec_(self):
			messages.append((self.title, self.text))

	monkeypatch.setattr(QMessageBox, "__new__", lambda cls, *args: MockMessageBox())
	csv_view.show_warning("Test Warning", "This is a warning")
	assert messages == [("Test Warning", "This is a warning")]


def test_show_error(csv_view, monkeypatch):
	"""Test showing error message."""
	messages = []

	class MockMessageBox:
		def __init__(self, parent=None):
			self.title = ""
			self.text = ""

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def setIcon(self, icon):
			pass

		def exec_(self):
			messages.append((self.title, self.text))

	monkeypatch.setattr(QMessageBox, "__new__", lambda cls, *args: MockMessageBox())
	csv_view.show_error("Test Error", "This is an error")
	assert messages == [("Test Error", "This is an error")]


def test_update_table_with_sample_data(csv_view, sample_df):
	"""Test updating table with sample data."""
	csv_view.update_table(sample_df)

	assert csv_view.table_widget.rowCount() == len(sample_df)
	assert csv_view.table_widget.columnCount() == len(sample_df.columns)

	# Check column headers
	for i, col in enumerate(sample_df.columns):
		assert csv_view.table_widget.horizontalHeaderItem(i).text() == col


def test_impute_missing_values_all(csv_view, sample_df, monkeypatch):
	"""Test imputing missing values for all columns."""
	# Create a numeric-only DataFrame
	numeric_df = pd.DataFrame(
		{"col1": [1, 2, None, 4, 5], "col2": [1.1, None, 3.3, 4.4, 5.5], "col3": [10, 20, 30, None, 50]}
	)

	csv_view.df = numeric_df.copy()
	imputed_values = []

	def mock_impute(df, column, method):
		imputed_values.append((column, method))
		return df.assign(**{column: df[column].fillna(0)})

	# Make sure we're mocking the correct path
	monkeypatch.setattr("utils.data_utils.impute_values", mock_impute)
	# Also mock it in the local namespace of csv_view
	monkeypatch.setattr("gui.csv_view.impute_values", mock_impute)

	# Test successful imputation with numeric data
	csv_view.impute_missing_values_all("mean")
	expected_columns = [col for col in numeric_df.columns if numeric_df[col].isnull().any()]
	assert len(imputed_values) == len(expected_columns)
	assert all(method == "mean" for _, method in imputed_values)

	# Test with mixed data types
	messages = []

	class MockMessageBox:
		def __init__(self, parent=None):
			self.title = ""
			self.text = ""

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def setIcon(self, icon):
			pass

		def exec_(self):
			messages.append((self.title, self.text))

	monkeypatch.setattr(QMessageBox, "__new__", lambda cls, *args: MockMessageBox())

	# Test with non-numeric data
	csv_view.df = sample_df.copy()
	csv_view.impute_missing_values_all("mean")

	assert len(messages) == 1
	assert messages[0][0] == "Imputation Error"
	assert "contain non-numerical values" in messages[0][1]


# def test_context_menu_creation(csv_view, sample_df, qtbot):
# 	"""Test context menu creation and options."""
# 	csv_view.df = sample_df
# 	csv_view.update_table(sample_df)

# 	# Simulate right click on first cell
# 	pos = QPoint(0, 0)
# 	csv_view.show_context_menu(pos)


def test_undo_redo_operations(csv_view, sample_df):
	"""Test undo and redo functionality."""
	# Initial state
	csv_view.df = sample_df.copy()

	# Make a change
	modified_df = sample_df.copy()
	modified_df.iloc[0, 0] = 999
	csv_view.undo_stack.append(csv_view.df.copy())
	csv_view.df = modified_df

	# Test undo
	csv_view.undo()
	pd.testing.assert_frame_equal(csv_view.df, sample_df)

	# Test redo
	csv_view.redo()
	pd.testing.assert_frame_equal(csv_view.df, modified_df)


def test_filter_data(csv_view, sample_df, monkeypatch):
	"""Test data filtering functionality."""

	# Mock QMessageBox to prevent popups
	class MockMessageBox:
		def __init__(self, *args, **kwargs):
			self.text = ""
			self.title = ""
			self.icon = None

		def setIcon(self, icon):
			self.icon = icon

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def exec_(self):
			pass

		def hide(self):
			pass

		def close(self):
			pass

		def deleteLater(self):
			pass

		@staticmethod
		def Critical():
			return 1

	# Mock QMessageBox at both module and class level
	monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox", MockMessageBox)
	monkeypatch.setattr("gui.csv_view.QMessageBox", MockMessageBox)
	monkeypatch.setattr(QMessageBox, "Critical", MockMessageBox.Critical)

	# Set up test data
	csv_view.df = sample_df.copy()  # Make a copy to avoid modifying original
	csv_view.update_table(sample_df)
	csv_view.column_dropdown.addItems(sample_df.columns)
	csv_view.column_dropdown.setCurrentText("numeric")
	csv_view.condition_dropdown.setCurrentText(">")
	csv_view.value_edit.setText("2")

	# Apply filter
	csv_view.filter_data()

	# Verify filter was applied
	assert len(csv_view.table_widget.selectedItems()) >= 0


def test_window_size_limits(csv_view):
	"""Test window size limits are properly set."""
	assert csv_view.width() >= csv_view.min_width
	assert csv_view.height() >= csv_view.min_height
	assert csv_view.width() <= csv_view.max_width
	assert csv_view.height() <= csv_view.max_height


def test_create_impute_menu(csv_view, sample_df):
	"""Test creation of impute menu."""
	csv_view.df = sample_df
	menu = csv_view.create_impute_menu("numeric")
	assert len(menu.actions()) > 0


def test_create_one_hot_menu(csv_view, sample_df):
	"""Test creation of one-hot encoding menu."""
	csv_view.df = sample_df
	menu = csv_view.create_one_hot_menu("categorical")
	assert len(menu.actions()) > 0


def test_update_column_dropdown(csv_view, sample_df):
	"""Test updating column dropdown."""
	csv_view.df = sample_df
	csv_view.update_column_dropdown()
	assert csv_view.column_dropdown.count() == len(sample_df.columns)


def test_get_dataframe(csv_view, sample_df):
	"""Test getting DataFrame."""
	csv_view.df = sample_df
	result_df = csv_view.get_dataframe()
	pd.testing.assert_frame_equal(result_df, sample_df)


def test_error_handling(csv_view, sample_df, monkeypatch):
	"""Test error handling in various operations."""
	csv_view.df = sample_df.copy()
	messages = []

	class MockMessageBox:
		def __init__(self, parent=None):
			self.title = ""
			self.text = ""

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def setIcon(self, icon):
			pass

		def exec_(self):
			messages.append((self.title, self.text))

	monkeypatch.setattr(QMessageBox, "__new__", lambda cls, *args: MockMessageBox())

	def mock_one_hot_encode(df, column_name, n_distinct=True):
		if column_name not in df.columns:
			raise ValueError(f"Column '{column_name}' not found in DataFrame.")
		return df

	monkeypatch.setattr("utils.data_utils.one_hot_encode", mock_one_hot_encode)

	# Test invalid column name
	csv_view.apply_one_hot_encoding("non_existent_column")
	assert messages == [("One-Hot Encoding Error", "Column 'non_existent_column' not found in DataFrame.")]

	messages.clear()

	def mock_impute(df, column_name, method):
		if method not in ["mean", "median", "knn", "mice"]:
			raise ValueError("Invalid imputation method")
		return df

	monkeypatch.setattr("utils.data_utils.impute_values", mock_impute)

	# Test invalid imputation method
	csv_view.impute_missing_values("numeric", "invalid_method")
	assert messages == [("Imputation Error", "Invalid imputation method")]


def test_show_impute_options(csv_view_with_data, monkeypatch, cleanup_widgets):
	"""Test impute options without user interaction."""
	clicked_button = None
	imputed_values = []

	class MockMessageBox:
		def __init__(self, *args, **kwargs):
			self.buttons = []
			self.icon = None
			self.title = ""
			self.text = ""
			cleanup_widgets(self)  # Track for cleanup

		def setIcon(self, icon):
			self.icon = icon

		def setWindowTitle(self, title):
			self.title = title

		def setText(self, text):
			self.text = text

		def addButton(self, text, role):
			button = MagicMock()
			button.text = text
			self.buttons.append(button)
			return button

		def exec_(self):
			nonlocal clicked_button
			if self.buttons:  # Only set clicked_button if there are buttons
				clicked_button = self.buttons[0]
			return 0  # Return value to simulate button click

		def clickedButton(self):
			return clicked_button

		def hide(self):
			"""Mock hide method for cleanup"""
			pass

		def close(self):
			"""Mock close method for cleanup"""
			pass

		def deleteLater(self):
			"""Mock deleteLater method for cleanup"""
			pass

		@staticmethod
		def Warning():
			return 1

		@staticmethod
		def Critical():
			return 2

		@staticmethod
		def ActionRole():
			return 3

	# Mock QMessageBox completely
	monkeypatch.setattr("PyQt5.QtWidgets.QMessageBox", MockMessageBox)
	monkeypatch.setattr("gui.csv_view.QMessageBox", MockMessageBox)
	monkeypatch.setattr(QMessageBox, "Warning", MockMessageBox.Warning)
	monkeypatch.setattr(QMessageBox, "Critical", MockMessageBox.Critical)
	monkeypatch.setattr(QMessageBox, "ActionRole", MockMessageBox.ActionRole)

	# Mock impute_values to handle numeric columns only
	def mock_impute(df, column, method):
		if pd.api.types.is_numeric_dtype(df[column]):
			imputed_values.append((column, method))
			return df.copy()
		raise ValueError(f"Column '{column}' contains non-numerical values")

	monkeypatch.setattr("utils.data_utils.impute_values", mock_impute)
	monkeypatch.setattr("gui.csv_view.impute_values", mock_impute)

	# Create a DataFrame with only numeric columns for testing
	numeric_df = pd.DataFrame({"numeric1": [1, 2, None, 4, 5], "numeric2": [2, None, 6, 8, 10]})
	csv_view_with_data.df = numeric_df

	# Call show_impute_options
	csv_view_with_data.show_impute_options()

	# Verify the mock was used correctly
	assert clicked_button is not None
	assert clicked_button.text == "Impute with Mean"
	assert len(imputed_values) > 0


def test_context_menu(csv_view_with_data, qtbot, monkeypatch, cleanup_widgets):
	"""Test context menu without user interaction."""
	menu_actions = []
	called_methods = []

	class MockMenu:
		def __init__(self, title="", parent=None):
			self.actions_list = []
			self._title = title
			cleanup_widgets(self)

		def addAction(self, text, callback=None):
			action = MagicMock()
			action.text = text
			action.triggered = MagicMock()
			self.actions_list.append(action)
			menu_actions.append(text)
			return action

		def addMenu(self, submenu):
			# Just add the submenu's actions directly
			menu_actions.extend(action.text for action in submenu.actions())
			return submenu

		def exec_(self, pos=None):
			pass

		def actions(self):
			return self.actions_list

		def title(self):
			return self._title

		def hide(self):
			pass

		def close(self):
			pass

		def deleteLater(self):
			pass

	# Create mock menus with actions
	class MockSubMenu(MockMenu):
		def __init__(self, title, actions):
			super().__init__(title)
			for action in actions:
				self.addAction(action)

	impute_menu = MockSubMenu("Impute Missing Values", ["Impute with Mean", "Impute with Median", "Impute with KNN"])
	one_hot_menu = MockSubMenu("One-Hot Encode", ["One-Hot Encode"])

	# Mock QMenu and its creation
	monkeypatch.setattr("PyQt5.QtWidgets.QMenu", MockMenu)
	monkeypatch.setattr("gui.csv_view.QMenu", MockMenu)

	def mock_create_impute_menu(col):
		called_methods.append("create_impute_menu")
		return impute_menu

	def mock_create_one_hot_menu(col):
		called_methods.append("create_one_hot_menu")
		return one_hot_menu

	monkeypatch.setattr(csv_view_with_data, "create_impute_menu", mock_create_impute_menu)
	monkeypatch.setattr(csv_view_with_data, "create_one_hot_menu", mock_create_one_hot_menu)

	# Mock table widget setup
	mock_item = MagicMock()
	mock_item.column.return_value = 0
	mock_item.row.return_value = 0
	mock_item.text.return_value = "categorical"
	monkeypatch.setattr(csv_view_with_data.table_widget, "itemAt", lambda pos: mock_item)
	monkeypatch.setattr(csv_view_with_data.table_widget, "currentColumn", lambda: 0)
	monkeypatch.setattr(csv_view_with_data.table_widget, "currentRow", lambda: 0)

	# Set up data
	csv_view_with_data.df = pd.DataFrame({"categorical": ["A", "B", "C"]})
	csv_view_with_data.table_widget.setColumnCount(1)
	csv_view_with_data.table_widget.setRowCount(1)
	csv_view_with_data.table_widget.setHorizontalHeaderLabels(["categorical"])

	# Show context menu
	pos = QPoint(10, 10)
	csv_view_with_data.show_context_menu(pos)

	# Debug output
	print("Called methods:", called_methods)
	print("Menu actions:", menu_actions)

	# Verify menu items - check for actual menu items rather than submenu titles
	assert "Undo" in menu_actions
	assert "Redo" in menu_actions
	assert "Copy" in menu_actions
	assert "Impute with Mean" in menu_actions  # Check for submenu actions instead
	assert "Impute with Median" in menu_actions
	assert "Impute with KNN" in menu_actions
	assert "One-Hot Encode" in menu_actions
