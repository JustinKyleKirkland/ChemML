import pandas as pd
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTabWidget

from gui.csv_view import CSVView


@pytest.fixture
def csv_view(qtbot):
    tab_widget = QTabWidget()
    view = CSVView(tab_widget)
    qtbot.addWidget(view)
    return view


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


def test_load_csv(csv_view, qtbot):
    # Simulate clicking the load button and selecting a CSV file
    qtbot.mouseClick(csv_view.load_button, Qt.LeftButton)
    assert not csv_view.df.empty


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


def test_apply_one_hot_encoding(csv_view):
    # Add some data to the undo stack
    csv_view.undo_stack.append(pd.DataFrame())

    # Call the apply_one_hot_encoding method
    csv_view.apply_one_hot_encoding("column_name")

    # Check if the apply_one_hot_encoding operation was successful
    assert len(csv_view.undo_stack) == 2
    assert len(csv_view.redo_stack) == 0


def test_impute_missing_values(csv_view):
    # Add some data to the undo stack
    csv_view.undo_stack.append(pd.DataFrame())

    # Call the impute_missing_values method
    csv_view.impute_missing_values("null_col", "mean")

    # Check if the impute_missing_values operation was successful
    assert len(csv_view.undo_stack) == 2
    assert len(csv_view.redo_stack) == 0


def test_show_errors(csv_view):
    # Call the show_errors method with some errors
    csv_view.show_errors(["Error 1", "Error 2"])

    # Check if the error label is updated correctly
    assert csv_view.error_label.text() == "Error 1\nError 2"


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


def test_copy_selected_values(csv_view):
    # Select some cells in the table widget
    csv_view.table_widget.setCurrentCell(0, 0)
    csv_view.table_widget.setCurrentCell(1, 1)

    # Call the copy_selected_values method
    csv_view.copy_selected_values()

    # TODO: Add assertions for the copied values


def test_update_column_dropdown(csv_view):
    # Create a test DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Set the test DataFrame as the current DataFrame
    csv_view.df = df

    # Call the update_column_dropdown method
    csv_view.update_column_dropdown()

    # Check if the column dropdown is updated correctly
    assert csv_view.column_dropdown.count() == len(df.columns)
    assert csv_view.column_dropdown.itemText(0) == "A"
    assert csv_view.column_dropdown.itemText(1) == "B"
