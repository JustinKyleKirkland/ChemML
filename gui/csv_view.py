import logging
from typing import List, Tuple

import pandas as pd
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFontMetrics
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from utils.data_utils import impute_values, one_hot_encode, validate_csv


class CSVView(QWidget):
    data_ready = pyqtSignal(pd.DataFrame)

    def __init__(self, tab_widget: QTabWidget) -> None:
        super().__init__()
        self.tab_widget = tab_widget
        self.logger: logging.Logger = logging.getLogger("CSVView")
        self.df: pd.DataFrame = pd.DataFrame()
        self.undo_stack: List[pd.DataFrame] = []
        self.redo_stack: List[pd.DataFrame] = []

        self.set_window_size_limits()
        self.layout = QVBoxLayout()

        # Initialize widgets
        self.error_label = self.create_error_label()
        self.load_button = self.create_load_button()
        self.column_dropdown, self.condition_dropdown, self.value_edit = self.create_filter_widgets()
        self.filter_button = self.create_filter_button()
        self.table_widget = self.create_table_widget()

        self.table_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.setup_context_menu()

        # Arrange layout
        self.layout.addWidget(self.table_widget)
        self.layout.addWidget(self.error_label)
        self.layout.addWidget(self.load_button)
        self.layout.addLayout(self.create_filter_layout())
        self.setLayout(self.layout)

        logging.info("CSVView initialized successfully.")

    def create_error_label(self) -> QLabel:
        logging.debug("Creating error label widget.")
        error_label = QLabel()
        error_label.setStyleSheet("color: red")
        return error_label

    def create_load_button(self) -> QPushButton:
        logging.debug("Creating load button widget.")
        load_button = QPushButton("Load CSV")
        load_button.clicked.connect(self.load_csv)
        return load_button

    def create_filter_widgets(self) -> Tuple[QComboBox, QComboBox, QLineEdit]:
        logging.debug("Creating filter widgets.")
        column_dropdown = QComboBox()
        condition_dropdown = QComboBox()
        condition_dropdown.addItems(["Contains", "Equals", "Starts with", "Ends with", ">=", "<=", ">", "<"])
        value_edit = QLineEdit()
        value_edit.setPlaceholderText("Value")
        return column_dropdown, condition_dropdown, value_edit

    def create_filter_button(self) -> QPushButton:
        logging.debug("Creating filter button widget.")
        filter_button = QPushButton("Apply Filter")
        filter_button.clicked.connect(self.filter_data)
        return filter_button

    def create_table_widget(self) -> QTableWidget:
        logging.debug("Creating table widget.")
        table_widget = QTableWidget(self)
        table_widget.setSortingEnabled(True)
        return table_widget

    def create_filter_layout(self) -> QHBoxLayout:
        logging.debug("Creating filter layout.")
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Column:"))
        filter_layout.addWidget(self.column_dropdown)
        filter_layout.addWidget(QLabel("Condition:"))
        filter_layout.addWidget(self.condition_dropdown)
        filter_layout.addWidget(QLabel("Value:"))
        filter_layout.addWidget(self.value_edit)
        filter_layout.addWidget(self.filter_button)
        return filter_layout

    def load_csv(self) -> None:
        logging.info("Load CSV button clicked.")
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
        if file_name:
            logging.info(f"Selected CSV file: {file_name}")
            self.display_csv_image(file_name)
            self.data_ready.emit(self.df)
        else:
            logging.info("No file selected")

    def display_csv_image(self, csv_file: str) -> None:
        try:
            logging.info(f"Reading CSV file: {csv_file}")
            self.df = pd.read_csv(csv_file)
            logging.info(f"CSV file '{csv_file}' read successfully.")

            errors = validate_csv(self.df)
            self.show_errors(errors)

            # self.tab_widget.widget(1).update_data(self.df)

            self.data_ready.emit(self.df)

            self.column_dropdown.clear()
            self.column_dropdown.addItems(self.df.columns)

            self.update_table(self.df)

            logging.info("CSV data displayed in the table.")
            self.adjust_window_size()
        except Exception as e:
            self.show_error_message("Error loading CSV file", str(e))
            logging.error(f"Error loading CSV file '{csv_file}': {e}")

    def undo(self) -> None:
        logging.info("Undo operation triggered.")
        if self.undo_stack:
            self.redo_stack.append(self.df.copy())
            self.df = self.undo_stack.pop()
            self.update_table(self.df)
            self.update_column_dropdown()
            self.data_ready.emit(self.df)
            self.show_errors(validate_csv(self.df))
            logging.info("Undo operation completed.")
        else:
            logging.info("Undo stack is empty; nothing to undo.")

    def redo(self) -> None:
        logging.info("Redo operation triggered.")
        if self.redo_stack:
            self.undo_stack.append(self.df.copy())
            self.df = self.redo_stack.pop()
            self.update_table(self.df)
            self.update_column_dropdown()
            self.data_ready.emit(self.df)
            self.show_errors(validate_csv(self.df))
            logging.info("Redo operation completed.")
        else:
            logging.info("Redo stack is empty; nothing to redo.")

    def apply_one_hot_encoding(self, column_name: str, n_distinct: bool = True) -> None:
        logging.info(f"Applying one-hot encoding to column: {column_name}")
        try:
            self.undo_stack.append(self.df.copy())
            self.redo_stack.clear()

            self.df = one_hot_encode(self.df, column_name, n_distinct)
            self.update_table(self.df)
            self.update_column_dropdown()

            self.data_ready.emit(self.df)

            # Check for null values in new columns
            new_columns_with_nulls = [col for col in self.df.columns if self.df[col].isnull().any()]
            if new_columns_with_nulls:
                warning_msg = (
                    f"Warning: The following columns contain null values: {', '.join(new_columns_with_nulls)}."
                )
                self.show_errors([warning_msg])
                logging.warning(warning_msg)
            else:
                logging.info(f"One-hot encoding applied to column: {column_name}")
        except ValueError as e:
            self.show_error_message("One-Hot Encoding Error", str(e))
            logging.error(f"Error in one-hot encoding: {e}")

    def impute_missing_values(self, column_name: str, strategy: str) -> None:
        logging.info(f"Imputing missing values for column: {column_name} using {strategy}")
        try:
            self.undo_stack.append(self.df.copy())
            self.redo_stack.clear()

            self.df = impute_values(self.df, column_name, strategy)

            # Debug: Log the state of the DataFrame column after imputation
            logging.debug(f"Post-imputation, {column_name} column data:\n{self.df[column_name]}")
            self.update_table(self.df)
            self.update_column_dropdown()

            if self.df[column_name].isnull().any():
                warning_msg = (
                    f"Warning: Column '{column_name}' still contains null values after applying {strategy} imputation."
                )
                self.show_errors([warning_msg])
                logging.warning(warning_msg)
            else:
                logging.info(f"Missing values imputed for column: {column_name} using {strategy}")
                self.show_errors([])  # Clear previous errors if successful
        except ValueError as e:
            self.show_error_message("Imputation Error", str(e))
            logging.error(f"Error in imputing missing values: {e}")

    def show_errors(self, errors: List[str]) -> None:
        if errors:
            logging.warning("Validation errors found in CSV data.")
            self.error_label.setText("\n".join(errors))
        else:
            self.error_label.setText("")
            logging.info("No validation errors found.")

    def adjust_window_size(self) -> None:
        try:
            logging.debug("Adjusting window size based on table content.")
            row_count = self.table_widget.rowCount()
            column_count = self.table_widget.columnCount()

            if column_count == 0 or row_count == 0:
                logging.warning("No data to adjust window size.")
                return

            # Calculate the height needed for all rows
            row_height = (
                max(self.table_widget.verticalHeader().sectionSize(i) for i in range(row_count)) + 10
            )  # Extra padding

            font_metrics = QFontMetrics(self.font())
            total_column_width = sum(
                max(
                    font_metrics.width(str(self.table_widget.item(row_index, col).text()))
                    for row_index in range(row_count)
                )
                for col in range(column_count)
            ) + (column_count * 75)  # Add padding for each column

            new_height = min(
                self.max_height, row_count * row_height + 10
            )  # Add some extra height for the header and margins
            new_width = min(self.max_width, total_column_width)

            self.resize(max(self.min_width, new_width), max(self.min_height, new_height))
            logging.info("Window size adjusted successfully.")
        except Exception as e:
            self.show_error_message("Error", f"Failed to adjust window size: {e}")
            logging.error(f"Failed to adjust window size: {e}")

    def create_impute_menu(self, column_name: str) -> QMenu:
        logging.debug(f"Creating impute menu for column: {column_name}.")
        impute_menu = QMenu("Impute Missing Values", self)
        impute_menu.addAction("Impute with mean", lambda: self.impute_missing_values(column_name, "mean"))
        impute_menu.addAction(
            "Impute with median",
            lambda: self.impute_missing_values(column_name, "median"),
        )
        return impute_menu

    def create_one_hot_menu(self, column_name: str) -> QMenu:
        logging.debug(f"Creating one-hot encode menu for column: {column_name}.")
        one_hot_menu = QMenu(f"One-hot encode '{column_name}'", self)
        one_hot_menu.addAction(
            "Create N distinct columns",
            lambda: self.apply_one_hot_encoding(column_name, n_distinct=True),
        )
        one_hot_menu.addAction(
            "Create N-1 distinct columns",
            lambda: self.apply_one_hot_encoding(column_name, n_distinct=False),
        )
        return one_hot_menu

    def update_table(self, df: pd.DataFrame) -> None:
        logging.debug("Updating table with new data.")
        self.table_widget.clear()
        self.table_widget.setRowCount(len(df))
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setHorizontalHeaderLabels(df.columns)

        for col_index, column in enumerate(df.columns):
            has_nan = df[column].isnull().any()  # Check if the column has any NaN values

            for row_index in range(len(df)):
                item = QTableWidgetItem(str(df.at[row_index, column]))

                if has_nan:
                    item.setBackground(QBrush(QColor(99, 7, 39)))  # Dark red color for the entire column

                self.table_widget.setItem(row_index, col_index, item)

        self.table_widget.resizeColumnsToContents()
        logging.info("Table updated successfully.")

    def filter_data(self) -> None:
        column_name = self.column_dropdown.currentText()
        condition = self.condition_dropdown.currentText()
        value = self.value_edit.text()

        if column_name and condition and value:
            try:
                logging.info(f"Applying filter: Column='{column_name}', Condition='{condition}', Value='{value}'")
                self.undo_stack.append(self.df.copy())
                self.redo_stack.clear()

                filtered_df = self.apply_filter(column_name, condition, value)
                self.update_table(filtered_df)
                logging.info("Filter applied successfully.")
            except Exception as e:
                self.show_error_message("Error applying filter", str(e))
                logging.error(f"Error applying filter: {e}")
        else:
            logging.warning("Filter criteria missing. Please select a column, condition, and value.")
            self.show_error_message("Invalid Filter", "Please select a column, condition, and value.")

    def apply_filter(self, column_name: str, condition: str, value: str) -> pd.DataFrame:
        if condition == "Contains":
            return self.df[self.df[column_name].str.contains(value)]
        elif condition == "Equals":
            return self.df[self.df[column_name] == value]
        elif condition == "Starts with":
            return self.df[self.df[column_name].str.startswith(value)]
        elif condition == "Ends with":
            return self.df[self.df[column_name].str.endswith(value)]
        elif condition in (">=", "<=", ">", "<"):
            return self.df[self.df[column_name].astype(float).eval(f"{condition} {float(value)}")]
        else:
            raise ValueError("Invalid filter condition")

    def copy_selected_values(self) -> None:
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            logging.warning("No cells selected to copy.")
            self.show_error_message("Copy Error", "No cells selected to copy.")
            return

        # Implement copy logic here (if needed)

    def show_error_message(self, title: str, message: str) -> None:
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle(title)
        error_dialog.setText(message)
        error_dialog.exec_()
        logging.error(f"Error dialog shown - Title: '{title}', Message: '{message}'")

    def set_window_size_limits(self) -> None:
        logging.info("Setting window size limits.")
        self.min_width, self.min_height = 400, 300
        self.max_width, self.max_height = 1200, 800
        self.setMinimumSize(self.min_width, self.min_height)
        self.setMaximumSize(self.max_width, self.max_height)

    def setup_context_menu(self) -> None:
        logging.debug("Setting up context menu for table widget.")
        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)

    def show_context_menu(self, pos: QPoint) -> None:
        logging.debug(f"Context menu requested at position: {pos}.")
        item = self.table_widget.itemAt(pos)

        if item is not None:
            column_index = item.column()
            column_name = self.df.columns[column_index]
            logging.debug(f"Context menu for column: {column_name}.")

            context_menu = QMenu(self)
            context_menu.addAction("Undo", self.undo)
            context_menu.addAction("Redo", self.redo)

            if self.df[column_name].dtype == "object":
                context_menu.addMenu(self.create_one_hot_menu(column_name))
            elif self.df[column_name].isnull().any():
                context_menu.addMenu(self.create_impute_menu(column_name))

            context_menu.addAction("Copy", self.copy_selected_values)
            context_menu.exec_(self.table_widget.mapToGlobal(pos))

    def update_column_dropdown(self) -> None:
        self.column_dropdown.clear()
        self.column_dropdown.addItems(self.df.columns)
