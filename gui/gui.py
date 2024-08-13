import logging
import sys

import pandas as pd
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.logging_config import setup_logging


class CSVInteractiveApp(QWidget):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger("CSVInteractiveApp")

        self.setWindowTitle("ChemML")
        self.setAcceptDrops(True)

        self.layout = QVBoxLayout()

        # Machine Learning Algorithm Dropdown
        # TODO: Make these actually link to something
        self.ml_dropdown = QComboBox()
        self.ml_dropdown.addItems(["Random Forest", "Logistic Regression", "SVM"])
        self.layout.addWidget(self.ml_dropdown)

        # Load CSV button
        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_button)

        # Filter interface
        filter_layout = QHBoxLayout()

        # Column selection dropdown
        self.column_dropdown = QComboBox()
        filter_layout.addWidget(QLabel("Filter Column:"))
        filter_layout.addWidget(self.column_dropdown)

        # Condition selection dropdown
        self.condition_dropdown = QComboBox(self)
        self.condition_dropdown.addItems(
            ["Contains", "Equals", "Starts with", "Ends with", ">=", "<=", ">", "<"]
        )
        filter_layout.addWidget(QLabel("Condition:"))
        filter_layout.addWidget(self.condition_dropdown)

        # Value Input
        self.value_edit = QLineEdit(self)
        self.value_edit.setPlaceholderText("Value")
        filter_layout.addWidget(QLabel("Value:"))
        filter_layout.addWidget(self.value_edit)

        # Apply filter button
        self.filter_button = QPushButton("Apply Filter")
        self.filter_button.clicked.connect(self.filter_data)
        filter_layout.addWidget(self.filter_button)

        self.layout.addLayout(filter_layout)

        # Table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setSortingEnabled(True)
        self.layout.addWidget(self.table_widget)

        self.setLayout(self.layout)

        self.min_width = 400
        self.min_height = 300
        self.max_width = 1200
        self.max_height = 800

        self.setMinimumSize(self.min_width, self.min_height)
        self.setMaximumSize(self.max_width, self.max_height)

    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)", options=options
        )
        if file_name:
            logging.info(f"Loading CSV file: {file_name}")
            self.display_csv_image(file_name)
        else:
            logging.info("No file selected")

    def display_csv_image(self, csv_file):
        try:
            self.df = pd.read_csv(csv_file)
            logging.info(f"CSV file read successfully: {csv_file}")

            self.column_dropdown.clear()
            self.column_dropdown.addItems(self.df.columns)

            self.table_widget.setRowCount(self.df.shape[0])
            self.table_widget.setColumnCount(self.df.shape[1])
            self.table_widget.setHorizontalHeaderLabels(self.df.columns)

            for row in range(self.df.shape[0]):
                for col in range(self.df.shape[1]):
                    item = QTableWidgetItem(str(self.df.iat[row, col]))
                    self.table_widget.setItem(row, col, item)

            logging.info("CSV data displayed in the table.")
            self.adjust_window_size()
        except Exception as e:
            self.show_error_message("Error loading CSV file", str(e))
            logging.error(f"Error loading CSV file: {e}")

    def adjust_window_size(self):
        """Adjust the window size based on the table contents."""
        try:
            # Calculate required size based on table dimensions
            row_count = self.table_widget.rowCount()
            column_count = self.table_widget.columnCount()

            if row_count == 0 or column_count == 0:
                return  # No data to display

            # Get font metrics for calculating size
            font_metrics = QFontMetrics(self.table_widget.font())
            row_height = font_metrics.height() + 10  # Row height with padding
            column_widths = [
                font_metrics.width(self.table_widget.horizontalHeaderItem(col).text())
                + 20
                for col in range(column_count)
            ]

            # Calculate the total height and width
            total_height = (
                row_count * row_height
                + self.table_widget.horizontalHeader().height()
                + 50
            )  # Header + some padding
            total_width = sum(column_widths) + 30  # Some padding

            # Set new geometry, ensuring it stays within limits
            new_width = max(self.min_width, min(total_width, self.max_width))
            new_height = max(self.min_height, min(total_height, self.max_height))

            self.setGeometry(self.x(), self.y(), new_width, new_height)
        except Exception as e:
            logging.error(f"Error adjusting window size: {e}")

    def filter_data(self):
        column = self.column_dropdown.currentText()
        condition = self.condition_dropdown.currentText()
        value = self.value_edit.text()

        operations = {
            "Equals": lambda df: df[df[column] == value],
            "Contains": lambda df: df[
                df[column].astype(str).str.contains(value, case=False, na=False)
            ],
            "Starts with": lambda df: df[
                df[column].astype(str).str.startswith(value, na=False)
            ],
            "Ends with": lambda df: df[
                df[column].astype(str).str.endswith(value, na=False)
            ],
            ">=": lambda df: df[df[column] >= float(value)],
            "<=": lambda df: df[df[column] <= float(value)],
            ">": lambda df: df[df[column] > float(value)],
            "<": lambda df: df[df[column] < float(value)],
        }

        try:
            if column and condition and value:
                logging.info(f"Applying filter: {column} {condition} {value}")
                filtered_df = operations.get(condition, lambda df: df)(self.df)
                self.update_table(filtered_df)
            else:
                raise ValueError("Invalid filter inputs.")
        except Exception as e:
            self.show_error_message("Error applying filter", str(e))
            logging.error(f"Error applying filter: {e}")

    def show_error_message(self, title, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle(title)
        error_dialog.setText(message)
        error_dialog.exec_()

    def update_table(self, df):
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setColumnCount(df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(df.columns)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[row, col]))
                self.table_widget.setItem(row, col, item)

        logging.info("Filtered data displayed in the table.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            logging.info("Drag event accepted.")
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(".csv"):
                logging.info(f"File dropped: {file_path}")
                self.display_csv_image(file_path)
            else:
                logging.warning("Invalid file type.")

    def closeEvent(self, event):
        event.accept()


if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    window = CSVInteractiveApp()
    window.show()
    logging.getLogger(__name__).info("Application started.")
    sys.exit(app.exec_())
