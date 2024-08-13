import io
import logging
import sys

import pandas as pd
from PIL import Image
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QApplication,
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
    QVBoxLayout,
    QWidget,
)
from rdkit import Chem
from rdkit.Chem import Draw

from utils.data_utils import impute_values, one_hot_encode, validate_csv
from utils.gui_utils import pil_image_to_pixmap
from utils.logging_config import setup_logging


class CSVInteractiveApp(QWidget):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger("CSVInteractiveApp")
        self.df = pd.DataFrame()  # Initialize the DataFrame
        self.undo_stack = []  # Stack to store DataFrame history
        self.redo_stack = []  # Stack to store undone DataFrame history

        self.setWindowTitle("ChemML")
        self.setAcceptDrops(True)

        self.layout = QVBoxLayout()
        self.error_label = self.create_error_label()
        self.ml_button, self.ml_menu = self.create_ml_button()
        self.load_button = self.create_load_button()
        self.column_dropdown, self.condition_dropdown, self.value_edit = (
            self.create_filter_widgets()
        )
        self.filter_button = self.create_filter_button()
        self.table_widget = self.create_table_widget()

        self.layout.addWidget(self.error_label)
        self.layout.addWidget(self.ml_button)
        self.layout.addWidget(self.load_button)
        self.layout.addLayout(self.create_filter_layout())
        self.layout.addWidget(self.table_widget)
        self.setLayout(self.layout)

        self.set_window_size_limits()
        self.setup_context_menu()

    def create_error_label(self):
        error_label = QLabel()
        error_label.setStyleSheet("color: red")
        return error_label

    def create_ml_button(self):
        ml_button = QPushButton("Select ML Methods")
        ml_button.setCheckable(True)
        ml_menu = QMenu(ml_button)

        for method in ["Random Forest", "Logistic Regression", "SVM"]:
            action = ml_menu.addAction(method)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, a=action: self.toggle_ml_method(a, checked)
            )

        ml_button.clicked.connect(self.show_ml_menu)
        ml_button.setMenu(ml_menu)  # Set the menu to the button
        return ml_button, ml_menu

    def create_load_button(self):
        load_button = QPushButton("Load CSV")
        load_button.clicked.connect(self.load_csv)
        return load_button

    def create_filter_widgets(self):
        column_dropdown = QComboBox()
        condition_dropdown = QComboBox()
        condition_dropdown.addItems(
            ["Contains", "Equals", "Starts with", "Ends with", ">=", "<=", ">", "<"]
        )
        value_edit = QLineEdit()
        value_edit.setPlaceholderText("Value")
        return column_dropdown, condition_dropdown, value_edit

    def create_filter_button(self):
        filter_button = QPushButton("Apply Filter")
        filter_button.clicked.connect(self.filter_data)
        return filter_button

    def create_table_widget(self):
        table_widget = QTableWidget(self)
        table_widget.setSortingEnabled(True)
        return table_widget

    def create_filter_layout(self):
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Column:"))
        filter_layout.addWidget(self.column_dropdown)
        filter_layout.addWidget(QLabel("Condition:"))
        filter_layout.addWidget(self.condition_dropdown)
        filter_layout.addWidget(QLabel("Value:"))
        filter_layout.addWidget(self.value_edit)
        filter_layout.addWidget(self.filter_button)
        return filter_layout

    def set_window_size_limits(self):
        self.min_width, self.min_height = 400, 300
        self.max_width, self.max_height = 1200, 800
        self.setMinimumSize(self.min_width, self.min_height)
        self.setMaximumSize(self.max_width, self.max_height)

    def setup_context_menu(self):
        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos: QPoint):
        item = self.table_widget.itemAt(pos)

        if item is not None:
            column_index = item.column()
            column_name = self.df.columns[column_index]

            context_menu = QMenu(self)
            context_menu.addAction("Undo", self.undo)
            context_menu.addAction("Redo", self.redo)

            if self.df[column_name].dtype == "object":
                context_menu.addMenu(self.create_one_hot_menu(column_name))
                context_menu.addAction(
                    "Visualize Molecule", lambda: self.visualize_chemdraw(item.text())
                )
            elif self.df[column_name].isnull().any():
                context_menu.addMenu(self.create_impute_menu(column_name))

            context_menu.addAction("Copy", self.copy_selected_values)
            context_menu.exec_(self.table_widget.mapToGlobal(pos))

    def visualize_chemdraw(self, smiles: str):
        if self.is_valid_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol)

            print(f"SMILES: {smiles}")
            if mol is None:
                print("Invalid SMILES string.")

            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)

            self.show_image_popup(img_byte_array)
        else:
            self.show_error_message("Invalid SMILES", "Invalid SMILES string.")

    def is_valid_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def show_image_popup(self, img_byte_array):
        img = Image.open(img_byte_array)
        img_pixmap = pil_image_to_pixmap(img)

        img_label = QLabel()
        img_label.setPixmap(img_pixmap)

        popup = QWidget()
        popup.setWindowTitle("ChemDraw Visualization")
        layout = QVBoxLayout()
        layout.addWidget(img_label)
        popup.setLayout(layout)

        print(f"Image size: {img.size}")
        print(f"Pixmap size: {img_pixmap.size()}")

        popup.resize(img_pixmap.width(), img_pixmap.height())
        main_window_rect = self.geometry()

        popup_x = main_window_rect.x() + (main_window_rect.width() - popup.width()) // 2
        popup_y = (
            main_window_rect.y() + (main_window_rect.height() - popup.height()) // 2
        )
        popup.move(popup_x, popup_y)
        popup.setAttribute(Qt.WA_DeleteOnClose)

        popup.show()

    def create_one_hot_menu(self, column_name):
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

    def create_impute_menu(self, column_name):
        impute_menu = QMenu("Impute Missing Values", self)
        impute_menu.addAction(
            "Impute with mean", lambda: self.impute_missing_values(column_name, "mean")
        )
        impute_menu.addAction(
            "Impute with median",
            lambda: self.impute_missing_values(column_name, "median"),
        )
        return impute_menu

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

            errors = validate_csv(self.df)
            self.show_errors(errors)

            self.column_dropdown.clear()
            self.column_dropdown.addItems(self.df.columns)

            self.update_table(self.df)
            logging.info("CSV data displayed in the table.")
            self.adjust_window_size()
        except Exception as e:
            self.show_error_message("Error loading CSV file", str(e))
            logging.error(f"Error loading CSV file: {e}")

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.df.copy())
            self.df = self.undo_stack.pop()
            self.update_table(self.df)
            self.column_dropdown.clear()
            self.column_dropdown.addItems(self.df.columns)

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.df.copy())
            self.df = self.redo_stack.pop()
            self.update_table(self.df)
            self.column_dropdown.clear()
            self.column_dropdown.addItems(self.df.columns)

    def apply_one_hot_encoding(self, column_name, n_distinct=True):
        """Apply one-hot encoding and update the DataFrame and table."""
        try:
            self.undo_stack.append(self.df.copy())
            self.redo_stack.clear()

            logging.info(f"One-hot encoding column: {column_name}")
            self.df = one_hot_encode(self.df, column_name, n_distinct)
            self.update_table(self.df)
            self.column_dropdown.clear()
            self.column_dropdown.addItems(self.df.columns)
            logging.info(f"One-hot encoding completed for column: {column_name}")
        except ValueError as e:
            self.show_error_message("Error", str(e))

    def impute_missing_values(self, column_name, method):
        try:
            self.undo_stack.append(self.df.copy())
            self.redo_stack.clear()

            logging.info(f"Imputing missing values for column: {column_name}")
            self.df = impute_values(self.df, column_name, method)
            self.update_table(self.df)
            logging.info(f"Imputed missing values for column: {column_name}")
        except ValueError as e:
            self.show_error_message("Error", str(e))

    def show_errors(self, errors):
        if errors:
            error_message = "\n".join(errors)
            self.show_error_message("CSV Validation Errors", error_message)
        else:
            self.error_label.clear()

    def show_ml_menu(self):
        self.ml_menu.exec_(
            self.ml_button.mapToGlobal(self.ml_button.rect().bottomLeft())
        )

    def toggle_ml_method(self, action, checked):
        if checked:
            logging.info(f"Selected ML method: {action.text()}")
        else:
            logging.info(f"Deselected ML method: {action.text()}")

    def adjust_window_size(self):
        """Adjust the window size based on the table contents."""
        try:
            row_count = self.table_widget.rowCount()
            column_count = self.table_widget.columnCount()

            if row_count == 0 or column_count == 0:
                return

            font_metrics = QFontMetrics(self.table_widget.font())
            row_height = font_metrics.height() + 10
            column_widths = [
                font_metrics.width(self.table_widget.horizontalHeaderItem(col).text())
                + 20
                for col in range(column_count)
            ]

            total_height = (
                row_count * row_height
                + self.table_widget.horizontalHeader().height()
                + 50
            )
            total_width = sum(column_widths) + 30

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
                self.undo_stack.append(self.df.copy())
                self.redo_stack.clear()
                self.update_table(filtered_df)
                logging.info("Filtered data displayed in the table.")
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

    # TODO: Get it to actually copy correctly
    def copy_selected_values(self):
        clipboard = QApplication.clipboard()
        selected_items = self.table_widget.selectedItems()

        if selected_items:
            copied_text = ""

            for item in selected_items:
                copied_text += item.text() + "\t"
                if item.column() == self.table_widget.columnCount() - 1:
                    copied_text = copied_text[:-1] + "\n"
            clipboard.setText(copied_text.strip())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            self.copy_selected_values()
        super().keyPressEvent(event)

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
