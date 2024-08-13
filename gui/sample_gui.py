import logging
import sys

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.logging_config import setup_logging


class CSVInteractiveApp(QWidget):
    """
    A PyQt5 application for converting CSV files to images.

    Attributes:
        ml_dropdown (QComboBox): A dropdown menu for selecting machine learning algorithms.
        load_button (QPushButton): A button for loading a CSV file.
        table_widget (QTableWidget): A table widget for displaying the CSV data.
        image_path (str): The path to save the generated image.

    Methods:
        __init__(): Initializes the CSVInteractiveApp.
        load_csv(): Opens a file dialog to load a CSV file.
        display_csv_image(csv_file): Displays the CSV data in the table widget.
        dragEnterEvent(event): Handles the drag enter event for file dropping.
        dropEvent(event): Handles the drop event for file dropping.
        closeEvent(event): Handles the close event of the application.
    """

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger("CSVInteractiveApp")

        self.setWindowTitle("CSV to Image Converter")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.ml_dropdown = QComboBox()
        self.ml_dropdown.addItems(["Random Forest", "Logistic Regression", "SVM"])
        self.layout.addWidget(self.ml_dropdown)

        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_button)

        self.table_widget = QTableWidget(self)
        self.layout.addWidget(self.table_widget)

        self.setLayout(self.layout)

        self.setAcceptDrops(True)

        self.image_path = "csv_image.png"

    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)", options=options
        )
        if file_name:
            logging.info(f"Loading CSV file: {file_name}")
            self.display_csv_image(file_name)
        else:
            print("test1")
            logging.info("No file selected")

    def display_csv_image(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            logging.info(f"CSV file read successfully: {csv_file}")

            self.table_widget.setRowCount(df.shape[0])
            self.table_widget.setColumnCount(df.shape[1])
            self.table_widget.setHorizontalHeaderLabels(df.columns)

            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iat[row, col]))
                    self.table_widget.setItem(row, col, item)

            logging.info("CSV data displayed in the table.")
        except Exception as e:
            logging.error(f"Error loading CSV file: {e}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            logging.info("Drag event accepted.")
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
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
