import logging
import sys

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QLabel
)
from PyQt5.QtCore import Qt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CSVInteractiveApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV to Image Converter")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.ml_dropdown = QComboBox()
        self.ml_dropdown.addItems(
            ["Random Forest", "Logistic Regression", "SVM"])
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
    app = QApplication(sys.argv)
    window = CSVInteractiveApp()
    window.show()
    logging.info("Application started.")
    sys.exit(app.exec_())
