import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CSVImageApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CSV to Image Converter")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_button)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

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

            plt.figure(figsize=(8, 6))
            plt.axis("tight")
            plt.axis("off")
            the_table = plt.table(
                cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
            )
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1.2, 1.2)

            plt.savefig(self.image_path, bbox_inches="tight", dpi=300)
            plt.close()
            logging.info(f"Image saved: {self.image_path}")

            self.image_label.setPixmap(QPixmap(self.image_path))
            logging.info("Image displayed")
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
        if os.path.exists("csv_image.png"):
            os.remove(self.image_path)
            logging.info("Temporary image deleted.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVImageApp()
    window.show()
    logging.info("Application started.")
    sys.exit(app.exec_())
