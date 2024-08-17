import logging
import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget

from gui.csv_view import CSVView
from gui.ml_view import MLView
from gui.plot_view import PlottingWidget
from utils.logging_config import setup_logging


class CSVInteractiveApp(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.logger: logging.Logger = logging.getLogger("CSVInteractiveApp")
        self.setWindowTitle("ChemML")

        # Layout setup
        self.layout = QVBoxLayout()
        self.tab_widget = QTabWidget()

        # Initialize CSV view and plotting widget
        self.csv_view = CSVView(self.tab_widget)
        self.plotting_widget = PlottingWidget(pd.DataFrame())
        self.ml_view = MLView()

        # Add tabs to the tab widget
        self.tab_widget.addTab(self.csv_view, "CSV View")
        self.tab_widget.addTab(self.plotting_widget, "Plot View")
        self.tab_widget.addTab(self.ml_view, "ML View")

        # Connect signals
        self.csv_view.data_ready.connect(self.plotting_widget.update_data)
        self.csv_view.data_ready.connect(self.ml_view.update_column_selection)

        # Set layout
        self.layout.addWidget(self.tab_widget)
        self.setLayout(self.layout)

        logging.info("CSVInteractiveApp initialized successfully.")


if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    main_app = CSVInteractiveApp()
    main_app.show()
    logging.info("Application started.")
    sys.exit(app.exec_())
