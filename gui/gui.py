import logging
import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget

from gui.csv_view import CSVView
from gui.ml_advanced_view import MLAdvancedView
from gui.ml_view import MLView
from gui.plot_view import PlottingWidget
from utils.logging_config import setup_logging


class CSVInteractiveApp(QWidget):
	"""Main application window for ChemML.

	This class serves as the main container for all application views including
	CSV viewing, plotting, and machine learning functionalities.
	"""

	def __init__(self) -> None:
		"""Initialize the main application window with all its components."""
		super().__init__()

		self.logger: logging.Logger = logging.getLogger("CSVInteractiveApp")
		self.setWindowTitle("ChemML")

		# Initialize UI components
		self._setup_ui()
		self._connect_signals()

		self.logger.info("CSVInteractiveApp initialized successfully.")

	def _setup_ui(self) -> None:
		"""Set up the user interface components."""
		self.layout = QVBoxLayout()
		self.tab_widget = QTabWidget()

		# Initialize view components
		self.csv_view = CSVView(self.tab_widget)
		self.plotting_widget = PlottingWidget(pd.DataFrame())
		self.ml_view = MLView()
		self.ml_advanced_view = MLAdvancedView()

		# Add tabs
		self.tab_widget.addTab(self.csv_view, "CSV View")
		self.tab_widget.addTab(self.plotting_widget, "Plot View")
		self.tab_widget.addTab(self.ml_view, "ML View")
		self.tab_widget.addTab(self.ml_advanced_view, "ML Settings")

		self.layout.addWidget(self.tab_widget)
		self.setLayout(self.layout)

	def _connect_signals(self) -> None:
		"""Connect all signal handlers between components."""
		self.csv_view.data_ready.connect(self.plotting_widget.update_data)
		self.csv_view.data_ready.connect(self.ml_view.update_column_selection)
		self.csv_view.data_ready.connect(self._handle_data_ready)

		# Connect ML settings changes to ML view
		self.ml_advanced_view.settings_changed.connect(self.ml_view.update_model_settings)

	def _handle_data_ready(self, df: pd.DataFrame) -> None:
		"""Handle the data_ready signal from CSV view.

		Args:
			df: The DataFrame containing the loaded data
		"""
		self.ml_view.set_dataframe(df)

	def closeEvent(self, event) -> None:
		"""Handle application close event.

		Args:
			event: The close event
		"""
		self.logger.info("Application closing...")
		event.accept()


if __name__ == "__main__":
	setup_logging()
	app = QApplication(sys.argv)
	main_app = CSVInteractiveApp()
	main_app.show()
	logging.info("Application started.")
	sys.exit(app.exec_())
