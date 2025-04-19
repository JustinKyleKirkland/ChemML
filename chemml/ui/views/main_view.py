import logging
import sys

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget, QLabel, QStatusBar, QSplashScreen

from chemml.ui.controllers import CSVController
from chemml.ui.views.csv_view import CSVView
from chemml.ui.views.plot_view import PlotView
from chemml.ui.views.ml_view import MLView
from chemml.ui.assets.stylesheet import COMPLETE_STYLE
from chemml.ui.assets.icons import (
	create_data_icon,
	create_plot_icon,
	create_ml_icon,
	create_settings_icon,
	create_molecule_icon,
)
from chemml.utils.logging_config import setup_logging


class ChemMLApp(QWidget):
	"""Main application window for ChemML.

	This class serves as the main container for all application views including
	CSV viewing, plotting, and machine learning functionalities.
	"""

	def __init__(self) -> None:
		"""Initialize the main application window with all its components."""
		super().__init__()

		self.logger: logging.Logger = logging.getLogger("ChemML")
		self.setWindowTitle("ChemML")

		# Apply the application-wide stylesheet
		self.setStyleSheet(COMPLETE_STYLE)

		# Create shared data model and controllers
		self.csv_controller = CSVController()
		self.current_df = pd.DataFrame()

		# Initialize UI components
		self._setup_ui()
		self._connect_signals()

		self.logger.info("ChemML application initialized successfully.")

	def _setup_ui(self) -> None:
		"""Set up the user interface components."""
		self.layout = QVBoxLayout()
		self.layout.setSpacing(10)
		self.layout.setContentsMargins(10, 10, 10, 10)

		# Create an attractive header with logo and title - with proper object names
		header_widget = QWidget()
		header_widget.setObjectName("header_container")
		header_layout = QVBoxLayout(header_widget)
		header_layout.setSpacing(5)
		header_layout.setContentsMargins(15, 15, 15, 10)

		# App logo
		logo_label = QLabel()
		app_icon = create_molecule_icon()
		logo_pixmap = app_icon.pixmap(48, 48)
		logo_label.setPixmap(logo_pixmap)
		logo_label.setAlignment(Qt.AlignCenter)

		# App title
		title_label = QLabel("ChemML")
		title_label.setObjectName("app_title")
		title_font = QFont()
		title_font.setPointSize(20)
		title_font.setBold(True)
		title_label.setFont(title_font)
		title_label.setAlignment(Qt.AlignCenter)

		# App subtitle
		subtitle_label = QLabel("Chemistry Machine Learning Interface")
		subtitle_label.setObjectName("app_subtitle")
		subtitle_font = QFont()
		subtitle_font.setPointSize(12)
		subtitle_label.setFont(subtitle_font)
		subtitle_label.setAlignment(Qt.AlignCenter)

		# Add to header layout
		header_layout.addWidget(logo_label)
		header_layout.addWidget(title_label)
		header_layout.addWidget(subtitle_label)

		# Create main container for content
		main_container = QWidget()
		main_container.setObjectName("main_container")
		main_layout = QVBoxLayout(main_container)
		main_layout.setContentsMargins(0, 0, 0, 0)

		# Create tab widget with modern appearance
		self.tab_widget = QTabWidget()
		self.tab_widget.setDocumentMode(True)  # More modern look
		self.tab_widget.setTabPosition(QTabWidget.North)
		self.tab_widget.setMovable(True)  # Allow reordering tabs

		# Initialize views
		self.csv_view = CSVView()
		self.plot_view = PlotView()
		self.ml_view = MLView()
		self.ml_settings_view = QWidget()  # Placeholder for ML Settings tab

		# Add tabs with custom icons
		self.tab_widget.addTab(self.csv_view, create_data_icon(), "Data")
		self.tab_widget.addTab(self.plot_view, create_plot_icon(), "Visualize")
		self.tab_widget.addTab(self.ml_view, create_ml_icon(), "ML Models")
		self.tab_widget.addTab(self.ml_settings_view, create_settings_icon(), "Settings")

		# Set tab tooltips
		self.tab_widget.setTabToolTip(0, "Load and preprocess data")
		self.tab_widget.setTabToolTip(1, "Create visualizations and plots")
		self.tab_widget.setTabToolTip(2, "Train and evaluate machine learning models")
		self.tab_widget.setTabToolTip(3, "Configure advanced ML settings")

		# Add tab widget to main container
		main_layout.addWidget(self.tab_widget)

		# Status bar for additional feedback
		self.status_bar = QStatusBar()
		self.status_bar.showMessage("Ready")

		# Add widgets to main layout
		self.layout.addWidget(header_widget)
		self.layout.addWidget(main_container, 1)  # Give it stretch factor
		self.layout.addWidget(self.status_bar)

		self.setLayout(self.layout)

		# Set window size and minimum size constraint
		self.resize(1100, 800)
		self.setMinimumSize(800, 600)

	def _connect_signals(self) -> None:
		"""Connect all signal handlers between components."""
		# Connect CSV view data_ready signal to update other views
		self.csv_view.data_ready.connect(self.update_data)

		# Connect tab change signals
		self.tab_widget.currentChanged.connect(self._handle_tab_change)

	def _handle_tab_change(self, index):
		"""Handle tab change events."""
		tab_name = self.tab_widget.tabText(index)
		self.status_bar.showMessage(f"Switched to {tab_name} view")

	def update_data(self, df: pd.DataFrame) -> None:
		"""
		Update all views with new data.

		Args:
		    df: New DataFrame to use
		"""
		self.current_df = df
		self.plot_view.update_data(df)
		self.ml_view.set_dataframe(df)

		# Update status bar with data information
		self.status_bar.showMessage(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")

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

	# Create an attractive splash screen
	splash_pixmap = QPixmap(500, 300)
	splash_pixmap.fill(Qt.white)

	# Create a splash screen painter to add visual elements
	painter = QPainter(splash_pixmap)
	painter.setRenderHint(QPainter.Antialiasing)

	# Draw app icon
	app_icon = create_molecule_icon()
	icon_pixmap = app_icon.pixmap(96, 96)
	painter.drawPixmap(splash_pixmap.width() // 2 - icon_pixmap.width() // 2, 70, icon_pixmap)

	# Draw app title
	title_font = QFont()
	title_font.setPointSize(24)
	title_font.setBold(True)
	painter.setFont(title_font)
	painter.setPen(QColor("#007bff"))
	painter.drawText(0, 180, splash_pixmap.width(), 40, Qt.AlignCenter, "ChemML")

	# Draw app subtitle
	subtitle_font = QFont()
	subtitle_font.setPointSize(14)
	painter.setFont(subtitle_font)
	painter.setPen(QColor("#6c757d"))
	painter.drawText(0, 220, splash_pixmap.width(), 30, Qt.AlignCenter, "Chemistry Machine Learning Interface")

	# Draw loading text
	loading_font = QFont()
	loading_font.setPointSize(10)
	painter.setFont(loading_font)
	painter.setPen(QColor("#6c757d"))
	painter.drawText(0, 260, splash_pixmap.width(), 20, Qt.AlignCenter, "Loading application...")

	# Finish painting
	painter.end()

	# Create and show the splash screen
	splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
	splash.show()
	app.processEvents()

	# Create and show main window
	main_app = ChemMLApp()
	main_app.show()

	# Close splash screen
	splash.finish(main_app)

	logging.info("Application started.")
	sys.exit(app.exec_())
