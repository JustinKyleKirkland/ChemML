#!/usr/bin/env python3
"""
ChemML - Chemistry Machine Learning Interface
=============================================

Main entry point for the ChemML application, which provides:
- Data exploration and visualization
- Chemical feature extraction and analysis
- Machine learning model training and evaluation
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, Qt

from chemml.ui.views import ChemMLApp
from chemml.utils.logging_config import setup_logging


def main():
	"""Main entry point for the ChemML application."""
	# Configure logging
	setup_logging()
	logger = logging.getLogger("ChemML")
	logger.info("Starting ChemML application")

	# Set application metadata
	QCoreApplication.setApplicationName("ChemML")
	QCoreApplication.setApplicationVersion("1.0.0")
	QCoreApplication.setOrganizationName("ChemML Team")

	# Enable high DPI scaling for better appearance on high-resolution displays
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

	# Create Qt application
	app = QApplication(sys.argv)

	# Set fusion style for a modern look across platforms
	app.setStyle("Fusion")

	# Create and show main application window
	main_window = ChemMLApp()
	main_window.show()

	# Run application event loop
	logger.info("Application initialized successfully")
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
