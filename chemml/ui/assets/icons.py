"""
ChemML Icon Resources
====================

This module provides access to icon resources used throughout the ChemML application.
Icons are created using basic QIcon resources, with custom colors for a consistent theme.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication

from chemml.ui.assets.colors import COLORS


def create_data_icon():
	"""Create an icon for the Data tab."""
	pixmap = QPixmap(32, 32)
	pixmap.fill(Qt.transparent)

	painter = QPainter(pixmap)
	painter.setRenderHint(QPainter.Antialiasing)

	# Draw a table/spreadsheet icon
	painter.setPen(QPen(QColor(COLORS["primary"]), 2))
	painter.setBrush(QColor(COLORS["light"]))

	# Draw outer rectangle (spreadsheet)
	painter.drawRect(4, 4, 24, 24)

	# Draw grid lines
	painter.setPen(QPen(QColor(COLORS["primary"]), 1))
	painter.drawLine(4, 12, 28, 12)  # Horizontal line 1
	painter.drawLine(4, 20, 28, 20)  # Horizontal line 2
	painter.drawLine(16, 4, 16, 28)  # Vertical line

	painter.end()
	return QIcon(pixmap)


def create_plot_icon():
	"""Create an icon for the Plot/Visualization tab."""
	pixmap = QPixmap(32, 32)
	pixmap.fill(Qt.transparent)

	painter = QPainter(pixmap)
	painter.setRenderHint(QPainter.Antialiasing)

	# Draw a simple bar chart
	painter.setPen(QPen(QColor(COLORS["primary"]), 2))
	painter.setBrush(QColor(COLORS["primary"]))

	# X and Y axis
	painter.drawLine(4, 28, 28, 28)  # X-axis
	painter.drawLine(4, 4, 4, 28)  # Y-axis

	# Draw bars
	painter.drawRect(8, 16, 4, 12)  # Bar 1
	painter.drawRect(14, 8, 4, 20)  # Bar 2
	painter.drawRect(20, 20, 4, 8)  # Bar 3

	painter.end()
	return QIcon(pixmap)


def create_ml_icon():
	"""Create an icon for the Machine Learning tab."""
	pixmap = QPixmap(32, 32)
	pixmap.fill(Qt.transparent)

	painter = QPainter(pixmap)
	painter.setRenderHint(QPainter.Antialiasing)

	# Draw a neural network styled icon
	painter.setPen(QPen(QColor(COLORS["primary"]), 2))

	# Draw nodes
	painter.setBrush(QColor(COLORS["light"]))

	# Input layer
	painter.drawEllipse(4, 6, 6, 6)
	painter.drawEllipse(4, 16, 6, 6)
	painter.drawEllipse(4, 26, 6, 6)

	# Hidden layer
	painter.drawEllipse(14, 11, 6, 6)
	painter.drawEllipse(14, 21, 6, 6)

	# Output layer
	painter.drawEllipse(24, 16, 6, 6)

	# Draw connections
	painter.setPen(QPen(QColor(COLORS["secondary"]), 1))

	# Input to hidden connections
	painter.drawLine(10, 9, 14, 14)  # Input 1 to Hidden 1
	painter.drawLine(10, 19, 14, 14)  # Input 2 to Hidden 1
	painter.drawLine(10, 29, 14, 24)  # Input 3 to Hidden 2
	painter.drawLine(10, 9, 14, 24)  # Input 1 to Hidden 2
	painter.drawLine(10, 19, 14, 24)  # Input 2 to Hidden 2

	# Hidden to output connections
	painter.drawLine(20, 14, 24, 19)  # Hidden 1 to Output
	painter.drawLine(20, 24, 24, 19)  # Hidden 2 to Output

	painter.end()
	return QIcon(pixmap)


def create_settings_icon():
	"""Create an icon for the Settings tab."""
	pixmap = QPixmap(32, 32)
	pixmap.fill(Qt.transparent)

	painter = QPainter(pixmap)
	painter.setRenderHint(QPainter.Antialiasing)

	# Draw a gear icon
	painter.setPen(QPen(QColor(COLORS["dark"]), 2))
	painter.setBrush(QColor(COLORS["light"]))

	# Draw outer circle
	painter.drawEllipse(8, 8, 16, 16)

	# Draw inner circle
	painter.drawEllipse(12, 12, 8, 8)

	# Draw gear teeth
	painter.setPen(QPen(QColor(COLORS["dark"]), 2))
	painter.drawLine(16, 4, 16, 8)  # Top
	painter.drawLine(16, 24, 16, 28)  # Bottom
	painter.drawLine(4, 16, 8, 16)  # Left
	painter.drawLine(24, 16, 28, 16)  # Right

	# Draw diagonal teeth
	painter.drawLine(8, 8, 5, 5)  # Top-left
	painter.drawLine(24, 8, 27, 5)  # Top-right
	painter.drawLine(8, 24, 5, 27)  # Bottom-left
	painter.drawLine(24, 24, 27, 27)  # Bottom-right

	painter.end()
	return QIcon(pixmap)


def create_molecule_icon():
	"""Create an icon representing a molecule."""
	pixmap = QPixmap(32, 32)
	pixmap.fill(Qt.transparent)

	painter = QPainter(pixmap)
	painter.setRenderHint(QPainter.Antialiasing)

	# Draw a simple molecule structure
	painter.setPen(QPen(QColor(COLORS["primary"]), 2))

	# Draw atoms (nodes)
	painter.setBrush(QColor("#FF5733"))  # Carbon - orange-red
	painter.drawEllipse(13, 5, 6, 6)  # Top atom

	painter.setBrush(QColor("#3498DB"))  # Oxygen - blue
	painter.drawEllipse(5, 16, 6, 6)  # Left atom

	painter.setBrush(QColor("#28B463"))  # Nitrogen - green
	painter.drawEllipse(21, 16, 6, 6)  # Right atom

	painter.setBrush(QColor("#F4D03F"))  # Hydrogen - yellow
	painter.drawEllipse(13, 22, 6, 6)  # Bottom atom

	# Draw bonds (connections)
	painter.setPen(QPen(QColor(COLORS["dark"]), 2))
	painter.drawLine(16, 11, 8, 16)  # Top to Left
	painter.drawLine(16, 11, 24, 16)  # Top to Right
	painter.drawLine(16, 22, 8, 19)  # Bottom to Left
	painter.drawLine(16, 22, 24, 19)  # Bottom to Right

	painter.end()
	return QIcon(pixmap)


def get_standard_icon(style_type):
	"""Get a standard icon from the application style.

	Args:
	    style_type: QStyle.StandardPixmap enum value

	Returns:
	    QIcon object
	"""
	return QApplication.style().standardIcon(style_type)
