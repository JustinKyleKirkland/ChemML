"""
ChemML Stylesheet Configuration
===============================

This module contains styling definitions for the ChemML application UI.
These styles apply a modern, consistent appearance across all views.
"""

# Base application style - applies to all widgets
BASE_STYLE = """
    QWidget {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
        color: #333333;
    }
    QMainWindow, QDialog {
        background-color: #f8f9fa;
    }
    QTabWidget::pane {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 5px;
        background-color: #ffffff;
    }
    QTabBar::tab {
        background-color: #f0f2f5;
        border: 1px solid #dee2e6;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 16px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background-color: #ffffff;
        border-bottom: 1px solid white;
        font-weight: bold;
    }
    QTabBar::tab:hover:!selected {
        background-color: #e9ecef;
    }
    QGroupBox {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        margin-top: 1.5em;
        padding-top: 8px;
        font-weight: bold;
        background-color: white;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #495057;
    }
    QFrame {
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }
    QScrollArea {
        border: none;
        background-color: transparent;
    }
"""

# Button styles
BUTTON_STYLE = """
    QPushButton {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        min-height: 30px;
    }
    QPushButton:hover {
        background-color: #0069d9;
    }
    QPushButton:pressed {
        background-color: #0062cc;
    }
    QPushButton:disabled {
        background-color: #cccccc;
        color: #666666;
    }
    QPushButton:flat {
        background-color: transparent;
        color: #007bff;
        border: 1px solid #007bff;
    }
    QPushButton:flat:hover {
        background-color: rgba(0, 123, 255, 0.1);
    }
"""

# Secondary button for less emphasis
SECONDARY_BUTTON_STYLE = """
    QPushButton.secondary {
        background-color: #6c757d;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        min-height: 30px;
    }
    QPushButton.secondary:hover {
        background-color: #5a6268;
    }
    QPushButton.secondary:pressed {
        background-color: #545b62;
    }
"""

# Input field styles
INPUT_STYLE = """
    QLineEdit, QTextEdit, QPlainTextEdit {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 8px;
        background-color: white;
        selection-background-color: #007bff;
    }
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 1px solid #80bdff;
        outline: 0;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
    }
"""

# Table and list styles
TABLE_STYLE = """
    QTableView, QTableWidget {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background-color: white;
        gridline-color: #e9ecef;
        selection-background-color: #e2edff;
        selection-color: #212529;
        alternate-background-color: #f9f9f9;
    }
    QTableView::item, QTableWidget::item {
        padding: 4px;
        border: none;
    }
    QTableView::item:selected, QTableWidget::item:selected {
        background-color: #007bff;
        color: white;
    }
    QHeaderView::section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 6px;
        font-weight: bold;
    }
"""

# List widget styles
LIST_STYLE = """
    QListWidget {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        background-color: white;
        alternate-background-color: #f8f9fa;
        padding: 4px;
    }
    QListWidget::item {
        padding: 8px;
        border-radius: 4px;
        margin: 2px;
    }
    QListWidget::item:selected {
        background-color: #007bff;
        color: white;
    }
    QListWidget::item:hover:!selected {
        background-color: #f0f2f5;
    }
"""

# Dropdown menu styles
COMBO_STYLE = """
    QComboBox {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 8px 24px 8px 8px;
        min-width: 6em;
        background-color: white;
    }
    QComboBox::drop-down {
        border: none;
        subcontrol-position: center right;
        subcontrol-origin: padding;
        width: 20px;
    }
    QComboBox::down-arrow {
        width: 14px;
        height: 14px;
    }
    QComboBox:on {
        border-bottom-right-radius: 0;
        border-bottom-left-radius: 0;
    }
    QComboBox QListView {
        border: 1px solid #ced4da;
        border-radius: 0px;
        background-color: white;
        selection-background-color: #007bff;
        selection-color: white;
        padding: 4px;
    }
"""

# Status bar style
STATUS_STYLE = """
    QStatusBar {
        background-color: #f8f9fa;
        border-top: 1px solid #dee2e6;
        padding: 5px;
        color: #6c757d;
    }
    QStatusBar QLabel {
        color: #6c757d;
    }
"""

# Progress bar style
PROGRESS_STYLE = """
    QProgressBar {
        border: 1px solid #dee2e6;
        border-radius: 4px;
        text-align: center;
        background-color: #f0f2f5;
        height: 20px;
    }
    QProgressBar::chunk {
        background-color: #007bff;
        border-radius: 3px;
    }
"""

# Message box style
MESSAGEBOX_STYLE = """
    QMessageBox {
        background-color: #ffffff;
    }
    QMessageBox QLabel {
        color: #212529;
    }
    QMessageBox QPushButton {
        min-width: 80px;
    }
"""

# Tab widget style (fixing contrast issues)
TAB_STYLE = """
    QTabWidget::pane {
        border: 1px solid #dee2e6;
        background-color: white;
        border-radius: 4px;
    }
    QTabBar::tab {
        background-color: #f0f2f5;
        color: #495057;
        border: 1px solid #dee2e6;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 16px;
        margin-right: 2px;
        min-width: 80px;
    }
    QTabBar::tab:selected {
        background-color: white;
        color: #007bff;
        font-weight: bold;
        border-bottom: 2px solid #007bff;
    }
    QTabBar::tab:!selected {
        margin-top: 2px;
    }
"""

# Main view specific styles for the header
MAIN_VIEW_STYLE = """
    QLabel#app_title {
        font-size: 18px;
        font-weight: bold;
        color: #007bff;
    }
    QLabel#app_subtitle {
        font-size: 14px;
        color: #6c757d;
    }
    QWidget#header_container {
        background-color: white;
        border-bottom: 1px solid #dee2e6;
    }
    QWidget#main_container {
        background-color: white;
    }
"""

# Combine styles for a complete application style
COMPLETE_STYLE = (
	BASE_STYLE
	+ BUTTON_STYLE
	+ SECONDARY_BUTTON_STYLE
	+ INPUT_STYLE
	+ TABLE_STYLE
	+ LIST_STYLE
	+ COMBO_STYLE
	+ STATUS_STYLE
	+ PROGRESS_STYLE
	+ MESSAGEBOX_STYLE
	+ TAB_STYLE
	+ MAIN_VIEW_STYLE
)
