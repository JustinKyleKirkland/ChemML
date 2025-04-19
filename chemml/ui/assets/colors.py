"""
ChemML Color Definitions
=======================

This module defines color constants used throughout the ChemML application
to ensure a consistent visual appearance.
"""

from PyQt5.QtGui import QColor

# Application color palette - bootstrap-inspired
COLORS = {
	"primary": "#007bff",  # Primary blue
	"secondary": "#6c757d",  # Gray
	"success": "#28a745",  # Green
	"danger": "#dc3545",  # Red
	"warning": "#ffc107",  # Yellow
	"info": "#17a2b8",  # Teal
	"light": "#f8f9fa",  # Light gray
	"dark": "#343a40",  # Dark gray
	"white": "#ffffff",
	"transparent": "transparent",
	# Additional application-specific colors
	"plot_background": "#ffffff",
	"plot_grid": "#e9ecef",
	"highlight": "#80bdff",
	"molecule_carbon": "#808080",
	"molecule_hydrogen": "#FFFFFF",
	"molecule_oxygen": "#FF0000",
	"molecule_nitrogen": "#0000FF",
	"molecule_sulfur": "#FFFF00",
	"molecule_phosphorus": "#FF8000",
	"molecule_bond": "#000000",
}

# Named gradients for plots and visualizations
GRADIENTS = {
	"blue_gradient": ["#cfe2ff", "#9ec5fe", "#6ea8fe", "#3d8bfd", "#0d6efd"],
	"green_gradient": ["#d1e7dd", "#a3cfbb", "#75c799", "#479f76", "#198754"],
	"purple_gradient": ["#e2d9f3", "#c5b3e6", "#a98eda", "#8c68cd", "#6f42c1"],
	"spectral": ["#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#e6f598", "#abdda4", "#66c2a5", "#3288bd"],
}

# Theme configuration
LIGHT_THEME = {
	"background": COLORS["light"],
	"foreground": COLORS["dark"],
	"accent": COLORS["primary"],
	"panel_bg": COLORS["white"],
	"panel_border": "#dee2e6",
	"header_bg": "#f1f3f5",
	"table_alternate_bg": "#f8f9fa",
}

DARK_THEME = {
	"background": "#212529",
	"foreground": "#f8f9fa",
	"accent": "#0d6efd",
	"panel_bg": "#343a40",
	"panel_border": "#495057",
	"header_bg": "#343a40",
	"table_alternate_bg": "#2b3035",
}

# Current theme (default to dark)
CURRENT_THEME = DARK_THEME

# Define color constants
NON_NUM_COLOR = QColor(0, 100, 100)  # Cyan color for non-numeric values
NAN_COLOR = QColor(100, 0, 100)  # Magenta color for NaN values
BUTTON_COLOR = QColor(0, 122, 255)  # Blue color for buttons
ERROR_COLOR = QColor(220, 53, 69)  # Red color for errors
