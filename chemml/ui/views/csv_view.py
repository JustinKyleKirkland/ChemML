import logging
from typing import List

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QBrush, QIcon
from PyQt5.QtWidgets import (
	QComboBox,
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMenu,
	QMessageBox,
	QPushButton,
	QTableWidget,
	QTableWidgetItem,
	QVBoxLayout,
	QWidget,
	QToolBar,
	QGroupBox,
	QFrame,
	QHeaderView,
)

from chemml.core.chem import detect_smiles_column
from chemml.ui.assets.colors import NON_NUM_COLOR, NAN_COLOR, ERROR_COLOR
from chemml.ui.assets.icons import create_data_icon
from chemml.ui.controllers.csv_controller import CSVController
from chemml.ui.widgets.molecule_widgets import MoleculeTooltip


class CSVView(QWidget):
	"""
	View component for displaying and manipulating CSV data.

	This view provides a UI for:
	- Loading CSV files
	- Displaying tabular data
	- Filtering data
	- Performing data transformations

	Signals:
	    data_ready: Emitted when CSV data is loaded and ready
	"""

	data_ready = pyqtSignal(pd.DataFrame)

	def __init__(self) -> None:
		"""Initialize the CSV view."""
		super().__init__()
		self.logger = logging.getLogger("CSVView")

		# Initialize the controller
		self.controller = CSVController()

		# Set up window properties
		self.set_window_size_limits()

		# Set up UI components
		self.layout = QVBoxLayout()
		self._setup_ui()
		self._connect_signals()

		self.logger.info("CSVView initialized successfully.")

	def set_window_size_limits(self) -> None:
		"""Set the minimum and maximum size limits for the window."""
		self.min_width, self.min_height = 400, 300
		self.max_width, self.max_height = 1200, 800
		self.setMinimumSize(self.min_width, self.min_height)
		self.setMaximumSize(self.max_width, self.max_height)

	def _setup_ui(self) -> None:
		"""Set up the user interface components."""
		# Create UI components
		self.error_label = self._create_error_label()
		self.table_widget = self._create_table_widget()

		# Initialize filter UI components before creating layouts that use them
		self.column_dropdown, self.condition_dropdown, self.value_edit = self._create_filter_widgets()
		self.filter_button = self._create_filter_button()

		# Create data operations toolbar
		self.toolbar = self._create_toolbar()

		# Create filter UI components in a group box for better organization
		filter_group = QGroupBox("Data Filtering")
		filter_group.setLayout(self._create_filter_layout())

		# Create data operations group
		operations_group = QGroupBox("Data Operations")
		operations_layout = QVBoxLayout()
		operations_layout.addWidget(self._create_impute_button())
		operations_group.setLayout(operations_layout)

		# Create bottom panel for controls
		bottom_panel = QFrame()
		bottom_panel.setFrameShape(QFrame.StyledPanel)
		bottom_layout = QHBoxLayout(bottom_panel)
		bottom_layout.addWidget(filter_group, 3)
		bottom_layout.addWidget(operations_group, 1)

		# Set up context menu for table
		self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
		self.table_widget.customContextMenuRequested.connect(self.show_context_menu)

		# Add widgets to layout
		self.layout.addWidget(self.toolbar)
		self.layout.addWidget(self.table_widget)
		self.layout.addWidget(self.error_label)
		self.layout.addWidget(bottom_panel)

		self.setLayout(self.layout)

	def _connect_signals(self) -> None:
		"""Connect all signal handlers."""
		# Add event filter for tooltips
		self.table_widget.setMouseTracking(True)
		self.table_widget.viewport().installEventFilter(self)

	def _create_error_label(self) -> QLabel:
		"""Create a label for displaying error messages."""
		error_label = QLabel()
		error_label.setStyleSheet(f"color: {ERROR_COLOR.name()}")
		return error_label

	def _create_load_button(self) -> QPushButton:
		"""Create a button for loading CSV files."""
		load_button = QPushButton("Load CSV")
		load_button.clicked.connect(self.load_csv)
		return load_button

	def _create_table_widget(self) -> QTableWidget:
		"""Create a table widget for displaying CSV data."""
		table_widget = QTableWidget(self)
		table_widget.setSortingEnabled(True)

		# Enhance table appearance
		table_widget.setAlternatingRowColors(True)
		table_widget.setShowGrid(True)
		table_widget.setGridStyle(Qt.SolidLine)

		# Set selection behavior to select entire rows
		table_widget.setSelectionBehavior(QTableWidget.SelectRows)
		table_widget.setSelectionMode(QTableWidget.ExtendedSelection)

		# Make the table horizontally scrollable
		table_widget.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)

		# Set horizontal header properties for better appearance
		header = table_widget.horizontalHeader()
		header.setSectionResizeMode(QHeaderView.Interactive)
		header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		header.setHighlightSections(True)

		# Set vertical header (row numbers) properties
		v_header = table_widget.verticalHeader()
		v_header.setDefaultSectionSize(28)  # Slightly larger row height

		return table_widget

	def _create_filter_widgets(self) -> tuple:
		"""Create widgets for filtering data."""
		column_dropdown = QComboBox()
		condition_dropdown = QComboBox()
		condition_dropdown.addItems(["Contains", "Equals", "Starts with", "Ends with", ">=", "<=", ">", "<"])
		value_edit = QLineEdit()
		value_edit.setPlaceholderText("Value")
		return column_dropdown, condition_dropdown, value_edit

	def _create_filter_button(self) -> QPushButton:
		"""Create a button for applying filters."""
		filter_button = QPushButton("Apply Filter")
		filter_button.clicked.connect(self.filter_data)
		return filter_button

	def _create_filter_layout(self) -> QHBoxLayout:
		"""Create a layout for the filter UI components with improved styling."""
		filter_layout = QHBoxLayout()
		filter_layout.setSpacing(10)
		filter_layout.setContentsMargins(10, 10, 10, 10)

		# Add label-combo pairs with better spacing
		column_label = QLabel("Filter Column:")
		column_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		column_label.setMinimumWidth(80)
		filter_layout.addWidget(column_label)

		# Style the dropdown
		self.column_dropdown.setMinimumWidth(120)
		self.column_dropdown.setMinimumHeight(30)
		filter_layout.addWidget(self.column_dropdown)

		# Add condition dropdown
		condition_label = QLabel("Condition:")
		condition_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		condition_label.setMinimumWidth(70)
		filter_layout.addWidget(condition_label)

		# Style the condition dropdown
		self.condition_dropdown.setMinimumHeight(30)
		filter_layout.addWidget(self.condition_dropdown)

		# Add value field
		value_label = QLabel("Value:")
		value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		value_label.setMinimumWidth(50)
		filter_layout.addWidget(value_label)

		# Style the value input field
		self.value_edit.setMinimumHeight(30)
		self.value_edit.setMinimumWidth(120)
		filter_layout.addWidget(self.value_edit)

		# Add apply button with better styling
		self.filter_button.setMinimumHeight(30)
		filter_layout.addWidget(self.filter_button)

		return filter_layout

	def _create_impute_button(self) -> QPushButton:
		"""Create a button for imputing missing values."""
		impute_button = QPushButton("Impute Missing Values")
		impute_button.clicked.connect(self.show_impute_options)
		impute_button.setMinimumHeight(30)
		impute_button.setStyleSheet("QPushButton { font-weight: bold; }")
		return impute_button

	def _create_toolbar(self) -> QToolBar:
		"""Create a toolbar with data operation buttons."""
		toolbar = QToolBar()
		toolbar.setIconSize(QSize(24, 24))
		toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

		# Load CSV action
		load_action = toolbar.addAction(create_data_icon(), "Load Data")
		load_action.setToolTip("Load data from a CSV file")
		load_action.triggered.connect(self.load_csv)

		toolbar.addSeparator()

		# Export action
		export_action = toolbar.addAction(QIcon(), "Export")
		export_action.setToolTip("Export current data to a file")
		export_action.triggered.connect(self.export_data)

		# Reset action (clear filters)
		reset_action = toolbar.addAction(QIcon(), "Reset Filters")
		reset_action.setToolTip("Clear all filters and transformations")
		reset_action.triggered.connect(self.reset_filters)

		toolbar.addSeparator()

		# Undo/Redo actions
		undo_action = toolbar.addAction(QIcon(), "Undo")
		undo_action.setToolTip("Undo last operation")
		undo_action.triggered.connect(self.undo)

		redo_action = toolbar.addAction(QIcon(), "Redo")
		redo_action.setToolTip("Redo last undone operation")
		redo_action.triggered.connect(self.redo)

		return toolbar

	def load_csv(self) -> None:
		"""Open a dialog to select and load a CSV file."""
		options = QFileDialog.Options()
		file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)

		if file_name:
			self.logger.info(f"Selected CSV file: {file_name}")
			try:
				df, errors = self.controller.load_csv(file_name)
				self.update_table(df)
				self.update_column_dropdown()
				self.show_errors(errors)
				self.data_ready.emit(df)
				self.logger.info("CSV file loaded successfully.")
			except Exception as e:
				self.show_error_message("Error Loading CSV", str(e))
				self.logger.error(f"Error loading CSV file: {e}")

	def update_table(self, df: pd.DataFrame) -> None:
		"""
		Update the table widget with data from a DataFrame.

		Args:
		    df: DataFrame to display
		"""
		try:
			self.table_widget.setRowCount(len(df))
			self.table_widget.setColumnCount(len(df.columns))
			self.table_widget.setHorizontalHeaderLabels(df.columns)

			# Pre-calculate SMILES columns for efficiency
			smiles_columns = {col: detect_smiles_column(df, col) for col in df.columns}

			# Process data in chunks for better performance
			chunk_size = 1000
			for start_idx in range(0, len(df), chunk_size):
				end_idx = min(start_idx + chunk_size, len(df))
				chunk = df.iloc[start_idx:end_idx]

				for i, (_, row) in enumerate(chunk.iterrows(), start=start_idx):
					for j, value in enumerate(row):
						item = QTableWidgetItem()

						if pd.isna(value):
							item.setBackground(QBrush(NAN_COLOR))
							item.setText("")
						else:
							if isinstance(value, (int, float, np.number)):
								text = str(int(value)) if isinstance(value, (int, np.integer)) else f"{value:.6g}"
								item.setData(Qt.DisplayRole, float(value))
							else:
								text = str(value)
								if not smiles_columns[df.columns[j]]:
									item.setBackground(QBrush(NON_NUM_COLOR))
							item.setText(text)

						self.table_widget.setItem(i, j, item)

			# Optimize column widths
			self.table_widget.resizeColumnsToContents()
			self.logger.info("Table updated successfully.")

		except Exception as e:
			self.show_error_message("Update Error", str(e))
			self.logger.error(f"Error updating table: {str(e)}")

	def update_column_dropdown(self) -> None:
		"""Update the column dropdown with current DataFrame columns."""
		self.column_dropdown.clear()
		self.column_dropdown.addItems(self.controller.df.columns)

	def filter_data(self) -> None:
		"""Apply a filter to the data based on the filter UI components."""
		column_name = self.column_dropdown.currentText()
		condition = self.condition_dropdown.currentText()
		value = self.value_edit.text()

		if not all([column_name, condition, value]):
			self.show_error_message("Invalid Filter", "Please select a column, condition, and value.")
			return

		try:
			filtered_df = self.controller.filter_data(column_name, condition, value)
			if filtered_df is not None:
				self.update_table(filtered_df)
				self.data_ready.emit(filtered_df)
				self.logger.info("Filter applied successfully.")
			else:
				self.show_error_message("Filter Error", "No rows match the filter criteria.")
		except Exception as e:
			self.show_error_message("Error Applying Filter", str(e))
			self.logger.error(f"Error applying filter: {e}")

	def show_impute_options(self) -> None:
		"""Show dialog with options for imputing missing values."""
		msg_box = QMessageBox()
		msg_box.setWindowTitle("Impute Missing Values")
		msg_box.setText("Choose a strategy to impute all NaN values:")

		mean_button = msg_box.addButton("Impute with Mean", QMessageBox.ActionRole)
		median_button = msg_box.addButton("Impute with Median", QMessageBox.ActionRole)
		knn_button = msg_box.addButton("Impute with KNN", QMessageBox.ActionRole)
		mice_button = msg_box.addButton("Impute with MICE", QMessageBox.ActionRole)

		msg_box.exec_()

		clicked_button = msg_box.clickedButton()
		strategy = None

		if clicked_button == mean_button:
			strategy = "mean"
		elif clicked_button == median_button:
			strategy = "median"
		elif clicked_button == knn_button:
			strategy = "knn"
		elif clicked_button == mice_button:
			strategy = "mice"

		if strategy:
			self.impute_all_missing_values(strategy)

	def impute_all_missing_values(self, strategy: str) -> None:
		"""
		Impute missing values in all columns.

		Args:
		    strategy: Imputation strategy to use
		"""
		try:
			df = self.controller.impute_all_missing_values(strategy)
			self.update_table(df)
			self.data_ready.emit(df)
			self.error_label.setText("")
			self.logger.info(f"Missing values imputed using {strategy} strategy.")
		except Exception as e:
			self.show_error_message("Imputation Error", str(e))
			self.logger.error(f"Error imputing missing values: {e}")

	def show_context_menu(self, pos) -> None:
		"""
		Show context menu for the table widget.

		Args:
		    pos: Position where right-click occurred
		"""
		item = self.table_widget.itemAt(pos)

		if not item:
			return

		column_index = item.column()
		column_name = self.controller.df.columns[column_index]

		context_menu = QMenu(self)

		# Add basic operations
		context_menu.addAction("Undo", self.undo)
		context_menu.addAction("Redo", self.redo)

		# Add encoding options for categorical columns
		if self.controller.df[column_name].dtype == "object":
			encoding_menu = QMenu("One-hot encode", self)
			encoding_menu.addAction("Create N distinct columns", lambda: self.apply_one_hot_encoding(column_name, True))
			encoding_menu.addAction(
				"Create N-1 distinct columns", lambda: self.apply_one_hot_encoding(column_name, False)
			)
			context_menu.addMenu(encoding_menu)

		# Add imputation options for columns with missing values
		if self.controller.df[column_name].isnull().any():
			impute_menu = QMenu("Impute Missing Values", self)
			impute_menu.addAction("Impute with mean", lambda: self.impute_column(column_name, "mean"))
			impute_menu.addAction("Impute with median", lambda: self.impute_column(column_name, "median"))
			impute_menu.addAction("Impute with KNN", lambda: self.impute_column(column_name, "knn"))
			context_menu.addMenu(impute_menu)

		# Add RDKit options for SMILES columns
		if detect_smiles_column(self.controller.df, column_name):
			rdkit_menu = QMenu("RDKit Operations", self)
			rdkit_menu.addAction("Convert to Canonical SMILES", lambda: self.canonicalize_smiles(column_name))

			# Add descriptors submenu
			descriptors_menu = QMenu("Add Descriptors", self)
			descriptors_menu.addAction(
				"Add Basic Properties (MW, LogP, TPSA)", lambda: self.add_descriptors(column_name, "basic")
			)
			rdkit_menu.addMenu(descriptors_menu)

			# Add fingerprints submenu
			fp_menu = QMenu("Add Fingerprints", self)
			fp_menu.addAction("Morgan (ECFP4)", lambda: self.add_fingerprints(column_name, "morgan"))
			fp_menu.addAction("MACCS Keys", lambda: self.add_fingerprints(column_name, "maccs"))
			rdkit_menu.addMenu(fp_menu)

			context_menu.addMenu(rdkit_menu)

		# Show the menu at cursor position
		context_menu.exec_(self.table_widget.viewport().mapToGlobal(pos))

	def undo(self) -> None:
		"""Undo the last operation."""
		df = self.controller.undo()
		if df is not None:
			self.update_table(df)
			self.update_column_dropdown()
			self.data_ready.emit(df)
			self.logger.info("Undo operation completed.")

	def redo(self) -> None:
		"""Redo the last undone operation."""
		df = self.controller.redo()
		if df is not None:
			self.update_table(df)
			self.update_column_dropdown()
			self.data_ready.emit(df)
			self.logger.info("Redo operation completed.")

	def apply_one_hot_encoding(self, column_name: str, n_distinct: bool) -> None:
		"""
		Apply one-hot encoding to a column.

		Args:
		    column_name: Column to encode
		    n_distinct: Whether to create N or N-1 columns
		"""
		try:
			df = self.controller.apply_one_hot_encoding(column_name, n_distinct)
			self.update_table(df)
			self.update_column_dropdown()
			self.data_ready.emit(df)
			self.logger.info(f"One-hot encoding applied to column: {column_name}")
		except Exception as e:
			self.show_error_message("One-Hot Encoding Error", str(e))
			self.logger.error(f"Error applying one-hot encoding: {e}")

	def impute_column(self, column_name: str, strategy: str) -> None:
		"""
		Impute missing values in a specific column.

		Args:
		    column_name: Column to impute
		    strategy: Imputation strategy to use
		"""
		try:
			df = self.controller.impute_missing_values(column_name, strategy)
			self.update_table(df)
			self.data_ready.emit(df)
			self.logger.info(f"Imputed missing values in column {column_name} using {strategy} strategy.")
		except Exception as e:
			self.show_error_message("Imputation Error", str(e))
			self.logger.error(f"Error imputing column {column_name}: {e}")

	def canonicalize_smiles(self, column_name: str) -> None:
		"""
		Convert SMILES strings to canonical form.

		Args:
		    column_name: Column containing SMILES strings
		"""
		try:
			df = self.controller.canonicalize_smiles_column(column_name)
			self.update_table(df)
			self.data_ready.emit(df)
			self.logger.info(f"Canonicalized SMILES in column: {column_name}")
		except Exception as e:
			self.show_error_message("SMILES Canonicalization Error", str(e))
			self.logger.error(f"Error canonicalizing SMILES: {e}")

	def add_descriptors(self, column_name: str, descriptor_set: str) -> None:
		"""
		Add molecular descriptors based on SMILES strings.

		Args:
		    column_name: Column containing SMILES strings
		    descriptor_set: Type of descriptors to add
		"""
		try:
			df = self.controller.add_molecular_descriptors(column_name, descriptor_set)
			self.update_table(df)
			self.update_column_dropdown()
			self.data_ready.emit(df)
			self.logger.info(f"Added {descriptor_set} descriptors to column: {column_name}")
		except Exception as e:
			self.show_error_message("Descriptor Calculation Error", str(e))
			self.logger.error(f"Error adding descriptors: {e}")

	def add_fingerprints(self, column_name: str, fp_type: str) -> None:
		"""
		Add molecular fingerprints based on SMILES strings.

		Args:
		    column_name: Column containing SMILES strings
		    fp_type: Type of fingerprints to add
		"""
		try:
			df = self.controller.add_fingerprints(column_name, fp_type)
			self.update_table(df)
			self.update_column_dropdown()
			self.data_ready.emit(df)
			self.logger.info(f"Added {fp_type} fingerprints to column: {column_name}")
		except Exception as e:
			self.show_error_message("Fingerprint Generation Error", str(e))
			self.logger.error(f"Error adding fingerprints: {e}")

	def show_errors(self, errors: List[str]) -> None:
		"""
		Display validation errors.

		Args:
		    errors: List of error messages
		"""
		if errors:
			self.error_label.setText("\n".join(errors))
		else:
			self.error_label.setText("")

	def show_error_message(self, title: str, message: str) -> None:
		"""
		Display an error message dialog.

		Args:
		    title: Dialog title
		    message: Error message
		"""
		msg_box = QMessageBox(self)
		msg_box.setIcon(QMessageBox.Critical)
		msg_box.setWindowTitle(title)
		msg_box.setText(message)
		msg_box.setStandardButtons(QMessageBox.Ok)
		msg_box.exec_()

	def export_data(self) -> None:
		"""Export current data to a CSV file."""
		if self.controller.df.empty:
			self.show_error_message("Export Error", "No data to export.")
			return

		options = QFileDialog.Options()
		file_name, _ = QFileDialog.getSaveFileName(
			self, "Export Data", "", "CSV Files (*.csv);;All Files (*)", options=options
		)

		if file_name:
			try:
				# Ensure file has .csv extension
				if not file_name.endswith(".csv"):
					file_name += ".csv"

				self.controller.df.to_csv(file_name, index=False)
				self.logger.info(f"Data exported to {file_name}")

				# Show success message
				QMessageBox.information(
					self, "Export Successful", f"Data successfully exported to {file_name}", QMessageBox.Ok
				)
			except Exception as e:
				self.show_error_message("Export Error", str(e))
				self.logger.error(f"Error exporting data: {e}")

	def reset_filters(self) -> None:
		"""Reset all filters and show original data."""
		if not hasattr(self.controller, "original_df") or self.controller.original_df is None:
			return

		try:
			self.controller.reset_to_original()
			df = self.controller.df
			self.update_table(df)
			self.update_column_dropdown()
			self.data_ready.emit(df)
			self.error_label.setText("")
			self.logger.info("Filters reset, showing original data.")
		except Exception as e:
			self.show_error_message("Reset Error", str(e))
			self.logger.error(f"Error resetting filters: {e}")

	def eventFilter(self, source, event) -> bool:
		"""
		Handle mouse events for molecule tooltips.

		Args:
		    source: Event source
		    event: Event to filter

		Returns:
		    True if event was handled, False otherwise
		"""
		if source is self.table_widget.viewport() and event.type() == event.MouseMove:
			# Get item under cursor
			pos = event.pos()
			item = self.table_widget.itemAt(pos)

			if item is not None:
				try:
					column_name = self.controller.df.columns[item.column()]

					# Check if this is a SMILES column
					if detect_smiles_column(self.controller.df, column_name):
						smiles = item.text()
						if smiles and pd.notna(smiles):
							# Create or update tooltip
							if not hasattr(self, "molecule_tooltip"):
								self.molecule_tooltip = MoleculeTooltip()

							self.molecule_tooltip.show_molecule(smiles, self.table_widget.viewport().mapToGlobal(pos))
							return True

				except Exception as e:
					self.logger.debug(f"Error in tooltip generation: {str(e)}")
					if hasattr(self, "molecule_tooltip"):
						self.molecule_tooltip.hide()

			# Hide tooltip when not over a valid SMILES item
			if hasattr(self, "molecule_tooltip"):
				self.molecule_tooltip.hide()

		elif event.type() == event.Leave:
			# Hide tooltip when mouse leaves the widget
			if hasattr(self, "molecule_tooltip"):
				self.molecule_tooltip.hide()

		return super().eventFilter(source, event)
