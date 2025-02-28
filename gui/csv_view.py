import io
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QFontMetrics, QPixmap
from PyQt5.QtWidgets import (
	QComboBox,
	QDialog,
	QDialogButtonBox,
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QListWidget,
	QListWidgetItem,
	QMenu,
	QMessageBox,
	QPushButton,
	QTableWidget,
	QTableWidgetItem,
	QTabWidget,
	QVBoxLayout,
	QWidget,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

from utils.color_assets import ERROR_COLOR, NAN_COLOR, NON_NUM_COLOR
from utils.data_utils import impute_values, one_hot_encode, validate_csv


# Cache molecule objects to avoid repeated parsing
@lru_cache(maxsize=1024)
def get_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
	"""Cache and return RDKit molecule objects."""
	if pd.isna(smiles):
		return None
	return Chem.MolFromSmiles(smiles)


class MoleculeTooltip(QLabel):
	"""Custom tooltip widget for displaying molecule images."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowFlags(Qt.ToolTip)
		self.setAttribute(Qt.WA_TranslucentBackground)
		self.setStyleSheet("QLabel { background-color: white; padding: 2px; border: 1px solid gray; }")
		self._pixmap_cache: Dict[str, QPixmap] = {}

	def show_molecule(self, smiles: str, pos: QPoint) -> None:
		"""Display molecule image at given position with caching."""
		if smiles in self._pixmap_cache:
			pixmap = self._pixmap_cache[smiles]
		else:
			mol = get_mol_from_smiles(smiles)
			if not mol:
				return

			img = Draw.MolToImage(mol)
			with io.BytesIO() as bio:
				img.save(bio, format="PNG")
				pixmap = QPixmap()
				pixmap.loadFromData(bio.getvalue())

			if pixmap.width() > 300:
				pixmap = pixmap.scaledToWidth(300)
			self._pixmap_cache[smiles] = pixmap

		self.setPixmap(pixmap)
		self.adjustSize()
		self.move(pos)
		self.show()


class CSVView(QWidget):
	"""
	A widget for viewing and manipulating CSV data.

	Signals:
	    data_ready: A signal emitted when the CSV data is ready to be processed.

	Attributes:
	    tab_widget (QTabWidget): The parent tab widget.
	    logger (logging.Logger): The logger for CSVView.
	    df (pd.DataFrame): The DataFrame containing the CSV data.
	    undo_stack (List[pd.DataFrame]): The stack for storing previous DataFrame states for undo operation.
	    redo_stack (List[pd.DataFrame]): The stack for storing undone DataFrame states for redo operation.
	    layout (QVBoxLayout): The layout for the widget.
	    error_label (QLabel): The label for displaying error messages.
	    load_button (QPushButton): The button for loading a CSV file.
	    column_dropdown (QComboBox): The dropdown for selecting a column for filtering.
	    condition_dropdown (QComboBox): The dropdown for selecting a condition for filtering.
	    value_edit (QLineEdit): The text field for entering a value for filtering.
	    filter_button (QPushButton): The button for applying a filter.
	    table_widget (QTableWidget): The table widget for displaying the CSV data.

	Methods:
	    create_error_label() -> QLabel:

	    create_load_button() -> QPushButton:

	    create_filter_widgets() -> Tuple[QComboBox, QComboBox, QLineEdit]:

	    create_filter_button() -> QPushButton:

	    create_table_widget() -> QTableWidget:

	    create_filter_layout() -> QHBoxLayout:

	    load_csv() -> None:

	    display_csv_image(csv_file: str) -> None:

	    undo() -> None:

	    redo() -> None:

	    apply_one_hot_encoding(column_name: str, n_distinct: bool = True) -> None:

	    impute_missing_values(column_name: str, strategy: str) -> None:

	    create_rdkit_menu(self, column_name: str) -> QMenu:

	    canonicalize_smiles(self, column_name: str) -> None:

	    show_descriptor_selector(self, column_name: str) -> None:

	    eventFilter(self, source, event) -> bool:

	    _contains_valid_smiles(self, column_name: str) -> bool:
	"""

	data_ready = pyqtSignal(pd.DataFrame)

	def __init__(self, tab_widget: QTabWidget) -> None:
		super().__init__()
		self.tab_widget = tab_widget
		self.logger: logging.Logger = logging.getLogger("CSVView")
		self.df: pd.DataFrame = pd.DataFrame()
		self.undo_stack: List[pd.DataFrame] = []
		self.redo_stack: List[pd.DataFrame] = []

		self.set_window_size_limits()
		self.layout = QVBoxLayout()

		self.error_label = self.create_error_label()
		self.load_button = self.create_load_button()
		self.column_dropdown, self.condition_dropdown, self.value_edit = self.create_filter_widgets()
		self.filter_button = self.create_filter_button()
		self.table_widget = self.create_table_widget()

		self.impute_button = self.create_impute_button()

		self.table_widget.customContextMenuRequested.connect(self.show_context_menu)
		self.setup_context_menu()

		self.layout.addWidget(self.table_widget)
		self.layout.addWidget(self.error_label)
		self.layout.addWidget(self.load_button)
		self.layout.addLayout(self.create_filter_layout())
		self.layout.addWidget(self.impute_button)
		self.setLayout(self.layout)

		logging.info("CSVView initialized successfully.")

	def create_impute_button(self) -> QPushButton:
		"""
		Creates a QPushButton widget for imputing missing values.

		Returns:
		    QPushButton: The created impute button widget.
		"""
		logging.debug("Creating impute button widget.")
		impute_button = QPushButton("Impute Missing Values")
		impute_button.clicked.connect(self.show_impute_options)
		return impute_button

	def show_impute_options(self) -> None:
		"""
		Displays a message box with options to impute missing values.

		Returns:
		    None
		"""
		logging.debug("Showing impute options.")

		# Create the message box for options
		msg_box = QMessageBox()
		msg_box.setWindowTitle("Impute Missing Values")
		msg_box.setText("Choose a strategy to impute all NaN values:")

		# Add buttons for mean and median
		mean_button = msg_box.addButton("Impute with Mean", QMessageBox.ActionRole)
		median_button = msg_box.addButton("Impute with Median", QMessageBox.ActionRole)
		knn_button = msg_box.addButton("Impute with KNN", QMessageBox.ActionRole)
		mice_button = msg_box.addButton("Impute with MICE", QMessageBox.ActionRole)

		# Show the message box and get the user's selection
		msg_box.exec_()

		# Determine which button was clicked and call the appropriate method
		if msg_box.clickedButton() == mean_button:
			self.impute_missing_values_all("mean")
		elif msg_box.clickedButton() == median_button:
			self.impute_missing_values_all("median")
		elif msg_box.clickedButton() == knn_button:
			self.impute_missing_values_all("knn")
		elif msg_box.clickedButton() == mice_button:
			self.impute_missing_values_all("mice")

	def impute_missing_values_all(self, strategy: str) -> None:
		"""
		Impute missing values in all columns with missing values.

		Args:
		    strategy: Imputation strategy ('mean', 'median', 'knn', or 'mice')
		"""
		try:
			# First check if all columns with missing values are numeric
			columns_with_missing = [col for col in self.df.columns if self.df[col].isnull().any()]
			non_numeric_cols = [col for col in columns_with_missing if not pd.api.types.is_numeric_dtype(self.df[col])]

			if non_numeric_cols:
				raise ValueError(f"The following columns contain non-numerical values: {', '.join(non_numeric_cols)}")

			self.undo_stack.append(self.df.copy())
			for column in columns_with_missing:
				self.df = impute_values(self.df, column, strategy)

			# Clear error messages if imputation was successful
			self.error_label.setText("")
			self.update_table(self.df)

		except ValueError as e:
			self.show_error("Imputation Error", str(e))
			self.logger.error(f"Error in imputation: {str(e)}")

	def create_error_label(self) -> QLabel:
		"""
		Creates and returns an error label widget.

		Returns:
		    QLabel: The created error label widget.
		"""
		logging.debug("Creating error label widget.")
		error_label = QLabel()
		error_label.setStyleSheet(f"color: {ERROR_COLOR.name()}")
		return error_label

	def create_load_button(self) -> QPushButton:
		"""
		Creates a QPushButton widget for loading a CSV file.

		Returns:
		    QPushButton: The created load button widget.
		"""
		logging.debug("Creating load button widget.")
		load_button = QPushButton("Load CSV")
		load_button.clicked.connect(self.load_csv)
		return load_button

	def create_filter_widgets(self) -> Tuple[QComboBox, QComboBox, QLineEdit]:
		"""
		Creates filter widgets.

		Returns:
		    Tuple[QComboBox, QComboBox, QLineEdit]: A tuple containing the column dropdown, condition dropdown,
		    and value edit widgets.
		"""
		logging.debug("Creating filter widgets.")
		column_dropdown = QComboBox()
		condition_dropdown = QComboBox()
		condition_dropdown.addItems(["Contains", "Equals", "Starts with", "Ends with", ">=", "<=", ">", "<"])
		value_edit = QLineEdit()
		value_edit.setPlaceholderText("Value")
		return column_dropdown, condition_dropdown, value_edit

	def create_filter_button(self) -> QPushButton:
		"""
		Creates a QPushButton widget for applying a filter.

		Returns:
		    QPushButton: The created filter button widget.
		"""
		logging.debug("Creating filter button widget.")
		filter_button = QPushButton("Apply Filter")
		filter_button.clicked.connect(self.filter_data)
		return filter_button

	def create_table_widget(self) -> QTableWidget:
		"""
		Creates and returns a QTableWidget object.

		Returns:
		    QTableWidget: The created QTableWidget object.
		"""
		logging.debug("Creating table widget.")
		table_widget = QTableWidget(self)
		table_widget.setSortingEnabled(True)

		# Add event filter for tooltips
		table_widget.setMouseTracking(True)
		table_widget.viewport().installEventFilter(self)

		return table_widget

	def eventFilter(self, source, event) -> bool:
		"""
		Filter events to show molecule tooltips on hover.

		Args:
		    source: The source widget of the event
		    event: The event to filter

		Returns:
		    bool: True if the event was handled, False otherwise
		"""
		if source is self.table_widget.viewport() and event.type() == event.MouseMove:
			# Get item under cursor
			pos = event.pos()
			item = self.table_widget.itemAt(pos)

			if item is not None:
				try:
					column_name = self.df.columns[item.column()]
					logging.debug(f"Mouse over column: {column_name}")

					# Check if this is a SMILES column (contains valid SMILES strings)
					if self.df[column_name].dtype == "object":
						is_smiles = self._contains_valid_smiles(column_name)
						logging.debug(f"Column {column_name} contains valid SMILES: {is_smiles}")

						if is_smiles:
							smiles = item.text()
							if smiles and pd.notna(smiles):
								logging.debug(f"Processing SMILES: {smiles}")

								# Generate molecule image
								mol = Chem.MolFromSmiles(smiles)
								if mol:
									img = Draw.MolToImage(mol)

									# Convert PIL image to QPixmap
									with io.BytesIO() as bio:
										img.save(bio, format="PNG")
										pixmap = QPixmap()
										pixmap.loadFromData(bio.getvalue())

									# Scale pixmap if too large
									if pixmap.width() > 300:
										pixmap = pixmap.scaledToWidth(300)

									# Create or update tooltip
									if not hasattr(self, "molecule_tooltip"):
										self.molecule_tooltip = MoleculeTooltip()

									self.molecule_tooltip.show_molecule(smiles, pos)
									return True
								else:
									logging.debug(f"Failed to create molecule from SMILES: {smiles}")
							else:
								logging.debug("Empty or NaN SMILES value")

				except Exception as e:
					logging.debug(f"Error in tooltip generation: {str(e)}")
					if hasattr(self, "molecule_tooltip"):
						self.molecule_tooltip.hide()

			else:
				# Hide tooltip when not over an item
				if hasattr(self, "molecule_tooltip"):
					self.molecule_tooltip.hide()

		elif event.type() == event.Leave:
			# Hide tooltip when mouse leaves the widget
			if hasattr(self, "molecule_tooltip"):
				self.molecule_tooltip.hide()

		return super().eventFilter(source, event)

	def _contains_valid_smiles(self, column_name: str) -> bool:
		"""
		Check if a column contains valid SMILES strings.

		Args:
		    column_name: Name of the column to check

		Returns:
		    bool: True if the column contains valid SMILES strings
		"""
		try:
			# Check first few non-null values
			sample = self.df[column_name].dropna().head()
			if len(sample) == 0:
				return False

			valid_count = sum(1 for smiles in sample if Chem.MolFromSmiles(smiles) is not None)
			# Consider it a SMILES column if at least 80% of samples are valid
			is_valid = (valid_count / len(sample)) >= 0.8

			logging.debug(f"SMILES validation for {column_name}: {valid_count}/{len(sample)} valid structures")
			return is_valid
		except Exception as e:
			logging.debug(f"Error validating SMILES in column {column_name}: {str(e)}")
			return False

	def create_filter_layout(self) -> QHBoxLayout:
		"""
		Creates and returns a QHBoxLayout for the filter layout.

		Returns:
		    QHBoxLayout: The filter layout containing the filter column dropdown, condition dropdown,
		                 value edit, and filter button.
		"""
		pass
		logging.debug("Creating filter layout.")
		filter_layout = QHBoxLayout()
		filter_layout.addWidget(QLabel("Filter Column:"))
		filter_layout.addWidget(self.column_dropdown)
		filter_layout.addWidget(QLabel("Condition:"))
		filter_layout.addWidget(self.condition_dropdown)
		filter_layout.addWidget(QLabel("Value:"))
		filter_layout.addWidget(self.value_edit)
		filter_layout.addWidget(self.filter_button)
		return filter_layout

	def load_csv(self) -> None:
		"""
		Loads a CSV file and displays its contents.

		Returns:
		    None
		"""
		logging.info("Load CSV button clicked.")
		options = QFileDialog.Options()
		file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
		if file_name:
			logging.info(f"Selected CSV file: {file_name}")
			self.display_csv_image(file_name)
			self.data_ready.emit(self.df)
		else:
			logging.info("No file selected")

	def display_csv_image(self, csv_file: str) -> None:
		"""
		Display the CSV file as an image in the GUI.

		Args:
		    csv_file (str): The path to the CSV file.

		Returns:
		    None
		"""
		try:
			logging.info(f"Reading CSV file: {csv_file}")
			self.df = pd.read_csv(csv_file)
			logging.info(f"CSV file '{csv_file}' read successfully.")

			int_columns = self.df.select_dtypes(include="int").columns
			if not int_columns.empty:
				logging.info(f"Converting integer columns to nullable integers: {int_columns}")
				self.df[int_columns] = self.df[int_columns].astype(float)

			errors = validate_csv(self.df)
			self.show_errors(errors)

			self.data_ready.emit(self.df)

			self.column_dropdown.clear()
			self.column_dropdown.addItems(self.df.columns)

			self.update_table(self.df)

			logging.info("CSV data displayed in the table.")
			self.adjust_window_size()
		except Exception as e:
			self.show_error_message("Error loading CSV file", str(e))
			logging.error(f"Error loading CSV file '{csv_file}': {e}")

	def undo(self) -> None:
		"""
		Undo the last operation.

		This method pops the last operation from the undo stack and applies it to the dataframe.
		If the undo stack is empty, nothing is done.

		Returns:
		    None
		"""
		logging.info("Undo operation triggered.")
		if self.undo_stack:
			self.redo_stack.append(self.df.copy())
			self.df = self.undo_stack.pop()
			self.update_table(self.df)
			self.update_column_dropdown()
			self.data_ready.emit(self.df)
			self.show_errors(validate_csv(self.df))
			logging.info("Undo operation completed.")
		else:
			logging.info("Undo stack is empty; nothing to undo.")

	def redo(self) -> None:
		"""
		Redoes the last undone operation.

		This method pops the last operation from the redo stack and applies it to the data frame.
		If the redo stack is empty, nothing is done.

		Returns:
		    None
		"""
		logging.info("Redo operation triggered.")
		if self.redo_stack:
			self.undo_stack.append(self.df.copy())
			self.df = self.redo_stack.pop()
			self.update_table(self.df)
			self.update_column_dropdown()
			self.data_ready.emit(self.df)
			self.show_errors(validate_csv(self.df))
			logging.info("Redo operation completed.")
		else:
			logging.info("Redo stack is empty; nothing to redo.")

	def apply_one_hot_encoding(self, column_name: str, n_distinct: bool = True) -> None:
		"""
		Apply one-hot encoding to a specified column in the DataFrame.

		Args:
		    column_name (str): The name of the column to apply one-hot encoding to.
		    n_distinct (bool, optional): Whether to use the 'n_distinct' parameter in the one-hot encoding function.
		        Defaults to True.

		Returns:
		    None

		Raises:
		    ValueError: If an error occurs during the one-hot encoding process.

		"""
		logging.info(f"Applying one-hot encoding to column: {column_name}")
		try:
			self.undo_stack.append(self.df.copy())
			self.redo_stack.clear()

			self.df = one_hot_encode(self.df, column_name, n_distinct)
			self.update_table(self.df)
			self.update_column_dropdown()

			self.data_ready.emit(self.df)

			new_columns_with_nulls = [col for col in self.df.columns if self.df[col].isnull().any()]
			if new_columns_with_nulls:
				warning_msg = (
					f"Warning: The following columns contain null values: {', '.join(new_columns_with_nulls)}."
				)
				self.show_errors([warning_msg])
				logging.warning(warning_msg)
			else:
				logging.info(f"One-hot encoding applied to column: {column_name}")
		except ValueError as e:
			self.show_error_message("One-Hot Encoding Error", str(e))
			logging.error(f"Error in one-hot encoding: {e}")

	def impute_missing_values(self, column_name: str, strategy: str) -> None:
		"""
		Impute missing values in a specified column.

		Args:
		    column_name: Name of the column to impute
		    strategy: Imputation strategy ('mean', 'median', 'knn', or 'mice')
		"""
		try:
			if column_name not in self.df.columns:
				raise ValueError(f"Column '{column_name}' not found in DataFrame")

			if not pd.api.types.is_numeric_dtype(self.df[column_name]):
				raise ValueError(f"Column '{column_name}' contains non-numerical values")

			self.undo_stack.append(self.df.copy())
			self.df = impute_values(self.df, column_name, strategy)

			# Clear error messages if imputation was successful
			self.error_label.setText("")
			self.update_table(self.df)

		except ValueError as e:
			self.show_error("Imputation Error", str(e))
			self.logger.error(f"Error in imputation: {str(e)}")

	def show_errors(self, errors: List[str]) -> None:
		"""
		Display validation errors in the CSV data.

		Args:
		    errors (List[str]): A list of validation errors.

		Returns:
		    None
		"""
		if errors:
			logging.warning("Validation errors found in CSV data.")
			self.error_label.setText("\n".join(errors))
		else:
			self.error_label.setText("")
			logging.info("No validation errors found.")

	def adjust_window_size(self) -> None:
		"""
		Adjusts the window size based on the content of the table.

		This method calculates the required height and width of the window based on the number of rows and columns
		in the table.
		It takes into account the height of each row, the width of each column,
		and the maximum allowed height and width.
		The window is then resized to the calculated dimensions.

		Raises:
		    Exception: If there is an error while adjusting the window size.

		Returns:
		    None
		"""
		try:
			logging.debug("Adjusting window size based on table content.")
			row_count = self.table_widget.rowCount()
			column_count = self.table_widget.columnCount()

			if column_count == 0 or row_count == 0:
				logging.warning("No data to adjust window size.")
				return

			row_height = max(self.table_widget.verticalHeader().sectionSize(i) for i in range(row_count)) + 10

			font_metrics = QFontMetrics(self.font())
			total_column_width = sum(
				max(
					font_metrics.width(str(self.table_widget.item(row_index, col).text()))
					for row_index in range(row_count)
				)
				for col in range(column_count)
			) + (column_count * 75)

			new_height = min(self.max_height, row_count * row_height + 10)
			new_width = min(self.max_width, total_column_width)

			self.resize(max(self.min_width, new_width), max(self.min_height, new_height))
			logging.info("Window size adjusted successfully.")
		except Exception as e:
			self.show_error_message("Error", f"Failed to adjust window size: {e}")
			logging.error(f"Failed to adjust window size: {e}")

	def create_impute_menu(self, column_name: str) -> QMenu:
		"""
		Creates a QMenu for imputing missing values in a specific column.

		Args:
		    column_name (str): The name of the column to impute missing values for.

		Returns:
		    QMenu: The created QMenu for imputing missing values.
		"""
		logging.debug(f"Creating impute menu for column: {column_name}.")
		impute_menu = QMenu("Impute Missing Values", self)
		impute_menu.addAction("Impute with mean", lambda: self.impute_missing_values(column_name, "mean"))
		impute_menu.addAction(
			"Impute with median",
			lambda: self.impute_missing_values(column_name, "median"),
		)
		impute_menu.addAction("Impute with KNN", lambda: self.impute_missing_values(column_name, "knn"))
		return impute_menu

	def create_one_hot_menu(self, column_name: str) -> QMenu:
		"""
		Creates a one-hot encode menu for the specified column.

		Args:
		    column_name (str): The name of the column to be one-hot encoded.

		Returns:
		    QMenu: The created one-hot encode menu.
		"""
		logging.debug(f"Creating one-hot encode menu for column: {column_name}.")
		one_hot_menu = QMenu(f"One-hot encode '{column_name}'", self)
		one_hot_menu.addAction(
			"Create N distinct columns",
			lambda: self.apply_one_hot_encoding(column_name, n_distinct=True),
		)
		one_hot_menu.addAction(
			"Create N-1 distinct columns",
			lambda: self.apply_one_hot_encoding(column_name, n_distinct=False),
		)
		return one_hot_menu

	def update_table(self, df: pd.DataFrame) -> None:
		"""Update the table widget with new DataFrame data efficiently."""
		try:
			self.table_widget.setRowCount(len(df))
			self.table_widget.setColumnCount(len(df.columns))
			self.table_widget.setHorizontalHeaderLabels(df.columns)

			# Pre-calculate SMILES columns for efficiency
			smiles_columns = {col: self._contains_valid_smiles(col) for col in df.columns}

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

		except Exception as e:
			self.show_error("Update Error", str(e))
			logging.error(f"Error updating table: {str(e)}")

	def filter_data(self) -> None:
		"""Apply a filter to the data based on the selected column, condition, and value."""
		column_name = self.column_dropdown.currentText()
		condition = self.condition_dropdown.currentText()
		value = self.value_edit.text()

		if not all([column_name, condition, value]):
			logging.warning("Filter criteria missing. Please select a column, condition, and value.")
			self.show_error_message("Invalid Filter", "Please select a column, condition, and value.")
			return

		try:
			logging.info(f"Applying filter: Column='{column_name}', Condition='{condition}', Value='{value}'")
			self.undo_stack.append(self.df.copy())
			self.redo_stack.clear()

			filtered_df = self.apply_filter(column_name, condition, value)
			if filtered_df is not None:
				self.df = filtered_df
				self.update_table(self.df)
				logging.info("Filter applied successfully.")
			else:
				self.show_error_message("Filter Error", "No rows match the filter criteria.")
		except Exception as e:
			self.show_error_message("Error applying filter", str(e))
			logging.error(f"Error applying filter: {e}")

	def apply_filter(self, column_name: str, condition: str, value: str) -> Optional[pd.DataFrame]:
		"""Apply a filter to the DataFrame based on the given criteria."""
		try:
			if condition == "Contains":
				mask = self.df[column_name].str.contains(value, na=False)
			elif condition == "Equals":
				# Try numeric comparison first, fall back to string comparison
				try:
					numeric_value = float(value)
					mask = self.df[column_name].astype(float) == numeric_value
				except ValueError:
					mask = self.df[column_name].astype(str) == value
			elif condition == "Starts with":
				mask = self.df[column_name].str.startswith(value, na=False)
			elif condition == "Ends with":
				mask = self.df[column_name].str.endswith(value, na=False)
			elif condition in (">=", "<=", ">", "<"):
				try:
					numeric_column = pd.to_numeric(self.df[column_name], errors="coerce")
					numeric_value = float(value)

					if condition == ">=":
						mask = numeric_column >= numeric_value
					elif condition == "<=":
						mask = numeric_column <= numeric_value
					elif condition == ">":
						mask = numeric_column > numeric_value
					else:  # <
						mask = numeric_column < numeric_value
				except ValueError as e:
					raise ValueError(f"Could not convert column '{column_name}' or value '{value}' to float: {e}")
			else:
				raise ValueError("Invalid filter condition")

			filtered_df = self.df[mask]
			return filtered_df if not filtered_df.empty else None

		except Exception as e:
			logging.error(f"Error in filter application: {e}")
			raise

	def copy_selected_values(self) -> None:
		"""
		Copies the selected values from the table widget.

		Returns:
		    None

		Raises:
		    None
		"""
		selected_items = self.table_widget.selectedItems()
		if not selected_items:
			logging.warning("No cells selected to copy.")
			self.show_error_message("Copy Error", "No cells selected to copy.")
			return

	def show_error_message(self, title: str, message: str) -> None:
		"""
		Display an error message dialog box.

		Parameters:
		- title (str): The title of the error dialog box.
		- message (str): The error message to be displayed.

		Returns:
		- None
		"""
		error_dialog = QMessageBox()
		error_dialog.setIcon(QMessageBox.Critical)
		error_dialog.setWindowTitle(title)
		error_dialog.setText(message)
		error_dialog.exec_()
		logging.error(f"Error dialog shown - Title: '{title}', Message: '{message}'")

	def set_window_size_limits(self) -> None:
		"""
		Sets the minimum and maximum size limits for the window.

		Returns:
		    None
		"""
		logging.info("Setting window size limits.")
		self.min_width, self.min_height = 400, 300
		self.max_width, self.max_height = 1200, 800
		self.setMinimumSize(self.min_width, self.min_height)
		self.setMaximumSize(self.max_width, self.max_height)

	def setup_context_menu(self) -> None:
		"""
		Sets up the context menu for the table widget.

		This method sets the context menu policy of the table widget to Qt.CustomContextMenu,
		allowing the widget to display a custom context menu when the user right-clicks on it.

		Returns:
		    None
		"""
		logging.debug("Setting up context menu for table widget.")
		self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)

	def show_context_menu(self, pos: QPoint) -> None:
		"""
		Displays a context menu at the given position.

		Args:
		    pos (QPoint): The position where the context menu is requested.
		"""
		logging.debug(f"Context menu requested at position: {pos}")
		item = self.table_widget.itemAt(pos)

		if not item:
			return

		column_index = item.column()
		column_name = self.df.columns[column_index]
		logging.debug(f"Context menu for column: {column_name}")

		context_menu = QMenu(self)
		context_menu.addAction("Undo", self.undo)
		context_menu.addAction("Redo", self.redo)

		if self.df[column_name].dtype == "object":
			context_menu.addMenu(self.create_one_hot_menu(column_name))
		elif self.df[column_name].isnull().any():
			context_menu.addMenu(self.create_impute_menu(column_name))

		context_menu.addAction("Copy", self.copy_selected_values)

		# Add RDKit menu for string columns that might contain SMILES
		if self.df[column_name].dtype == "object":
			context_menu.addMenu(self.create_rdkit_menu(column_name))

		context_menu.exec_(self.table_widget.mapToGlobal(pos))

	def update_column_dropdown(self) -> None:
		"""
		Updates the column dropdown menu with the columns of the DataFrame.

		Returns:
		    None
		"""
		self.column_dropdown.clear()
		self.column_dropdown.addItems(self.df.columns)

	def get_dataframe(self) -> pd.DataFrame:
		"""
		Returns the DataFrame containing the CSV data.

		Returns:
		    pd.DataFrame: The DataFrame containing the CSV data.
		"""
		return self.df

	def create_message_box(self, title: str, text: str, icon: QMessageBox.Icon = QMessageBox.Warning) -> QMessageBox:
		"""Creates a standardized message box."""
		msg_box = QMessageBox(self)
		msg_box.setIcon(icon)
		msg_box.setWindowTitle(title)
		msg_box.setText(text)
		return msg_box

	def show_warning(self, title: str, message: str) -> None:
		"""Shows a warning message box."""
		self.create_message_box(title, message, QMessageBox.Warning).exec_()

	def show_error(self, title: str, message: str) -> None:
		"""Shows an error message box."""
		self.create_message_box(title, message, QMessageBox.Critical).exec_()

	def create_rdkit_menu(self, column_name: str) -> QMenu:
		"""Creates a menu for RDKit operations on SMILES columns."""
		rdkit_menu = QMenu("RDKit Operations", self)

		# Add canonicalization option
		rdkit_menu.addAction("Convert to Canonical SMILES", lambda: self.canonicalize_smiles(column_name))

		# Add separator before descriptor options
		rdkit_menu.addSeparator()

		# Add RDKit descriptor options
		rdkit_menu.addAction(
			"Add Basic Properties (MW, LogP, TPSA)",
			lambda: self.add_rdkit_descriptors(column_name, descriptor_set="basic"),
		)
		rdkit_menu.addAction("Add Custom Properties...", lambda: self.show_descriptor_selector(column_name))
		rdkit_menu.addAction(
			"Add All Properties", lambda: self.add_rdkit_descriptors(column_name, descriptor_set="all")
		)

		# Add fingerprint submenu
		fp_menu = QMenu("Add Fingerprints", self)
		fp_menu.addAction(
			"ECFP4 (Morgan)",
			lambda: self.add_rdkit_descriptors(column_name, descriptor_set="fingerprints", fp_type="morgan"),
		)
		fp_menu.addAction(
			"ECFP6 (Morgan r=3)",
			lambda: self.add_rdkit_descriptors(column_name, descriptor_set="fingerprints", fp_type="morgan_r3"),
		)
		fp_menu.addAction(
			"MACCS Keys",
			lambda: self.add_rdkit_descriptors(column_name, descriptor_set="fingerprints", fp_type="maccs"),
		)
		fp_menu.addAction(
			"Topological",
			lambda: self.add_rdkit_descriptors(column_name, descriptor_set="fingerprints", fp_type="topological"),
		)
		fp_menu.addAction(
			"Atom Pairs",
			lambda: self.add_rdkit_descriptors(column_name, descriptor_set="fingerprints", fp_type="atompairs"),
		)
		rdkit_menu.addMenu(fp_menu)

		return rdkit_menu

	def show_descriptor_selector(self, column_name: str) -> None:
		"""
		Show dialog for selecting custom RDKit descriptors.

		Args:
			column_name: Name of column containing SMILES strings
		"""
		try:
			# Get list of all available descriptors
			desc_names = sorted([desc_name[0] for desc_name in Descriptors._descList])

			# Create dialog
			dialog = QDialog(self)
			dialog.setWindowTitle("Select RDKit Descriptors")
			layout = QVBoxLayout()

			# Add search box
			search_box = QLineEdit()
			search_box.setPlaceholderText("Search descriptors...")
			layout.addWidget(search_box)

			# Add list widget with checkboxes
			list_widget = QListWidget()
			list_widget.setSelectionMode(QListWidget.MultiSelection)

			# Add all descriptors to list
			for desc in desc_names:
				item = QListWidgetItem(desc)
				item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
				item.setCheckState(Qt.Unchecked)
				list_widget.addItem(item)

			layout.addWidget(list_widget)

			# Add search functionality
			def filter_list():
				search_text = search_box.text().lower()
				for i in range(list_widget.count()):
					item = list_widget.item(i)
					item.setHidden(search_text not in item.text().lower())

			search_box.textChanged.connect(filter_list)

			# Add buttons
			button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
			button_box.accepted.connect(dialog.accept)
			button_box.rejected.connect(dialog.reject)
			layout.addWidget(button_box)

			dialog.setLayout(layout)

			# Show dialog and get result
			if dialog.exec_() == QDialog.Accepted:
				# Get selected descriptors
				selected_descs = []
				for i in range(list_widget.count()):
					item = list_widget.item(i)
					if item.checkState() == Qt.Checked:
						selected_descs.append(item.text())

				if selected_descs:
					self.add_rdkit_descriptors(column_name, descriptor_set="custom", custom_descriptors=selected_descs)
				else:
					self.show_warning("No Selection", "No descriptors were selected.")

		except Exception as e:
			self.show_error("Error", f"Failed to show descriptor selector: {str(e)}")
			logging.error(f"Error in descriptor selector: {str(e)}")

	def add_rdkit_descriptors(
		self,
		column_name: str,
		descriptor_set: str = "basic",
		custom_descriptors: List[str] = None,
		fp_type: str = "morgan",
	) -> None:
		"""Add RDKit descriptors or fingerprints for molecules in a SMILES column."""
		try:
			# Validate custom descriptors first
			if descriptor_set == "custom":
				if not custom_descriptors:
					raise ValueError("No descriptors specified for custom descriptor set")
				if not all(hasattr(Descriptors, desc_name) for desc_name in custom_descriptors):
					raise ValueError("Invalid descriptor name found")

			# Convert SMILES to molecules using cached function
			mols = [get_mol_from_smiles(smiles) for smiles in self.df[column_name] if pd.notna(smiles)]
			if not all(mols):
				raise ValueError("Invalid SMILES strings detected in column")

			# Save current state for undo
			self.undo_stack.append(self.df.copy())
			self.redo_stack.clear()

			if descriptor_set == "fingerprints":
				self._add_fingerprints(column_name, fp_type)
			elif descriptor_set == "basic":
				self._add_basic_descriptors(column_name)
			elif descriptor_set == "all":
				self._add_all_descriptors(column_name)
			elif descriptor_set == "custom":
				self._add_custom_descriptors(column_name, custom_descriptors)

			self.update_table(self.df)
			self.data_ready.emit(self.df)

		except Exception as e:
			if isinstance(e, ValueError):
				raise e
			self.show_error("RDKit Error", str(e))
			logging.error(f"Error in RDKit processing: {str(e)}")
			raise

	def _add_basic_descriptors(self, column_name: str) -> None:
		"""Add basic molecular descriptors efficiently."""
		descriptors = {"MW": Descriptors.ExactMolWt, "LogP": Descriptors.MolLogP, "TPSA": Descriptors.TPSA}

		for desc_name, desc_func in descriptors.items():
			self.df[f"{column_name}_{desc_name}"] = self.df[column_name].apply(
				lambda x: desc_func(get_mol_from_smiles(x)) if pd.notna(x) else None
			)

	def _add_fingerprints(self, column_name: str, fp_type: str) -> None:
		"""Add molecular fingerprints with optimized computation."""
		if fp_type == "morgan":
			self._add_morgan_fingerprints(column_name, radius=2)
		elif fp_type == "morgan_r3":
			self._add_morgan_fingerprints(column_name, radius=3)
		elif fp_type == "maccs":
			self._add_maccs_fingerprints(column_name)
		elif fp_type == "topological":
			self._add_topological_fingerprints(column_name)
		elif fp_type == "atompairs":
			self._add_atompair_fingerprints(column_name)

	def _add_morgan_fingerprints(self, column_name: str, radius: int = 2) -> None:
		"""Add Morgan fingerprints efficiently."""

		def get_morgan_fp(smiles: str) -> List[int]:
			mol = get_mol_from_smiles(smiles)
			if mol is None:
				return [0] * 1024
			return [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, radius, 1024).ToBitString()]

		fps = pd.DataFrame(
			self.df[column_name].apply(get_morgan_fp).tolist(),
			columns=[f"{column_name}_ECFP{radius*2}_{i}" for i in range(1024)],
		)
		self.df = pd.concat([self.df, fps], axis=1)

	def _add_maccs_fingerprints(self, column_name: str) -> None:
		"""Add MACCS keys fingerprints efficiently."""

		def get_maccs_fp(smiles: str) -> List[int]:
			mol = get_mol_from_smiles(smiles)
			if mol is None:
				return [0] * 167
			return [int(x) for x in AllChem.GetMACCSKeysFingerprint(mol).ToBitString()]

		fps = pd.DataFrame(
			self.df[column_name].apply(get_maccs_fp).tolist(), columns=[f"{column_name}_MACCS_{i}" for i in range(167)]
		)
		self.df = pd.concat([self.df, fps], axis=1)

	def _add_topological_fingerprints(self, column_name: str) -> None:
		"""Add topological fingerprints efficiently."""

		def get_topological_fp(smiles: str) -> List[int]:
			mol = get_mol_from_smiles(smiles)
			if mol is None:
				return [0] * 2048
			return [int(x) for x in AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048).ToBitString()]

		fps = pd.DataFrame(
			self.df[column_name].apply(get_topological_fp).tolist(),
			columns=[f"{column_name}_Topological_{i}" for i in range(2048)],
		)
		self.df = pd.concat([self.df, fps], axis=1)

	def _add_atompair_fingerprints(self, column_name: str) -> None:
		"""Add atom pair fingerprints efficiently."""

		def get_atompair_fp(smiles: str) -> List[int]:
			mol = get_mol_from_smiles(smiles)
			if mol is None:
				return [0] * 2048
			return [int(x) for x in AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048).ToBitString()]

		fps = pd.DataFrame(
			self.df[column_name].apply(get_atompair_fp).tolist(),
			columns=[f"{column_name}_AtomPair_{i}" for i in range(2048)],
		)
		self.df = pd.concat([self.df, fps], axis=1)

	def _add_all_descriptors(self, column_name: str) -> None:
		"""Add all available descriptors efficiently."""
		desc_names = [desc_name[0] for desc_name in Descriptors._descList]

		# Test descriptors on first valid molecule
		test_mol = next(
			mol
			for mol in (get_mol_from_smiles(smiles) for smiles in self.df[column_name] if pd.notna(smiles))
			if mol is not None
		)

		# Filter descriptors that work with test molecule
		valid_descriptors = {}
		for desc_name in desc_names:
			try:
				desc_func = getattr(Descriptors, desc_name)
				_ = desc_func(test_mol)
				valid_descriptors[desc_name] = desc_func
			except Exception as e:
				logging.warning(f"Skipping descriptor {desc_name} due to error: {str(e)}")
				continue

		# Calculate descriptors in batches
		for desc_name, desc_func in valid_descriptors.items():
			self.df[f"{column_name}_{desc_name}"] = self.df[column_name].apply(
				lambda x: desc_func(get_mol_from_smiles(x)) if pd.notna(x) else None
			)

	def _add_custom_descriptors(self, column_name: str, custom_descriptors: List[str]) -> None:
		"""Add custom descriptors efficiently."""
		valid_descriptors = {}
		for desc_name in custom_descriptors:
			desc_func = getattr(Descriptors, desc_name)
			valid_descriptors[desc_name] = desc_func

		for desc_name, desc_func in valid_descriptors.items():
			self.df[f"{column_name}_{desc_name}"] = self.df[column_name].apply(
				lambda x: desc_func(get_mol_from_smiles(x)) if pd.notna(x) else None
			)

	def canonicalize_smiles(self, column_name: str) -> None:
		"""
		Convert SMILES strings to their canonical form using RDKit.

		Args:
			column_name: Name of column containing SMILES strings
		"""
		try:
			# Validate SMILES strings first
			mols = [Chem.MolFromSmiles(smiles) for smiles in self.df[column_name] if pd.notna(smiles)]
			if not all(mols):
				raise ValueError("Invalid SMILES strings detected in column")

			# Save current state for undo
			self.undo_stack.append(self.df.copy())
			self.redo_stack.clear()

			# Convert to canonical SMILES
			def to_canonical(smiles):
				if pd.isna(smiles):
					return smiles
				mol = Chem.MolFromSmiles(smiles)
				return Chem.MolToSmiles(mol, canonical=True)

			self.df[column_name] = self.df[column_name].apply(to_canonical)

			self.update_table(self.df)
			self.data_ready.emit(self.df)

		except Exception as e:
			# Re-raise ValueError exceptions
			if isinstance(e, ValueError):
				raise e
			# Log and show other errors
			self.show_error("RDKit Error", str(e))
			logging.error(f"Error in SMILES canonicalization: {str(e)}")
			raise  # Re-raise the exception to ensure test failure
