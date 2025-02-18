import logging
from typing import List, Tuple

import pandas as pd
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QFontMetrics
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
	QTabWidget,
	QVBoxLayout,
	QWidget,
)

from utils.color_assets import ERROR_COLOR, NAN_COLOR, NON_NUM_COLOR
from utils.data_utils import impute_values, one_hot_encode, validate_csv


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
		Imputes missing values for all columns in the DataFrame using the specified strategy.

		Parameters:
		    strategy (str): The imputation strategy to use.

		Returns:
		    None
		"""
		logging.info(f"Imputing missing values for all columns using {strategy}")
		try:
			self.undo_stack.append(self.df.copy())
			self.redo_stack.clear()

			for column in self.df.columns:
				if self.df[column].isnull().any():
					self.df = impute_values(self.df, column, strategy)

			self.update_table(self.df)
			self.update_column_dropdown()

			self.data_ready.emit(self.df)
			self.show_errors(validate_csv(self.df))

			logging.info(f"Missing values imputed for all columns using {strategy}")
		except ValueError as e:
			self.show_error_message("Imputation Error", str(e))
			logging.error(f"Error in imputing missing values: {e}")

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
		return table_widget

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
		Imputes missing values in a specific column of the DataFrame using a specified strategy.

		Args:
		    column_name (str): The name of the column to impute missing values.
		    strategy (str): The imputation strategy to use.

		Returns:
		    None

		Raises:
		    ValueError: If an error occurs during the imputation process.

		"""
		logging.info(f"Imputing missing values for column: {column_name} using {strategy}")
		try:
			self.undo_stack.append(self.df.copy())
			self.redo_stack.clear()

			self.df = impute_values(self.df, column_name, strategy)

			self.update_table(self.df)
			self.update_column_dropdown()

			if self.df[column_name].isnull().any():
				warning_msg = (
					f"Warning: Column '{column_name}' still contains null values after applying {strategy} imputation."
				)
				self.show_errors([warning_msg])
				logging.warning(warning_msg)
			else:
				logging.info(f"Missing values imputed for column: {column_name} using {strategy}")
				self.show_errors([])  # Clear previous errors if successful
		except ValueError as e:
			self.show_error_message("Imputation Error", str(e))
			logging.error(f"Error in imputing missing values: {e}")

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
		"""
		Update the table widget with new data from a DataFrame.

		Args:
		    df (pd.DataFrame): The DataFrame containing the new data.

		Returns:
		    None
		"""
		logging.debug("Updating table with new data.")
		self.table_widget.clear()
		self.table_widget.setRowCount(len(df))
		self.table_widget.setColumnCount(len(df.columns))
		self.table_widget.setHorizontalHeaderLabels(df.columns)

		for col_index, column in enumerate(df.columns):
			has_nan = df[column].isnull().any()
			is_numerical = pd.api.types.is_numeric_dtype(df[column])

			for row_index in range(len(df)):
				item = QTableWidgetItem(str(df.at[row_index, column]))

				if has_nan:
					item.setBackground(QBrush(NAN_COLOR))

				if not is_numerical:
					item.setBackground(QBrush(NON_NUM_COLOR))

				self.table_widget.setItem(row_index, col_index, item)

		self.table_widget.resizeColumnsToContents()
		logging.info("Table updated successfully.")

	def filter_data(self) -> None:
		"""
		Apply a filter to the data based on the selected column, condition, and value.

		Returns:
		    None

		Raises:
		    Exception: If an error occurs while applying the filter.

		Warnings:
		    If the filter criteria are missing, a warning is logged and an error message is displayed.

		Logging:
		    - Logs the filter being applied with the selected column, condition, and value.
		    - Logs an error if an exception occurs while applying the filter.
		    - Logs a warning if the filter criteria are missing.

		Stack:
		    - Appends a copy of the current DataFrame to the undo stack.
		    - Clears the redo stack.

		Table Update:
		    - Updates the table with the filtered DataFrame.

		Error Handling:
		    - Displays an error message if an exception occurs while applying the filter.
		"""
		column_name = self.column_dropdown.currentText()
		condition = self.condition_dropdown.currentText()
		value = self.value_edit.text()

		if column_name and condition and value:
			try:
				logging.info(f"Applying filter: Column='{column_name}', Condition='{condition}', Value='{value}'")
				self.undo_stack.append(self.df.copy())
				self.redo_stack.clear()

				filtered_df = self.apply_filter(column_name, condition, value)
				self.update_table(filtered_df)
				logging.info("Filter applied successfully.")
			except Exception as e:
				self.show_error_message("Error applying filter", str(e))
				logging.error(f"Error applying filter: {e}")
		else:
			logging.warning("Filter criteria missing. Please select a column, condition, and value.")
			self.show_error_message("Invalid Filter", "Please select a column, condition, and value.")

	# TODO: Fix filter function
	def apply_filter(self, column_name: str, condition: str, value: str) -> pd.DataFrame:
		"""
		Apply a filter to the DataFrame based on the given column name, condition, and value.

		Parameters:
		    column_name (str): The name of the column to filter on.
		    condition (str): The condition to apply for the filter. Valid options are "Contains", "Equals",
		    "Starts with", "Ends with", ">=", "<=", ">", "<".
		    value (str): The value to compare against for the filter.

		Returns:
		    pd.DataFrame: The filtered DataFrame.

		Raises:
		    ValueError: If an invalid filter condition is provided or if the column or value cannot be converted
		    to float when using numeric conditions.

		"""
		if condition == "Contains":
			return self.df[self.df[column_name].str.contains(value)]
		elif condition == "Equals":
			return self.df[self.df[column_name] == value]
		elif condition == "Starts with":
			return self.df[self.df[column_name].str.startswith(value)]
		elif condition == "Ends with":
			return self.df[self.df[column_name].str.endswith(value)]
		elif condition in (">=", "<=", ">", "<"):
			try:
				numeric_column = self.df[column_name].astype(float)
				numeric_value = float(value)
			except ValueError as e:
				raise ValueError(f"Could not convert column '{column_name}' or value '{value}' to float: {e}")

			if condition == ">=":
				return self.df[numeric_column >= numeric_value]
			elif condition == "<=":
				return self.df[numeric_column <= numeric_value]
			elif condition == ">":
				return self.df[numeric_column > numeric_value]
			elif condition == "<":
				return self.df[numeric_column < numeric_value]
		else:
			raise ValueError("Invalid filter condition")

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
