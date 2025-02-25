import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDialog,
	QDialogButtonBox,
	QFileDialog,
	QFormLayout,
	QHBoxLayout,
	QLabel,
	QListWidget,
	QMessageBox,
	QPushButton,
	QVBoxLayout,
	QWidget,
)

from ml_backend.ml_backend import download_results_as_json, run_ml_methods

AVAILABLE_ML_METHODS = [
	"Linear Regression",
	"Ridge Regression",
	"Lasso Regression",
	"Elastic Net Regression",
	"Decision Tree Regression",
	"Random Forest Regression",
	"Support Vector Regression",
	"Neural Network Regression",
	"Gradient Boosting Regression",
	"AdaBoost Regression",
	"K-Nearest Neighbors Regression",
	"Gaussian Process Regression",
]

BUTTON_STYLE = """
	background-color: #007bff;
	color: white;
	font-weight: bold;
"""


@dataclass
class MLResult:
	"""Stores the results of ML model execution."""

	method_name: str
	cv_mean_score: float
	cv_std_score: float
	train_mse: float
	test_mse: float
	train_r2: float
	test_r2: float
	best_hyperparameters: Dict[str, Any]
	y_test: List[float]
	test_predictions: List[float]


class MethodSelectionWidget(QWidget):
	"""Widget for selecting ML methods."""

	def __init__(self) -> None:
		super().__init__()
		self.setup_ui()

	def setup_ui(self) -> None:
		layout = QHBoxLayout()

		# Available methods list
		self.available_list = self._create_list_widget("Select Machine Learning Methods:")
		self.available_list.addItems(AVAILABLE_ML_METHODS)

		# Selected methods list
		self.selected_list = self._create_list_widget("Selected ML Methods")

		# Transfer buttons
		button_layout = self._create_transfer_buttons()

		layout.addLayout(self._wrap_in_layout(self.available_list))
		layout.addLayout(button_layout)
		layout.addLayout(self._wrap_in_layout(self.selected_list))

		self.setLayout(layout)

	def _create_list_widget(self, label_text: str) -> QListWidget:
		"""Creates a labeled list widget with multi-selection."""
		list_widget = QListWidget()
		list_widget.setSelectionMode(QListWidget.MultiSelection)
		list_widget.setMinimumWidth(200)
		return list_widget

	def _create_transfer_buttons(self) -> QVBoxLayout:
		"""Creates the transfer buttons between lists."""
		layout = QVBoxLayout()

		right_button = QPushButton("→")
		left_button = QPushButton("←")

		right_button.clicked.connect(self._move_to_selected)
		left_button.clicked.connect(self._move_to_available)

		layout.addWidget(right_button)
		layout.addWidget(left_button)
		return layout

	def _wrap_in_layout(self, list_widget: QListWidget) -> QVBoxLayout:
		"""Wraps a list widget in a vertical layout with a label."""
		layout = QVBoxLayout()
		layout.addWidget(QLabel("Available Methods:" if list_widget == self.available_list else "Selected Methods:"))
		layout.addWidget(list_widget)
		return layout

	def _move_items(self, source: QListWidget, target: QListWidget) -> None:
		"""Moves selected items from source to target list."""
		for item in source.selectedItems():
			target.addItem(item.text())
			source.takeItem(source.row(item))

	def _move_to_selected(self) -> None:
		self._move_items(self.available_list, self.selected_list)

	def _move_to_available(self) -> None:
		self._move_items(self.selected_list, self.available_list)

	def get_selected_methods(self) -> List[str]:
		"""Returns list of selected ML methods."""
		return [self.selected_list.item(i).text() for i in range(self.selected_list.count())]


class FeatureSelectionWidget(QWidget):
	"""Widget for selecting target and feature columns."""

	def __init__(self) -> None:
		super().__init__()
		self.previous_target: Optional[str] = None
		self.setup_ui()

	def setup_ui(self) -> None:
		layout = QVBoxLayout()

		# Target selection
		target_layout = QFormLayout()
		self.target_combo = QComboBox()
		self.target_combo.addItem("Select Target Column")
		target_layout.addRow("Target Column:", self.target_combo)
		layout.addLayout(target_layout)

		# Feature selection
		feature_layout = QHBoxLayout()

		self.available_list = self._create_list_widget("Available Feature Columns:")
		self.selected_list = self._create_list_widget("Selected Feature Columns")

		button_layout = self._create_transfer_buttons()

		feature_layout.addLayout(self._wrap_in_layout(self.available_list))
		feature_layout.addLayout(button_layout)
		feature_layout.addLayout(self._wrap_in_layout(self.selected_list))

		layout.addLayout(feature_layout)
		self.setLayout(layout)

	def _create_list_widget(self, label_text: str) -> QListWidget:
		"""Creates a labeled list widget with multi-selection."""
		list_widget = QListWidget()
		list_widget.setSelectionMode(QListWidget.MultiSelection)
		list_widget.setMinimumWidth(200)
		return list_widget

	def _create_transfer_buttons(self) -> QVBoxLayout:
		"""Creates the transfer buttons between lists."""
		layout = QVBoxLayout()

		right_button = QPushButton("→")
		left_button = QPushButton("←")

		right_button.clicked.connect(self._move_to_selected)
		left_button.clicked.connect(self._move_to_available)

		layout.addWidget(right_button)
		layout.addWidget(left_button)
		return layout

	def _wrap_in_layout(self, list_widget: QListWidget) -> QVBoxLayout:
		"""Wraps a list widget in a vertical layout with a label."""
		layout = QVBoxLayout()
		layout.addWidget(QLabel("Available Features:" if list_widget == self.available_list else "Selected Features:"))
		layout.addWidget(list_widget)
		return layout

	def _move_items(self, source: QListWidget, target: QListWidget) -> None:
		"""Moves selected items from source to target list."""
		for item in source.selectedItems():
			target.addItem(item.text())
			source.takeItem(source.row(item))

	def _move_to_selected(self) -> None:
		self._move_items(self.available_list, self.selected_list)

	def _move_to_available(self) -> None:
		self._move_items(self.selected_list, self.available_list)

	def update_columns(self, df: pd.DataFrame) -> None:
		"""Updates available columns from new dataframe."""
		columns = df.columns.tolist()

		self.target_combo.clear()
		self.target_combo.addItem("Select Target Column")
		self.target_combo.addItems(columns)

		self.available_list.clear()
		self.available_list.addItems(columns)

		self.selected_list.clear()

	def get_target_column(self) -> str:
		"""Returns selected target column."""
		return self.target_combo.currentText()

	def get_feature_columns(self) -> List[str]:
		"""Returns list of selected feature columns."""
		return [self.selected_list.item(i).text() for i in range(self.selected_list.count())]


class MLView(QWidget):
	"""Main ML view widget."""

	data_ready_signal = pyqtSignal(pd.DataFrame)

	def __init__(self) -> None:
		super().__init__()
		self.logger = logging.getLogger("MLView")
		self.df = pd.DataFrame()

		self.setup_ui()
		self._connect_signals()

	def setup_ui(self) -> None:
		"""Sets up the main UI layout."""
		self.setWindowTitle("Machine Learning View")
		layout = QVBoxLayout()

		# Create sub-widgets
		self.method_selection = MethodSelectionWidget()
		self.feature_selection = FeatureSelectionWidget()

		# Create action buttons
		self.run_button = self._create_action_button("Run ML Methods", self._run_ml_methods)
		self.plot_button = self._create_action_button("Plot Results", self._plot_results)

		# Add widgets to layout
		layout.addWidget(self.method_selection)
		layout.addWidget(self.feature_selection)
		layout.addWidget(self.run_button)
		layout.addWidget(self.plot_button)

		self.setLayout(layout)

	def _create_action_button(self, text: str, callback: Callable) -> QPushButton:
		"""Creates a styled action button."""
		button = QPushButton(text)
		button.setStyleSheet(BUTTON_STYLE)
		button.setFixedHeight(40)
		button.clicked.connect(callback)
		return button

	def _connect_signals(self) -> None:
		"""Connects all signal handlers."""
		self.data_ready_signal.connect(self.update_column_selection)
		self.feature_selection.target_combo.currentIndexChanged.connect(self._handle_target_change)

	def set_dataframe(self, df: pd.DataFrame) -> None:
		"""Updates the view with new data."""
		self.df = df
		self.data_ready_signal.emit(df)

	def update_column_selection(self, df: pd.DataFrame) -> None:
		"""Updates column selections when data changes."""
		self.feature_selection.update_columns(df)

	def _handle_target_change(self) -> None:
		"""Handles changes to target column selection."""
		if not self.df.empty:
			current_target = self.feature_selection.get_target_column()

			# Get all available columns
			columns = self.df.columns.tolist()

			# Clear and update available features list
			self.feature_selection.available_list.clear()

			if current_target and current_target != "Select Target Column":
				# Add all columns except the target
				available_columns = [col for col in columns if col != current_target]
				self.feature_selection.available_list.addItems(available_columns)
			else:
				# If no target selected, show all columns
				self.feature_selection.available_list.addItems(columns)

	def _validate_selections(self) -> bool:
		"""Validates that all necessary selections are made."""
		target = self.feature_selection.get_target_column()
		features = self.feature_selection.get_feature_columns()
		methods = self.method_selection.get_selected_methods()

		if target == "Select Target Column":
			self._show_warning("Please select a target column.")
			return False
		if not features:
			self._show_warning("Please select feature columns.")
			return False
		if not methods:
			self._show_warning("Please select machine learning methods.")
			return False
		if self.df.empty:
			self._show_warning("Please load a CSV file.")
			return False

		return True

	def _show_warning(self, message: str) -> None:
		"""Shows a warning message box."""
		self.logger.warning(message)
		QMessageBox.warning(self, "Warning", message)

	def _run_ml_methods(self) -> None:
		"""Runs selected ML methods and displays results."""
		if not self._validate_selections():
			return

		try:
			results = run_ml_methods(
				self.df,
				self.feature_selection.get_target_column(),
				self.feature_selection.get_feature_columns(),
				self.method_selection.get_selected_methods(),
			)

			self._show_results_dialog(results)

		except Exception as e:
			self.logger.error(f"Error running ML methods: {e}", exc_info=True)
			QMessageBox.critical(self, "Error", f"Error running ML methods: {e}")

	def _show_results_dialog(self, results: Dict[str, MLResult]) -> None:
		"""Shows dialog with ML results and save option."""
		best_method = max(results.keys(), key=lambda m: results[m].cv_mean_score)
		best_result = results[best_method]

		result_str = self._format_result_string(best_method, best_result)

		msg_box = QMessageBox(self)
		msg_box.setIcon(QMessageBox.Information)
		msg_box.setWindowTitle("Best ML Method Result")
		msg_box.setText(result_str)

		close_button = msg_box.addButton("Close", QMessageBox.RejectRole)
		save_button = msg_box.addButton("Save All Results", QMessageBox.ActionRole)
		msg_box.exec_()

		if msg_box.clickedButton() == save_button:
			self._save_results(results)

	def _format_result_string(self, method: str, result: MLResult) -> str:
		"""Formats the result string for display."""
		return (
			f"Best Method: {method}\n"
			f"  CV Mean Score: {result.cv_mean_score:.4f}\n"
			f"  CV Std Score: {result.cv_std_score:.4f}\n"
			f"  Train MSE: {result.train_mse:.4f}\n"
			f"  Test MSE: {result.test_mse:.4f}\n"
			f"  Train R2: {result.train_r2:.4f}\n"
			f"  Test R2: {result.test_r2:.4f}\n"
			f"  Best Hyperparameters: {result.best_hyperparameters}\n"
		)

	def _save_results(self, results: Dict[str, MLResult]) -> None:
		"""Saves results to a JSON file."""
		filename, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "JSON Files (*.json)")
		if filename:
			download_results_as_json(results, filename)
			self.logger.info(f"Results saved to {filename}")

	def _plot_results(self) -> None:
		"""Opens dialog to select models for plotting."""
		selected_methods = self.method_selection.get_selected_methods()
		if not selected_methods:
			self._show_warning("No ML methods available for plotting.")
			return

		dialog = self._create_plot_selection_dialog(selected_methods)
		dialog.exec_()

	def _create_plot_selection_dialog(self, methods: List[str]) -> QDialog:
		"""Creates dialog for selecting methods to plot."""
		dialog = QDialog(self)
		dialog.setWindowTitle("Select Method to Plot")
		layout = QVBoxLayout()

		# Create checkboxes for each method
		checkboxes = {method: QCheckBox(method) for method in methods}
		for checkbox in checkboxes.values():
			layout.addWidget(checkbox)

		# Add buttons
		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		button_box.accepted.connect(
			lambda: self._create_selected_plots([m for m, cb in checkboxes.items() if cb.isChecked()])
		)
		button_box.rejected.connect(dialog.reject)
		layout.addWidget(button_box)

		dialog.setLayout(layout)
		return dialog

	def _create_selected_plots(self, methods: List[str]) -> None:
		"""Creates plots for selected methods."""
		if not methods:
			self._show_warning("No methods selected for plotting.")
			return

		try:
			results = run_ml_methods(
				self.df,
				self.feature_selection.get_target_column(),
				self.feature_selection.get_feature_columns(),
				methods,
			)

			for method in methods:
				self._create_scatter_plot(results[method], method)

		except Exception as e:
			self._show_warning(f"Error plotting results: {str(e)}")

	def _create_scatter_plot(self, result: MLResult, method: str) -> None:
		"""Creates a scatter plot comparing actual vs predicted values."""
		plt.figure()
		plt.scatter(result.y_test, result.test_predictions, alpha=0.7, label="Predictions")
		plt.xlabel("Actual Values")
		plt.ylabel("Predicted Values")
		plt.title(f"{method} - Actual vs Predicted")
		plt.grid(True)
		plt.show()
