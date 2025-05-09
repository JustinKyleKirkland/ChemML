import ast
import logging
from typing import Dict, List, Optional, Set, Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDialog,
	QDialogButtonBox,
	QFileDialog,
	QFrame,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QListWidget,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QScrollArea,
	QStatusBar,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)

from chemml.ui.controllers.ml_controller import MLController, MLResult, AVAILABLE_ML_METHODS, MODEL_CONFIGS


# Modern styling constants
MAIN_STYLE = """
    QWidget {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
    }
    QGroupBox {
        border: 1px solid #cccccc;
        border-radius: 4px;
        margin-top: 1em;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
    }
"""

BUTTON_STYLE = """
    QPushButton {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #0056b3;
    }
    QPushButton:pressed {
        background-color: #004085;
    }
    QPushButton:disabled {
        background-color: #cccccc;
    }
"""

LIST_STYLE = """
    QListWidget {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px;
    }
    QListWidget::item {
        padding: 4px;
        border-radius: 2px;
    }
    QListWidget::item:selected {
        background-color: #007bff;
        color: white;
    }
    QListWidget::item:hover {
        background-color: #e9ecef;
    }
"""

COMBO_STYLE = """
    QComboBox {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px 8px;
        min-width: 6em;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
"""

STATUS_STYLE = """
    QStatusBar {
        background-color: #f8f9fa;
        border-top: 1px solid #dee2e6;
        padding: 4px;
    }
"""


class CustomParamsDialog(QDialog):
	"""Dialog for customizing model parameters with modern styling."""

	def __init__(self, model_name: str, current_params: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self.model_name = model_name
		self.current_params = current_params
		self.setup_ui()

	def setup_ui(self) -> None:
		"""Sets up the dialog UI with modern styling."""
		self.setStyleSheet(MAIN_STYLE)
		self.setWindowTitle(f"Customize Parameters - {self.model_name}")
		self.setMinimumWidth(500)

		layout = QVBoxLayout()
		layout.setSpacing(15)

		# Create header
		header = QLabel(f"Configure Hyperparameters for {self.model_name}")
		header.setStyleSheet("font-weight: bold; font-size: 14px;")
		layout.addWidget(header)

		# Add description
		description = QLabel(
			"Enter parameters as a Python dictionary. Each parameter should be a list of values to try.\n"
			"Example: {'max_depth': [10, 20], 'min_samples_split': [2, 5]}"
		)
		description.setWordWrap(True)
		description.setStyleSheet("color: #666666;")
		layout.addWidget(description)

		# Create parameter input
		param_group = QGroupBox("Parameters")
		param_layout = QVBoxLayout()

		self.param_input = QTextEdit()
		self.param_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px;
                background-color: #ffffff;
            }
        """)
		self.param_input.setText(str(self.current_params))
		self.param_input.setMinimumHeight(100)

		param_layout.addWidget(self.param_input)
		param_group.setLayout(param_layout)
		layout.addWidget(param_group)

		# Add button box with styling
		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		button_box.accepted.connect(self.accept)
		button_box.rejected.connect(self.reject)

		for button in button_box.buttons():
			button.setStyleSheet(BUTTON_STYLE)

		layout.addWidget(button_box)
		self.setLayout(layout)

	def get_params(self) -> Dict[str, Any]:
		"""Returns the custom parameters as a dictionary with error handling."""
		try:
			params = ast.literal_eval(self.param_input.toPlainText())
			if not isinstance(params, dict):
				raise ValueError("Parameters must be a dictionary")
			return params
		except (SyntaxError, ValueError) as e:
			QMessageBox.warning(
				self,
				"Invalid Parameters",
				f"Error parsing parameters: {str(e)}\nUsing default parameters instead.",
				QMessageBox.Ok,
			)
			return self.current_params


class MethodSelectionWidget(QWidget):
	"""Widget for selecting ML methods with an improved modern interface."""

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self.available_methods: Set[str] = set(AVAILABLE_ML_METHODS)
		self.selected_methods: Set[str] = set()
		self.custom_params: Dict[str, Dict[str, Any]] = {}
		self.setup_ui()

	def setup_ui(self) -> None:
		"""Sets up the main UI layout with modern styling."""
		layout = QVBoxLayout()
		self.setStyleSheet(MAIN_STYLE)

		# Create main group box
		group_box = QGroupBox("Method Selection")
		group_layout = QHBoxLayout()

		# Available methods section
		available_layout = QVBoxLayout()
		available_label = QLabel("Available Methods")
		available_label.setStyleSheet("font-weight: bold;")
		self.available_list = self._create_list_widget()
		self.available_list.addItems(sorted(self.available_methods))
		available_layout.addWidget(available_label)
		available_layout.addWidget(self.available_list)

		# Transfer buttons
		button_layout = self._create_transfer_buttons()

		# Selected methods section
		selected_layout = QVBoxLayout()
		selected_label = QLabel("Selected Methods")
		selected_label.setStyleSheet("font-weight: bold;")
		self.selected_list = self._create_list_widget()
		self.customize_button = QPushButton("Customize Parameters")
		self.customize_button.setStyleSheet(BUTTON_STYLE)
		self.customize_button.clicked.connect(self._customize_parameters)
		self.customize_button.setToolTip("Configure hyperparameters for the selected method")

		selected_layout.addWidget(selected_label)
		selected_layout.addWidget(self.selected_list)
		selected_layout.addWidget(self.customize_button)

		# Add all sections to the group layout
		group_layout.addLayout(available_layout)
		group_layout.addLayout(button_layout)
		group_layout.addLayout(selected_layout)

		# Set the group box layout
		group_box.setLayout(group_layout)
		layout.addWidget(group_box)
		self.setLayout(layout)

	def _create_list_widget(self) -> QListWidget:
		"""Creates a styled list widget with multi-selection."""
		list_widget = QListWidget()
		list_widget.setSelectionMode(QListWidget.MultiSelection)
		list_widget.setStyleSheet(LIST_STYLE)
		list_widget.setMinimumWidth(250)
		list_widget.setMinimumHeight(200)
		return list_widget

	def _create_transfer_buttons(self) -> QVBoxLayout:
		"""Creates styled transfer buttons between lists."""
		layout = QVBoxLayout()
		layout.addStretch()

		right_button = QPushButton("→")
		left_button = QPushButton("←")

		for button in (right_button, left_button):
			button.setStyleSheet(BUTTON_STYLE)
			button.setFixedWidth(40)
			button.setFixedHeight(40)

		right_button.setToolTip("Add selected methods")
		left_button.setToolTip("Remove selected methods")

		right_button.clicked.connect(self._move_to_selected)
		left_button.clicked.connect(self._move_to_available)

		layout.addWidget(right_button)
		layout.addWidget(left_button)
		layout.addStretch()

		return layout

	def _move_items(self, source: QListWidget, target: QListWidget) -> None:
		"""Moves selected items between lists with visual feedback."""
		items = source.selectedItems()
		for item in items:
			target.addItem(item.text())
			source.takeItem(source.row(item))
		target.sortItems()

	def _move_to_selected(self) -> None:
		self._move_items(self.available_list, self.selected_list)

	def _move_to_available(self) -> None:
		self._move_items(self.selected_list, self.available_list)

	def _customize_parameters(self) -> None:
		"""Opens an improved dialog to customize parameters."""
		selected_items = self.selected_list.selectedItems()
		if not selected_items:
			QMessageBox.warning(
				self, "No Method Selected", "Please select a method to customize its parameters.", QMessageBox.Ok
			)
			return

		method_name = selected_items[0].text()
		current_params = self.custom_params.get(method_name, MODEL_CONFIGS[method_name]["param_grid"])

		dialog = CustomParamsDialog(method_name, current_params, self)
		if dialog.exec_() == QDialog.Accepted:
			self.custom_params[method_name] = dialog.get_params()

	def get_selected_methods(self) -> List[str]:
		"""Returns list of selected ML methods."""
		return sorted([self.selected_list.item(i).text() for i in range(self.selected_list.count())])

	def get_custom_params(self) -> Dict[str, Dict[str, Any]]:
		"""Returns dictionary of custom parameters for each method."""
		return self.custom_params.copy()


class FeatureSelectionWidget(QWidget):
	"""Widget for selecting target and feature columns with an improved modern interface."""

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self.previous_target: Optional[str] = None
		self._columns_cache: Dict[str, List[str]] = {}
		self.setup_ui()

	def setup_ui(self) -> None:
		"""Sets up the main UI layout with modern styling."""
		layout = QVBoxLayout()
		self.setStyleSheet(MAIN_STYLE)

		# Create main group box
		group_box = QGroupBox("Feature Selection")
		group_layout = QVBoxLayout()

		# Target selection section
		target_layout = QHBoxLayout()
		target_label = QLabel("Target Column:")
		target_label.setStyleSheet("font-weight: bold;")
		self.target_combo = QComboBox()
		self.target_combo.setStyleSheet(COMBO_STYLE)
		self.target_combo.addItem("Select Target Column")
		self.target_combo.setMinimumWidth(200)
		self.target_combo.setToolTip("Select the column to predict")

		target_layout.addWidget(target_label)
		target_layout.addWidget(self.target_combo)
		target_layout.addStretch()

		# Feature selection section
		feature_layout = QHBoxLayout()

		# Available features
		available_layout = QVBoxLayout()
		available_label = QLabel("Available Features")
		available_label.setStyleSheet("font-weight: bold;")
		self.available_list = self._create_list_widget()
		available_layout.addWidget(available_label)
		available_layout.addWidget(self.available_list)

		# Transfer buttons
		button_layout = self._create_transfer_buttons()

		# Selected features
		selected_layout = QVBoxLayout()
		selected_label = QLabel("Selected Features")
		selected_label.setStyleSheet("font-weight: bold;")
		self.selected_list = self._create_list_widget()
		selected_layout.addWidget(selected_label)
		selected_layout.addWidget(self.selected_list)

		# Add sections to feature layout
		feature_layout.addLayout(available_layout)
		feature_layout.addLayout(button_layout)
		feature_layout.addLayout(selected_layout)

		# Add all layouts to group
		group_layout.addLayout(target_layout)
		group_layout.addSpacing(10)
		group_layout.addLayout(feature_layout)

		# Set the group box layout
		group_box.setLayout(group_layout)
		layout.addWidget(group_box)
		self.setLayout(layout)

	def _create_list_widget(self) -> QListWidget:
		"""Creates a styled list widget with multi-selection."""
		list_widget = QListWidget()
		list_widget.setSelectionMode(QListWidget.MultiSelection)
		list_widget.setStyleSheet(LIST_STYLE)
		list_widget.setMinimumWidth(250)
		list_widget.setMinimumHeight(200)
		return list_widget

	def _create_transfer_buttons(self) -> QVBoxLayout:
		"""Creates styled transfer buttons between lists."""
		layout = QVBoxLayout()
		layout.addStretch()

		right_button = QPushButton("→")
		left_button = QPushButton("←")

		for button in (right_button, left_button):
			button.setStyleSheet(BUTTON_STYLE)
			button.setFixedWidth(40)
			button.setFixedHeight(40)

		right_button.setToolTip("Add selected features")
		left_button.setToolTip("Remove selected features")

		right_button.clicked.connect(self._move_to_selected)
		left_button.clicked.connect(self._move_to_available)

		layout.addWidget(right_button)
		layout.addWidget(left_button)
		layout.addStretch()

		return layout

	def _move_items(self, source: QListWidget, target: QListWidget) -> None:
		"""Moves selected items between lists with visual feedback."""
		items = source.selectedItems()
		for item in items:
			target.addItem(item.text())
			source.takeItem(source.row(item))
		target.sortItems()

	def _move_to_selected(self) -> None:
		self._move_items(self.available_list, self.selected_list)

	def _move_to_available(self) -> None:
		self._move_items(self.selected_list, self.available_list)

	def _get_cached_columns(self, df: pd.DataFrame) -> List[str]:
		"""Returns cached columns for a dataframe."""
		key = str(sorted(df.columns.tolist()))
		if key not in self._columns_cache:
			self._columns_cache[key] = sorted(df.columns.tolist())
		return self._columns_cache[key]

	def update_columns(self, df: pd.DataFrame) -> None:
		"""Updates available columns from new dataframe."""
		columns = self._get_cached_columns(df)

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
		return sorted([self.selected_list.item(i).text() for i in range(self.selected_list.count())])


class MLView(QWidget):
	"""Main ML view widget with modern interface and status feedback."""

	data_ready_signal = pyqtSignal(pd.DataFrame)

	def __init__(self) -> None:
		super().__init__()
		self.logger = logging.getLogger("MLView")

		# Initialize the controller
		self.controller = MLController()

		self.setup_ui()
		self._connect_signals()

	def setup_ui(self) -> None:
		"""Sets up the main UI layout with modern styling."""
		self.setWindowTitle("Machine Learning View")
		self.setStyleSheet(MAIN_STYLE)

		# Create main layout
		main_layout = QVBoxLayout()
		main_layout.setSpacing(10)

		# Create scroll area for content
		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setFrameShape(QFrame.NoFrame)

		# Create content widget
		content = QWidget()
		content_layout = QVBoxLayout()
		content_layout.setSpacing(15)

		# Add selection widgets
		self.method_selection = MethodSelectionWidget()
		self.feature_selection = FeatureSelectionWidget()

		content_layout.addWidget(self.feature_selection)
		content_layout.addWidget(self.method_selection)

		# Create progress bar
		self.progress_bar = QProgressBar()
		self.progress_bar.setVisible(False)
		self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
		content_layout.addWidget(self.progress_bar)

		# Create action buttons container
		button_container = QWidget()
		button_layout = QHBoxLayout()
		button_layout.setSpacing(10)

		# Create action buttons
		self.run_button = self._create_action_button(
			"Run ML Methods", self._run_ml_methods, "Train and evaluate selected machine learning models"
		)
		self.plot_button = self._create_action_button(
			"Plot Results", self._plot_results, "Visualize the results of trained models"
		)
		self.plot_button.setEnabled(False)  # Disabled until results are available

		button_layout.addStretch()
		button_layout.addWidget(self.run_button)
		button_layout.addWidget(self.plot_button)
		button_layout.addStretch()

		button_container.setLayout(button_layout)
		content_layout.addWidget(button_container)

		# Set content layout
		content.setLayout(content_layout)
		scroll.setWidget(content)

		# Create status bar
		self.status_bar = QStatusBar()
		self.status_bar.setStyleSheet(STATUS_STYLE)

		# Add components to main layout
		main_layout.addWidget(scroll)
		main_layout.addWidget(self.status_bar)

		self.setLayout(main_layout)

		# Set initial status
		self.update_status("Ready to start. Please select features and methods.")

	def _create_action_button(self, text: str, callback: callable, tooltip: str = "") -> QPushButton:
		"""Creates a styled action button with tooltip."""
		button = QPushButton(text)
		button.setStyleSheet(BUTTON_STYLE)
		button.setFixedHeight(40)
		button.setMinimumWidth(150)
		button.clicked.connect(callback)
		if tooltip:
			button.setToolTip(tooltip)
		return button

	def _connect_signals(self) -> None:
		"""Connects all signal handlers."""
		self.data_ready_signal.connect(self.update_column_selection)
		self.feature_selection.target_combo.currentIndexChanged.connect(self._handle_target_change)

	def set_dataframe(self, df: pd.DataFrame) -> None:
		"""Updates the view with new data."""
		if df is not None and not df.empty:
			self.controller.set_dataframe(df)
			self.data_ready_signal.emit(df)
			self.update_status(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

	def update_column_selection(self, df: pd.DataFrame) -> None:
		"""Updates column selections when data changes."""
		self.feature_selection.update_columns(df)

	def update_status(self, message: str) -> None:
		"""Updates the status bar with a new message."""
		self.status_bar.showMessage(message)

	def _handle_target_change(self) -> None:
		"""Handles changes to target column selection."""
		if self.controller.df.empty:
			return

		current_target = self.feature_selection.get_target_column()
		columns = sorted(self.controller.df.columns.tolist())

		self.feature_selection.available_list.clear()

		if current_target and current_target != "Select Target Column":
			available_columns = [col for col in columns if col != current_target]
			self.feature_selection.available_list.addItems(available_columns)
			self.update_status(f"Selected '{current_target}' as target variable")
		else:
			self.feature_selection.available_list.addItems(columns)

	def _validate_selections(self) -> bool:
		"""Validates that all necessary selections are made."""
		target = self.feature_selection.get_target_column()
		features = self.feature_selection.get_feature_columns()
		methods = self.method_selection.get_selected_methods()

		# Use the controller for validation
		is_valid, error_message = self.controller.validate_selections(target, features, methods)

		if not is_valid:
			self._show_warning(error_message)

		return is_valid

	def _show_warning(self, message: str, title: str = "Warning") -> None:
		"""Shows a warning message box and updates status."""
		self.logger.warning(message)
		self.update_status(f"Warning: {message}")
		QMessageBox.warning(self, title, message)

	def _run_ml_methods(self) -> None:
		"""Run selected ML methods with current settings."""
		if not self._validate_selections():
			return

		# Get selections
		target_column = self.feature_selection.get_target_column()
		selected_features = self.feature_selection.get_feature_columns()
		selected_methods = self.method_selection.get_selected_methods()

		# Update custom parameters in controller
		for method, params in self.method_selection.get_custom_params().items():
			self.controller.set_custom_parameters(method, params)

		# Update status and show progress bar
		self.update_status("Running ML methods...")
		self.run_button.setEnabled(False)
		self.progress_bar.setVisible(True)
		self.progress_bar.setRange(0, len(selected_methods))
		self.progress_bar.setValue(0)

		try:
			# Run methods with controller
			results = self.controller.run_ml_methods(
				target_column=target_column, feature_columns=selected_features, selected_methods=selected_methods
			)

			# Update progress bar
			self.progress_bar.setValue(len(selected_methods))

			# Enable plot button and show results
			self.plot_button.setEnabled(True)
			self._show_results_dialog(results)
			self.update_status("ML methods completed successfully.")

		except Exception as e:
			self.logger.error(f"Error running ML methods: {str(e)}")
			self._show_warning(f"Failed to run ML methods: {str(e)}", "Error")

		finally:
			self.run_button.setEnabled(True)
			self.progress_bar.setVisible(False)

	def _show_results_dialog(self, results: Dict[str, MLResult]) -> None:
		"""Shows dialog with ML results and save option."""
		# Find best model based on R²
		best_method = max(results.keys(), key=lambda m: results[m].test_r2)
		best_result = results[best_method]

		result_str = self._format_result_string(best_method, best_result)

		msg_box = QMessageBox(self)
		msg_box.setIcon(QMessageBox.Information)
		msg_box.setWindowTitle("Best ML Method Result")
		msg_box.setText(result_str)

		save_button = msg_box.addButton("Save All Results", QMessageBox.ActionRole)
		msg_box.addButton("Close", QMessageBox.RejectRole)

		msg_box.exec_()

		if msg_box.clickedButton() == save_button:
			self._save_results(results)

	def _format_result_string(self, method: str, result: MLResult) -> str:
		"""Formats the result string for display."""
		return (
			f"Best Method: {method}\n\n"
			f"Cross-Validation Results:\n"
			f"  Mean Score: {result.cv_mean_score:.4f}\n"
			f"  Std Score: {result.cv_std_score:.4f}\n\n"
			f"Performance Metrics:\n"
			f"  Train MSE: {result.train_mse:.4f}\n"
			f"  Test MSE: {result.test_mse:.4f}\n"
			f"  Train R²: {result.train_r2:.4f}\n"
			f"  Test R²: {result.test_r2:.4f}\n\n"
			f"Best Hyperparameters:\n"
			f"  {result.best_hyperparameters}\n"
		)

	def _save_results(self, results: Dict[str, MLResult]) -> None:
		"""Saves results to a JSON file."""
		filename, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "JSON Files (*.json)")
		if filename:
			try:
				self.controller.save_results_to_json(results, filename)
				self.update_status(f"Results saved to {filename}")
			except Exception as e:
				self._show_warning(f"Error saving results: {str(e)}", "Save Error")

	def _plot_results(self) -> None:
		"""Opens dialog to select models for plotting."""
		selected_methods = self.method_selection.get_selected_methods()
		if not selected_methods:
			self._show_warning("No ML methods available for plotting.")
			return

		dialog = self._create_plot_selection_dialog(selected_methods)
		dialog.exec_()

	def _create_plot_selection_dialog(self, methods: List[str]) -> QDialog:
		"""Creates an improved dialog for selecting methods to plot."""
		dialog = QDialog(self)
		dialog.setWindowTitle("Select Methods to Plot")
		dialog.setStyleSheet(MAIN_STYLE)
		layout = QVBoxLayout()

		# Add header
		header = QLabel("Select the methods you want to visualize:")
		header.setStyleSheet("font-weight: bold;")
		layout.addWidget(header)

		# Create checkboxes with better styling
		checkboxes = {}
		for method in sorted(methods):
			checkbox = QCheckBox(method)
			checkbox.setStyleSheet("padding: 5px;")
			layout.addWidget(checkbox)
			checkboxes[method] = checkbox

		# Add buttons with styling
		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		button_box.accepted.connect(
			lambda: self._create_selected_plots([m for m, cb in checkboxes.items() if cb.isChecked()])
		)
		button_box.rejected.connect(dialog.reject)

		for button in button_box.buttons():
			button.setStyleSheet(BUTTON_STYLE)

		layout.addWidget(button_box)
		dialog.setLayout(layout)
		return dialog

	def _create_selected_plots(self, methods: List[str]) -> None:
		"""Creates plots for selected methods with progress feedback."""
		if not methods:
			self._show_warning("No methods selected for plotting.")
			return

		try:
			target = self.feature_selection.get_target_column()
			features = self.feature_selection.get_feature_columns()

			for method in methods:
				self.update_status(f"Creating plot for {method}...")

				# Check if result is in cache
				result = self.controller.get_cached_result(method, target, features)

				if result:
					self._create_scatter_plot(result, method)
				else:
					# Run the model if not in cache
					results = self.controller.run_ml_methods(
						target_column=target, feature_columns=features, selected_methods=[method]
					)

					if method in results:
						self._create_scatter_plot(results[method], method)

			self.update_status("Plots created successfully.")

		except Exception as e:
			self._show_warning(f"Error plotting results: {str(e)}", "Plot Error")

	def _create_scatter_plot(self, result: MLResult, method: str) -> None:
		"""Creates an improved scatter plot comparing actual vs predicted values."""
		# Set the style
		sns.set_style("whitegrid")
		sns.set_context("notebook", font_scale=1.5)

		# Create figure with improved aesthetics
		plt.figure(figsize=(12, 8))

		# Calculate the perfect prediction line
		min_val = min(min(result.y_test), min(result.test_predictions))
		max_val = max(max(result.y_test), max(result.test_predictions))
		perfect_line = np.linspace(min_val, max_val, 100)

		# Plot perfect prediction line
		plt.plot(perfect_line, perfect_line, "--", color="#ff4d4d", label="Perfect Prediction", alpha=0.5, linewidth=2)

		# Create scatter plot with seaborn
		sns.scatterplot(
			x=result.y_test, y=result.test_predictions, alpha=0.6, label="Test Data", color="#007bff", s=100
		)

		# Customize the plot
		plt.xlabel("Actual Values", fontsize=12, fontweight="bold")
		plt.ylabel("Predicted Values", fontsize=12, fontweight="bold")
		plt.title(f"{method}\nActual vs Predicted Values", fontsize=16, pad=20, fontweight="bold")

		# Enhance the legend with larger font and box
		legend = plt.legend(
			frameon=True,
			fancybox=True,
			shadow=True,
			fontsize=11,
			bbox_to_anchor=(1.02, 1),
			loc="upper left",
			edgecolor="#cccccc",
			facecolor="white",
			markerscale=1.5,
		)
		# Set transparency for the legend box
		legend.get_frame().set_alpha(0.9)

		# Add R² score text with better styling and larger box
		plt.text(
			0.05,
			0.90,
			f"R² Score: {result.test_r2:.2f}",
			transform=plt.gca().transAxes,
			bbox=dict(
				facecolor="white",
				alpha=0.9,
				edgecolor="#cccccc",
				boxstyle="round,pad=1.0",
				mutation_scale=2.0,
				linewidth=2,
			),
			fontsize=12,
			fontweight="bold",
			verticalalignment="center",
		)

		# Adjust layout to prevent text cutoff
		plt.tight_layout()

		# Show the plot
		plt.show()

		# Reset the style for future plots
		plt.style.use("default")
