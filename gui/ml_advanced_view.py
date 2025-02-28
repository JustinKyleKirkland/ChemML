import logging
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QScrollArea,
	QSpinBox,
	QStackedWidget,
	QTabWidget,
	QVBoxLayout,
	QWidget,
)

from ml_backend.model_configs import MODEL_CONFIGS

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


class ParamInputWidget(QWidget):
	"""Widget for configuring a single parameter."""

	param_changed = pyqtSignal(str, object)  # Parameter name, new value

	def __init__(self, param_name: str, param_config: Any, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self.param_name = param_name
		self.param_config = param_config
		self.setup_ui()
		self._emit_value()  # Emit initial value

	def setup_ui(self) -> None:
		"""Set up the parameter input interface."""
		layout = QHBoxLayout()
		layout.setContentsMargins(0, 0, 0, 0)

		# Parameter label
		label = QLabel(self.param_name)
		label.setToolTip(f"Configure {self.param_name} parameter")
		layout.addWidget(label)

		# Input type selector
		self.type_combo = QComboBox()
		self.type_combo.addItems(["Single Value", "Grid Search", "Random Search"])
		self.type_combo.setStyleSheet(COMBO_STYLE)
		layout.addWidget(self.type_combo)

		# Create stacked widget for input types
		self.stack = QStackedWidget()

		# Create and add single value widget
		single_value_container = QWidget()
		single_value_layout = QHBoxLayout()
		single_value_layout.setContentsMargins(0, 0, 0, 0)
		self.single_value_widget = self._create_value_widget()
		single_value_layout.addWidget(self.single_value_widget)
		single_value_container.setLayout(single_value_layout)
		self.stack.addWidget(single_value_container)

		# Create and add range widgets
		range_container = QWidget()
		range_layout = QHBoxLayout()
		range_layout.setContentsMargins(0, 0, 0, 0)
		self.range_widgets = self._create_range_widgets()
		for widget in self.range_widgets:
			range_layout.addWidget(widget)
		range_container.setLayout(range_layout)
		self.stack.addWidget(range_container)

		layout.addWidget(self.stack)
		self.setLayout(layout)
		self.type_combo.currentTextChanged.connect(self._handle_type_change)

	def _create_range_widgets(self) -> List[QWidget]:
		"""Create widgets for range input (start, stop, step)."""
		widgets = []

		# Determine widget type based on parameter type
		if isinstance(self.param_config, dict):
			param_type = self.param_config.get("type", "float")
			default_value = self.param_config.get("default", 0)
		elif isinstance(self.param_config, (list, tuple)) and self.param_config:
			first_value = self.param_config[0]
			param_type = type(first_value).__name__
			default_value = first_value
		else:
			param_type = "float"
			default_value = 0

		# Create appropriate range widgets based on parameter type
		if param_type == "int" or isinstance(default_value, int):
			for name in ["Start", "Stop", "Step"]:
				widget = QSpinBox()
				widget.setRange(-1000000, 1000000)
				widget.setValue(default_value)
				widget.setPrefix(f"{name}: ")
				widget.valueChanged.connect(self._emit_value)
				widgets.append(widget)
		elif param_type == "float" or isinstance(default_value, float):
			for name in ["Start", "Stop", "Step"]:
				widget = QDoubleSpinBox()
				widget.setRange(-1000000.0, 1000000.0)
				widget.setDecimals(4)
				widget.setValue(float(default_value))
				widget.setPrefix(f"{name}: ")
				widget.valueChanged.connect(self._emit_value)
				widgets.append(widget)
		else:
			# For non-numeric types, create a list input
			widget = QLineEdit()
			widget.setPlaceholderText("Enter values separated by commas")
			widget.setText(str(default_value))
			widget.textChanged.connect(self._emit_value)
			widgets.append(widget)

		return widgets

	def _create_value_widget(self) -> QWidget:
		"""Create the appropriate input widget based on parameter type."""
		# Determine parameter type and default value
		if isinstance(self.param_config, dict):
			param_type = self.param_config.get("type", "float")
			default_value = self.param_config.get("default")
		elif isinstance(self.param_config, (list, tuple)) and self.param_config:
			first_value = self.param_config[0]
			param_type = type(first_value).__name__
			default_value = first_value
		else:
			param_type = "float"
			default_value = None

		# Create appropriate widget based on parameter type
		if param_type == "int" or isinstance(default_value, int):
			widget = QSpinBox()
			widget.setRange(-1000000, 1000000)
			widget.setValue(default_value if default_value is not None else 0)
			widget.valueChanged.connect(self._emit_value)
		elif param_type == "float" or isinstance(default_value, float):
			widget = QDoubleSpinBox()
			widget.setRange(-1000000.0, 1000000.0)
			widget.setDecimals(4)
			widget.setValue(default_value if default_value is not None else 0.0)
			widget.valueChanged.connect(self._emit_value)
		elif param_type == "bool" or isinstance(default_value, bool):
			widget = QCheckBox()
			widget.setChecked(default_value if default_value is not None else False)
			widget.stateChanged.connect(self._emit_value)
		elif param_type == "str" or isinstance(default_value, str):
			widget = QComboBox()
			if isinstance(self.param_config, (list, tuple)):
				widget.addItems([str(x) for x in self.param_config])
			else:
				widget.addItems([str(default_value)] if default_value else [])
			if default_value:
				widget.setCurrentText(str(default_value))
			widget.currentTextChanged.connect(self._emit_value)
		else:
			widget = QLineEdit()
			widget.setText(str(default_value) if default_value is not None else "")
			widget.textChanged.connect(self._emit_value)

		return widget

	def _handle_type_change(self, input_type: str) -> None:
		"""Handle changes in the input type."""
		# Switch to appropriate widget stack
		if input_type == "Single Value":
			self.stack.setCurrentIndex(0)
		else:  # Grid Search or Random Search
			self.stack.setCurrentIndex(1)

		self._emit_value()

	def _emit_value(self) -> None:
		"""Emit the current parameter value."""
		value = self._get_current_value()
		self.param_changed.emit(self.param_name, value)

	def _get_current_value(self) -> Any:
		"""Get the current value based on input type."""
		input_type = self.type_combo.currentText()

		if input_type == "Single Value":
			if isinstance(self.single_value_widget, QSpinBox):
				return self.single_value_widget.value()
			elif isinstance(self.single_value_widget, QDoubleSpinBox):
				return self.single_value_widget.value()
			elif isinstance(self.single_value_widget, QCheckBox):
				return self.single_value_widget.isChecked()
			elif isinstance(self.single_value_widget, QComboBox):
				return self.single_value_widget.currentText()
			elif isinstance(self.single_value_widget, QLineEdit):
				return self.single_value_widget.text()
		else:  # Grid Search or Random Search
			if len(self.range_widgets) == 3:  # Numeric type with start/stop/step
				start = self.range_widgets[0].value()
				stop = self.range_widgets[1].value()
				step = self.range_widgets[2].value()
				if isinstance(self.range_widgets[0], QSpinBox):
					return list(range(start, stop + 1, max(1, step)))
				else:  # QDoubleSpinBox
					import numpy as np

					return list(np.arange(start, stop + step / 2, step))
			else:  # Non-numeric type
				try:
					return [x.strip() for x in self.range_widgets[0].text().split(",")]
				except:
					return []

		return None


class ModelParamGroup(QGroupBox):
	"""Group box containing parameters for a specific model."""

	params_changed = pyqtSignal(str, dict)  # Model name, parameter dict

	def __init__(
		self, model_name: str, param_config: Dict[str, Dict[str, Any]], parent: Optional[QWidget] = None
	) -> None:
		super().__init__(model_name, parent)
		self.model_name = model_name
		self.param_config = param_config
		self.current_params: Dict[str, Any] = {}
		self.param_widgets: List[ParamInputWidget] = []
		self.setup_ui()

	def setup_ui(self) -> None:
		"""Set up the parameter group interface."""
		layout = QFormLayout()

		for param_name, param_config in self.param_config.items():
			param_widget = ParamInputWidget(param_name, param_config)
			param_widget.param_changed.connect(self._handle_param_change)
			layout.addRow(param_widget)
			self.param_widgets.append(param_widget)

		self.setLayout(layout)

	def _handle_param_change(self, param_name: str, value: Any) -> None:
		"""Handle changes in parameter values."""
		self.current_params[param_name] = value
		self.params_changed.emit(self.model_name, self.current_params.copy())


class MLAdvancedView(QWidget):
	"""Advanced ML settings view with parameter configuration capabilities."""

	settings_changed = pyqtSignal(dict)  # Updated settings dictionary

	def __init__(self) -> None:
		super().__init__()
		self.logger = logging.getLogger("MLAdvancedView")
		self.current_settings: Dict[str, Dict[str, Any]] = {
			"general": {
				"cv_folds": 5,
				"test_size": 0.2,
				"random_state": 42,
				"n_jobs": -1,
			}
		}
		self.model_groups: Dict[str, ModelParamGroup] = {}
		self.setup_ui()

	def setup_ui(self) -> None:
		"""Set up the main interface."""
		self.setStyleSheet(MAIN_STYLE)
		layout = QVBoxLayout()

		# Add help text
		help_label = QLabel(
			"Configure advanced settings for machine learning models. "
			"You can set specific parameter values or define parameter ranges for grid/random search."
		)
		help_label.setWordWrap(True)
		help_label.setStyleSheet("color: #666666; padding: 10px;")
		layout.addWidget(help_label)

		# Create scroll area for model parameters
		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setFrameShape(scroll.NoFrame)

		# Create main content widget
		content = QWidget()
		content_layout = QVBoxLayout()

		# Create tabs for different categories of models
		self.tab_widget = QTabWidget()

		# General Settings Tab
		general_tab = QWidget()
		general_layout = QFormLayout()

		# Cross-validation folds
		self.cv_folds = QSpinBox()
		self.cv_folds.setRange(2, 20)
		self.cv_folds.setValue(self.current_settings["general"]["cv_folds"])
		self.cv_folds.setToolTip("Number of folds for cross-validation")
		self.cv_folds.valueChanged.connect(lambda v: self._update_general_setting("cv_folds", v))
		general_layout.addRow("Cross-validation Folds:", self.cv_folds)

		# Test size
		self.test_size = QDoubleSpinBox()
		self.test_size.setRange(0.1, 0.5)
		self.test_size.setSingleStep(0.05)
		self.test_size.setDecimals(2)
		self.test_size.setValue(self.current_settings["general"]["test_size"])
		self.test_size.setToolTip("Proportion of data to use for testing (0.1 to 0.5)")
		self.test_size.valueChanged.connect(lambda v: self._update_general_setting("test_size", v))
		general_layout.addRow("Test Set Size:", self.test_size)

		# Random state
		self.random_state = QSpinBox()
		self.random_state.setRange(-1, 999999)
		self.random_state.setValue(self.current_settings["general"]["random_state"])
		self.random_state.setToolTip("Random seed for reproducibility (-1 for random)")
		self.random_state.valueChanged.connect(lambda v: self._update_general_setting("random_state", v))
		general_layout.addRow("Random Seed:", self.random_state)

		# Number of jobs
		self.n_jobs = QSpinBox()
		self.n_jobs.setRange(-1, 32)
		self.n_jobs.setValue(self.current_settings["general"]["n_jobs"])
		self.n_jobs.setToolTip("Number of parallel jobs (-1 for all available cores)")
		self.n_jobs.valueChanged.connect(lambda v: self._update_general_setting("n_jobs", v))
		general_layout.addRow("Parallel Jobs:", self.n_jobs)

		# Add descriptions
		descriptions = QLabel(
			"\nSettings Description:\n\n"
			"• Cross-validation Folds: Number of folds for model validation\n"
			"• Test Set Size: Proportion of data used for testing (e.g., 0.2 = 20%)\n"
			"• Random Seed: Set for reproducible results (-1 for random)\n"
			"• Parallel Jobs: Number of CPU cores to use (-1 for all cores)\n"
		)
		descriptions.setWordWrap(True)
		descriptions.setStyleSheet("color: #666666; padding: 10px;")
		general_layout.addRow(descriptions)

		general_tab.setLayout(general_layout)
		self.tab_widget.addTab(general_tab, "General Settings")

		# Linear Models Tab
		linear_tab = QWidget()
		linear_layout = QVBoxLayout()
		linear_models = {
			k: v
			for k, v in MODEL_CONFIGS.items()
			if "regression" in k.lower() and any(x in k.lower() for x in ["linear", "ridge", "lasso", "elastic"])
		}
		for model_name, config in linear_models.items():
			group = ModelParamGroup(model_name, config.param_grid)
			group.params_changed.connect(self._handle_settings_change)
			linear_layout.addWidget(group)
			self.model_groups[model_name] = group
		linear_tab.setLayout(linear_layout)
		self.tab_widget.addTab(linear_tab, "Linear Models")

		# Tree-based Models Tab
		tree_tab = QWidget()
		tree_layout = QVBoxLayout()
		tree_models = {
			k: v for k, v in MODEL_CONFIGS.items() if any(x in k.lower() for x in ["tree", "forest", "boost"])
		}
		for model_name, config in tree_models.items():
			group = ModelParamGroup(model_name, config.param_grid)
			group.params_changed.connect(self._handle_settings_change)
			tree_layout.addWidget(group)
			self.model_groups[model_name] = group
		tree_tab.setLayout(tree_layout)
		self.tab_widget.addTab(tree_tab, "Tree-Based Models")

		# Other Models Tab
		other_tab = QWidget()
		other_layout = QVBoxLayout()
		other_models = {
			k: v
			for k, v in MODEL_CONFIGS.items()
			if not any(x in k.lower() for x in ["linear", "ridge", "lasso", "elastic", "tree", "forest", "boost"])
		}
		for model_name, config in other_models.items():
			group = ModelParamGroup(model_name, config.param_grid)
			group.params_changed.connect(self._handle_settings_change)
			other_layout.addWidget(group)
			self.model_groups[model_name] = group
		other_tab.setLayout(other_layout)
		self.tab_widget.addTab(other_tab, "Other Models")

		content_layout.addWidget(self.tab_widget)
		content.setLayout(content_layout)
		scroll.setWidget(content)
		layout.addWidget(scroll)

		self.setLayout(layout)

	def _handle_settings_change(self, model_name: str, params: Dict[str, Any]) -> None:
		"""Handle changes in model parameters."""
		self.current_settings[model_name] = params
		self.settings_changed.emit(self.current_settings.copy())
		self.logger.debug(f"Settings changed for {model_name}: {params}")

	def get_current_settings(self) -> Dict[str, Dict[str, Any]]:
		"""Get the current settings for all models."""
		return self.current_settings.copy()

	def _update_general_setting(self, setting_name: str, value: Any) -> None:
		"""Update a general setting value."""
		self.current_settings["general"][setting_name] = value
		self.settings_changed.emit(self.current_settings.copy())
		self.logger.debug(f"General setting {setting_name} changed to {value}")
