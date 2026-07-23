"""Settings view for configuring machine-learning run parameters.

Exposes the train/test split, cross-validation folds and random seed used by the
ML view, so users no longer have to accept hard-coded defaults. The view emits
:pyattr:`settings_changed` whenever a value changes and provides
:pymeth:`get_config` returning a plain dict consumed by ``MLView``.
"""

import logging
from typing import Any, Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
	QCheckBox,
	QDoubleSpinBox,
	QFormLayout,
	QGroupBox,
	QLabel,
	QPushButton,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

DEFAULT_CONFIG: Dict[str, Any] = {
	"test_size": 0.2,
	"cv_folds": 5,
	"random_state": 42,
	"scale_features": True,
}


class SettingsView(QWidget):
	"""Widget for configuring ML training hyper-parameters shared across runs."""

	settings_changed = pyqtSignal(dict)

	def __init__(self, parent: QWidget = None) -> None:
		super().__init__(parent)
		self.logger = logging.getLogger("SettingsView")
		self._setup_ui()

	def _setup_ui(self) -> None:
		layout = QVBoxLayout()
		layout.setContentsMargins(20, 20, 20, 20)
		layout.setSpacing(15)

		header = QLabel("Machine Learning Settings")
		header.setStyleSheet("font-size: 16px; font-weight: bold;")
		layout.addWidget(header)

		description = QLabel(
			"These settings control how models are trained and evaluated in the "
			"ML Models tab. Changes apply to the next training run."
		)
		description.setWordWrap(True)
		description.setStyleSheet("color: #6c757d;")
		layout.addWidget(description)

		train_group = QGroupBox("Training && Evaluation")
		form = QFormLayout()
		form.setSpacing(10)

		self.test_size_spin = QDoubleSpinBox()
		self.test_size_spin.setRange(0.05, 0.5)
		self.test_size_spin.setSingleStep(0.05)
		self.test_size_spin.setValue(DEFAULT_CONFIG["test_size"])
		self.test_size_spin.setToolTip("Fraction of the data held out for testing (0.05–0.5).")
		form.addRow("Test set size:", self.test_size_spin)

		self.cv_folds_spin = QSpinBox()
		self.cv_folds_spin.setRange(2, 20)
		self.cv_folds_spin.setValue(DEFAULT_CONFIG["cv_folds"])
		self.cv_folds_spin.setToolTip("Number of cross-validation folds used during grid search.")
		form.addRow("Cross-validation folds:", self.cv_folds_spin)

		self.random_state_spin = QSpinBox()
		self.random_state_spin.setRange(0, 999999)
		self.random_state_spin.setValue(DEFAULT_CONFIG["random_state"])
		self.random_state_spin.setToolTip("Random seed for reproducible train/test splits.")
		form.addRow("Random seed:", self.random_state_spin)

		self.scale_checkbox = QCheckBox("Standardize features before training")
		self.scale_checkbox.setChecked(DEFAULT_CONFIG["scale_features"])
		self.scale_checkbox.setToolTip("Apply StandardScaler so features have zero mean and unit variance.")
		form.addRow("", self.scale_checkbox)

		train_group.setLayout(form)
		layout.addWidget(train_group)

		reset_button = QPushButton("Reset to Defaults")
		reset_button.clicked.connect(self.reset_defaults)
		layout.addWidget(reset_button, alignment=Qt.AlignLeft)

		layout.addStretch(1)
		self.setLayout(layout)

		# Emit whenever any control changes.
		self.test_size_spin.valueChanged.connect(self._emit_changed)
		self.cv_folds_spin.valueChanged.connect(self._emit_changed)
		self.random_state_spin.valueChanged.connect(self._emit_changed)
		self.scale_checkbox.stateChanged.connect(self._emit_changed)

	def get_config(self) -> Dict[str, Any]:
		"""Return the current settings as a dict."""
		return {
			"test_size": round(self.test_size_spin.value(), 2),
			"cv_folds": self.cv_folds_spin.value(),
			"random_state": self.random_state_spin.value(),
			"scale_features": self.scale_checkbox.isChecked(),
		}

	def reset_defaults(self) -> None:
		"""Restore all settings to their default values."""
		self.test_size_spin.setValue(DEFAULT_CONFIG["test_size"])
		self.cv_folds_spin.setValue(DEFAULT_CONFIG["cv_folds"])
		self.random_state_spin.setValue(DEFAULT_CONFIG["random_state"])
		self.scale_checkbox.setChecked(DEFAULT_CONFIG["scale_features"])

	def _emit_changed(self, *args) -> None:
		config = self.get_config()
		self.logger.debug(f"ML settings changed: {config}")
		self.settings_changed.emit(config)
