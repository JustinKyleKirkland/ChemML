import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from gui.gui import CSVInteractiveApp


class MockSignal:
	"""Mock signal that can be used to track emissions."""

	def __init__(self):
		self.handlers = []
		self.connect = self._connect
		self.emit = self._emit

	def _connect(self, handler):
		"""Connect a handler to the signal."""
		self.handlers.append(handler)

	def _emit(self, *args, **kwargs):
		"""Emit the signal to all connected handlers."""
		for handler in self.handlers:
			handler(*args, **kwargs)


class MockCSVView(QWidget):
	"""Mock CSVView class that emits signals properly."""

	def __init__(self):
		super().__init__()
		self.data_ready = MockSignal()


class MockPlottingWidget(QWidget):
	"""Mock PlottingWidget class."""

	def __init__(self):
		super().__init__()
		self.update_data = MagicMock()


class MockMLView(QWidget):
	"""Mock MLView class."""

	def __init__(self):
		super().__init__()
		self.update_column_selection = MagicMock()
		self.set_dataframe = MagicMock()


@pytest.fixture(scope="session")
def qapp():
	"""Create the QApplication instance only once."""
	app = QApplication.instance()
	if app is None:
		app = QApplication(sys.argv)
	yield app


@pytest.fixture
def main_window(qapp):
	"""Create the main window for each test."""
	window = CSVInteractiveApp()

	# Replace the views with mocks
	window.csv_view = MockCSVView()
	window.plotting_widget = MockPlottingWidget()
	window.ml_view = MockMLView()

	# Reconnect signals with mocks
	window._connect_signals()

	yield window
	QTest.qWait(10)  # Allow for any pending events to process
	window.close()
	window.deleteLater()


def test_window_initialization(main_window):
	"""Test if the main window initializes correctly."""
	assert main_window.windowTitle() == "ChemML"
	assert main_window.tab_widget.count() == 3
	assert main_window.tab_widget.tabText(0) == "CSV View"
	assert main_window.tab_widget.tabText(1) == "Plot View"
	assert main_window.tab_widget.tabText(2) == "ML View"


def test_data_ready_signal_connections(main_window, qapp):
	"""Test if data ready signals are properly connected."""
	test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

	# Verify signal connections
	assert len(main_window.csv_view.data_ready.handlers) == 3

	# Emit data ready signal
	main_window.csv_view.data_ready.emit(test_df)

	# Process any pending events
	QTest.qWait(10)

	# Verify handlers were called
	main_window.plotting_widget.update_data.assert_called_once_with(test_df)
	main_window.ml_view.update_column_selection.assert_called_once_with(test_df)
	main_window.ml_view.set_dataframe.assert_called_once_with(test_df)


@patch("logging.Logger.info")
def test_close_event_logging(mock_logger, main_window, qapp):
	"""Test if closing the application logs the appropriate message."""
	main_window.close()
	QTest.qWait(10)  # Allow for any pending events to process
	mock_logger.assert_called_with("Application closing...")
