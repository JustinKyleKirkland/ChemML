"""Smoke tests for CSVView wiring against the current (controller-backed) API."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from PyQt5.QtWidgets import QFileDialog

from chemml.ui.views.csv_view import CSVView


@pytest.fixture
def csv_path(tmp_path):
	df = pd.DataFrame(
		{
			"numeric": [1, 2, 3, None, 5],
			"categorical": ["A", "B", "A", "C", "B"],
		}
	)
	path = tmp_path / "test.csv"
	df.to_csv(path, index=False)
	return str(path)


@pytest.fixture
def view(qtbot):
	widget = CSVView()
	qtbot.addWidget(widget)
	return widget


def test_view_initializes_empty(view):
	assert view.controller.df.empty
	assert view.table_widget.rowCount() == 0


def test_load_csv_populates_table(view, csv_path, monkeypatch):
	monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (csv_path, ""))
	view.load_csv()
	assert not view.controller.df.empty
	assert view.table_widget.rowCount() == 5
	assert view.table_widget.columnCount() == 2
	assert view.column_dropdown.count() == 2


def test_undo_redo_through_view(view, csv_path, monkeypatch):
	monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (csv_path, ""))
	view.load_csv()
	view.impute_all_missing_values("mean")
	assert not view.controller.df["numeric"].isnull().any()
	view.undo()
	assert view.controller.df["numeric"].isnull().any()
