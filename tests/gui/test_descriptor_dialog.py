"""Tests for the molecular descriptor selection dialog."""

import pytest

from chemml.ui.widgets.descriptor_dialog import DescriptorSelectionDialog


@pytest.fixture
def dialog(qtbot):
	dlg = DescriptorSelectionDialog()
	qtbot.addWidget(dlg)
	return dlg


def test_populates_descriptors(dialog):
	assert dialog.list_widget.count() > 100  # RDKit exposes ~200 descriptors


def test_search_filters_list(dialog):
	dialog.search_box.setText("TPSA")
	visible = [
		dialog.list_widget.item(i).text()
		for i in range(dialog.list_widget.count())
		if not dialog.list_widget.item(i).isHidden()
	]
	assert visible  # at least one match
	assert all("tpsa" in name.lower() for name in visible)


def test_search_is_case_insensitive(dialog):
	dialog.search_box.setText("molwt")
	visible = [
		dialog.list_widget.item(i).text()
		for i in range(dialog.list_widget.count())
		if not dialog.list_widget.item(i).isHidden()
	]
	assert any("MolWt" in name for name in visible)


def test_get_selected_descriptors(dialog):
	dialog.list_widget.item(0).setSelected(True)
	dialog.list_widget.item(1).setSelected(True)
	selected = dialog.get_selected_descriptors()
	assert len(selected) == 2
