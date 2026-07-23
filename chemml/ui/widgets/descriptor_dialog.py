"""Dialog for selecting molecular descriptors to compute from a SMILES column."""

from typing import List, Optional

from PyQt5.QtWidgets import (
	QAbstractItemView,
	QDialog,
	QDialogButtonBox,
	QLabel,
	QLineEdit,
	QListWidget,
	QListWidgetItem,
	QVBoxLayout,
	QWidget,
)

from chemml.core.chem import get_available_descriptors


class DescriptorSelectionDialog(QDialog):
	"""A searchable, multi-select list of RDKit descriptors.

	Use :pymeth:`get_selected_descriptors` after the dialog is accepted to
	retrieve the chosen descriptor names.
	"""

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self._all_descriptors = get_available_descriptors()
		self._setup_ui()

	def _setup_ui(self) -> None:
		self.setWindowTitle("Select Molecular Descriptors")
		self.setMinimumSize(420, 480)

		layout = QVBoxLayout()
		layout.setSpacing(10)
		layout.setContentsMargins(15, 15, 15, 15)

		header = QLabel(f"Choose from {len(self._all_descriptors)} available RDKit descriptors:")
		header.setStyleSheet("font-weight: bold;")
		layout.addWidget(header)

		self.search_box = QLineEdit()
		self.search_box.setPlaceholderText("Search descriptors (e.g. 'MolWt', 'TPSA', 'NumH')")
		self.search_box.setClearButtonEnabled(True)
		self.search_box.textChanged.connect(self._filter_list)
		layout.addWidget(self.search_box)

		self.list_widget = QListWidget()
		self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
		for name in self._all_descriptors:
			self.list_widget.addItem(QListWidgetItem(name))
		layout.addWidget(self.list_widget)

		self.selection_label = QLabel("0 selected")
		self.selection_label.setStyleSheet("color: #6c757d;")
		self.list_widget.itemSelectionChanged.connect(self._update_count)
		layout.addWidget(self.selection_label)

		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		button_box.accepted.connect(self.accept)
		button_box.rejected.connect(self.reject)
		layout.addWidget(button_box)

		self.setLayout(layout)

	def _filter_list(self, text: str) -> None:
		"""Show only descriptors whose name contains the search text."""
		needle = text.strip().lower()
		for i in range(self.list_widget.count()):
			item = self.list_widget.item(i)
			item.setHidden(needle not in item.text().lower())

	def _update_count(self) -> None:
		self.selection_label.setText(f"{len(self.list_widget.selectedItems())} selected")

	def get_selected_descriptors(self) -> List[str]:
		"""Return the names of the currently selected descriptors."""
		return [item.text() for item in self.list_widget.selectedItems()]
