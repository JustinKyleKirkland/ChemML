from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QLabel

from chemml.core.chem import get_mol_from_smiles, mol_to_image


class MoleculeTooltip(QLabel):
	"""
	Custom tooltip widget for displaying molecule images.

	This widget shows molecule structures as tooltips when hovering
	over SMILES strings in the application.
	"""

	def __init__(self, parent=None):
		"""Initialize the molecule tooltip widget."""
		super().__init__(parent)
		self.setWindowFlags(Qt.ToolTip)
		self.setAttribute(Qt.WA_TranslucentBackground)
		self.setStyleSheet("QLabel { background-color: white; padding: 2px; border: 1px solid gray; }")
		self._pixmap_cache = {}

	def show_molecule(self, smiles: str, pos: QPoint) -> None:
		"""
		Display molecule image at given position with caching.

		Args:
		    smiles: SMILES string of the molecule to display
		    pos: Screen position where the tooltip should appear
		"""
		if smiles in self._pixmap_cache:
			pixmap = self._pixmap_cache[smiles]
		else:
			mol = get_mol_from_smiles(smiles)
			if not mol:
				return

			pixmap = mol_to_image(mol)
			if not pixmap:
				return

			self._pixmap_cache[smiles] = pixmap

		self.setPixmap(pixmap)
		self.adjustSize()
		self.move(pos)
		self.show()

	def clear_cache(self) -> None:
		"""Clear the pixmap cache to free memory."""
		self._pixmap_cache.clear()
