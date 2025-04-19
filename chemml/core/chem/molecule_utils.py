import io
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
from PyQt5.QtGui import QPixmap
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw


# Cache molecule objects to avoid repeated parsing
@lru_cache(maxsize=1024)
def get_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
	"""Cache and return RDKit molecule objects.

	Args:
	    smiles: SMILES string representation of molecule

	Returns:
	    RDKit Mol object or None if invalid SMILES
	"""
	if pd.isna(smiles):
		return None
	return Chem.MolFromSmiles(smiles)


def detect_smiles_column(df: pd.DataFrame, column_name: str, threshold: float = 0.8) -> bool:
	"""
	Check if a column contains valid SMILES strings.

	Args:
	    df: DataFrame containing the column
	    column_name: Name of the column to check
	    threshold: Minimum fraction of valid SMILES needed (default 0.8)

	Returns:
	    bool: True if the column contains valid SMILES strings
	"""
	try:
		# Check first few non-null values
		sample = df[column_name].dropna().head()
		if len(sample) == 0:
			return False

		valid_count = sum(1 for smiles in sample if Chem.MolFromSmiles(smiles) is not None)
		# Consider it a SMILES column if at least threshold% of samples are valid
		return (valid_count / len(sample)) >= threshold
	except Exception:
		return False


def canonicalize_smiles(smiles: str) -> str:
	"""
	Convert a SMILES string to its canonical form using RDKit.

	Args:
	    smiles: Input SMILES string

	Returns:
	    Canonical SMILES string or original string if conversion fails
	"""
	if pd.isna(smiles):
		return smiles
	try:
		mol = Chem.MolFromSmiles(smiles)
		if mol:
			return Chem.MolToSmiles(mol, canonical=True)
	except:
		pass
	return smiles


def calculate_basic_descriptors(mol: Chem.Mol) -> Dict[str, float]:
	"""
	Calculate basic molecular descriptors for an RDKit molecule.

	Args:
	    mol: RDKit Mol object

	Returns:
	    Dictionary of descriptor name to value
	"""
	if mol is None:
		return {}

	return {
		"MW": Descriptors.ExactMolWt(mol),
		"LogP": Descriptors.MolLogP(mol),
		"TPSA": Descriptors.TPSA(mol),
		"NumHDonors": Descriptors.NumHDonors(mol),
		"NumHAcceptors": Descriptors.NumHAcceptors(mol),
	}


def calculate_descriptor(mol: Chem.Mol, descriptor_name: str) -> Optional[float]:
	"""
	Calculate a specific RDKit descriptor for a molecule.

	Args:
	    mol: RDKit Mol object
	    descriptor_name: Name of the descriptor function

	Returns:
	    Descriptor value or None if calculation fails
	"""
	if mol is None:
		return None

	try:
		desc_func = getattr(Descriptors, descriptor_name)
		return desc_func(mol)
	except:
		return None


def get_available_descriptors() -> List[str]:
	"""
	Get list of all available RDKit descriptors.

	Returns:
	    List of descriptor names
	"""
	return sorted([desc_name[0] for desc_name in Descriptors._descList])


def generate_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> List[int]:
	"""
	Generate Morgan fingerprint (ECFP) for a molecule.

	Args:
	    mol: RDKit Mol object
	    radius: Fingerprint radius (2=ECFP4, 3=ECFP6)
	    n_bits: Number of bits in the fingerprint

	Returns:
	    Binary fingerprint as list of integers (0 or 1)
	"""
	if mol is None:
		return [0] * n_bits
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
	return [int(b) for b in fp.ToBitString()]


def generate_maccs_fingerprint(mol: Chem.Mol) -> List[int]:
	"""
	Generate MACCS keys fingerprint for a molecule.

	Args:
	    mol: RDKit Mol object

	Returns:
	    Binary fingerprint as list of integers (0 or 1)
	"""
	if mol is None:
		return [0] * 167
	fp = AllChem.GetMACCSKeysFingerprint(mol)
	return [int(b) for b in fp.ToBitString()]


def mol_to_image(mol: Chem.Mol, size: int = 300) -> Optional[QPixmap]:
	"""
	Convert an RDKit molecule to a QPixmap image.

	Args:
	    mol: RDKit Mol object
	    size: Target image width

	Returns:
	    QPixmap object or None if conversion fails
	"""
	if mol is None:
		return None

	img = Draw.MolToImage(mol)
	with io.BytesIO() as bio:
		img.save(bio, format="PNG")
		pixmap = QPixmap()
		pixmap.loadFromData(bio.getvalue())

	if pixmap.width() > size:
		pixmap = pixmap.scaledToWidth(size)

	return pixmap
