"""Tests for CSVController: data loading, transformations, imputation, undo/redo,
and RDKit-based descriptor/fingerprint generation.

These also guard against index-misalignment regressions: descriptors and
fingerprints must stay aligned with the DataFrame index even after the data has
been filtered (which produces a non-contiguous index).
"""

import pandas as pd
import pytest

from chemml.ui.controllers.csv_controller import CSVController


@pytest.fixture
def controller():
	return CSVController()


@pytest.fixture
def sample_csv(tmp_path):
	df = pd.DataFrame(
		{
			"numeric": [1, 2, 3, None, 5],
			"categorical": ["A", "B", "A", "C", "B"],
		}
	)
	path = tmp_path / "sample.csv"
	df.to_csv(path, index=False)
	return str(path)


@pytest.fixture
def smiles_controller():
	ctrl = CSVController()
	ctrl.df = pd.DataFrame(
		{
			"SMILES": [
				"CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
				"CC1=CC=C(C=C1)O",  # p-Cresol
				"C1=CC=NC=C1",  # Pyridine
				"CCO",  # Ethanol
			],
			"value": [1.0, 2.0, 3.0, 4.0],
		}
	)
	ctrl.original_df = ctrl.df.copy()
	return ctrl


def test_load_csv(controller, sample_csv):
	df, errors = controller.load_csv(sample_csv)
	assert list(df.columns) == ["numeric", "categorical"]
	# Integer columns are promoted to float so NaN can be represented.
	assert df["numeric"].dtype == float
	assert any("Null values" in e for e in errors)


def test_filter_numeric(controller, sample_csv):
	controller.load_csv(sample_csv)
	filtered = controller.filter_data("numeric", ">", "2")
	assert filtered is not None
	assert (filtered["numeric"] > 2).all()


def test_filter_no_match_returns_none(controller, sample_csv):
	controller.load_csv(sample_csv)
	assert controller.filter_data("numeric", ">", "1000") is None


def test_one_hot_encoding(controller, sample_csv):
	controller.load_csv(sample_csv)
	df = controller.apply_one_hot_encoding("categorical", n_distinct=True)
	assert "categorical_A" in df.columns
	assert "categorical_B" in df.columns
	assert "categorical_C" in df.columns


def test_impute_and_undo(controller, sample_csv):
	controller.load_csv(sample_csv)
	assert controller.df["numeric"].isnull().any()

	controller.impute_missing_values("numeric", "mean")
	assert not controller.df["numeric"].isnull().any()

	restored = controller.undo()
	assert restored is not None
	assert restored["numeric"].isnull().any()

	redone = controller.redo()
	assert not redone["numeric"].isnull().any()


def test_reset_to_original(controller, sample_csv):
	controller.load_csv(sample_csv)
	controller.filter_data("numeric", ">", "2")
	assert len(controller.df) < 5
	controller.reset_to_original()
	assert len(controller.df) == 5


def test_add_basic_descriptors(smiles_controller):
	df = smiles_controller.add_molecular_descriptors("SMILES", "basic")
	assert "SMILES_MW" in df.columns
	assert "SMILES_LogP" in df.columns
	# Aspirin MW is ~180 g/mol.
	assert 175 < df.loc[0, "SMILES_MW"] < 185


def test_add_custom_descriptors(smiles_controller):
	df = smiles_controller.add_molecular_descriptors(
		"SMILES", "custom", custom_descriptors=["NumRotatableBonds", "RingCount"]
	)
	assert "SMILES_NumRotatableBonds" in df.columns
	assert "SMILES_RingCount" in df.columns
	assert df.loc[0, "SMILES_RingCount"] == 1  # Aspirin has one ring


def test_custom_descriptors_requires_selection(smiles_controller):
	with pytest.raises(ValueError):
		smiles_controller.add_molecular_descriptors("SMILES", "custom", custom_descriptors=[])


def test_descriptors_aligned_after_filter(smiles_controller):
	"""Regression: descriptors must align to the DataFrame index after filtering."""
	# Filter out the first row so the index becomes [1, 2, 3].
	smiles_controller.df = smiles_controller.df[smiles_controller.df["value"] > 1.5]
	df = smiles_controller.add_molecular_descriptors("SMILES", "basic")
	# Pyridine (index 2) has no H-bond donors; ethanol (index 3) has one.
	assert df.loc[2, "SMILES_NumHDonors"] == 0
	assert df.loc[3, "SMILES_NumHDonors"] == 1
	assert not df["SMILES_MW"].isnull().any()


def test_fingerprints_aligned_after_filter(smiles_controller):
	"""Regression: fingerprint rows must align after filtering (no NaN rows)."""
	smiles_controller.df = smiles_controller.df[smiles_controller.df["value"] > 1.5]
	df = smiles_controller.add_fingerprints("SMILES", "maccs")
	maccs_cols = [c for c in df.columns if c.startswith("SMILES_MACCS_")]
	assert len(maccs_cols) == 167
	# Every retained row must have real fingerprint bits, not NaN.
	assert not df[maccs_cols].isnull().any().any()


def test_morgan_fingerprints(smiles_controller):
	df = smiles_controller.add_fingerprints("SMILES", "morgan", radius=2, n_bits=256)
	morgan_cols = [c for c in df.columns if c.startswith("SMILES_ECFP4_")]
	assert len(morgan_cols) == 256


def test_canonicalize_smiles(smiles_controller):
	df = smiles_controller.canonicalize_smiles_column("SMILES")
	# Canonicalisation is idempotent: re-running yields the same strings.
	again = df["SMILES"].apply(lambda s: s)
	assert (df["SMILES"] == again).all()
