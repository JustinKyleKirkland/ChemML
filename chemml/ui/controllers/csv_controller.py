import logging
from typing import List, Optional, Tuple

import pandas as pd

from chemml.core.chem import (
	canonicalize_smiles,
	detect_smiles_column,
	calculate_basic_descriptors,
	get_mol_from_smiles,
	generate_morgan_fingerprint,
	generate_maccs_fingerprint,
)
from chemml.core.data import impute_values, one_hot_encode, validate_csv


class CSVController:
	"""
	Controller for CSV data operations, separating business logic from UI.

	This class handles all data operations for the CSV view, including:
	- Loading and validating CSV data
	- Filtering data
	- One-hot encoding
	- Imputing missing values
	- Adding molecular descriptors
	"""

	def __init__(self):
		"""Initialize the CSV controller."""
		self.logger = logging.getLogger("CSVController")
		self.df = pd.DataFrame()
		self.undo_stack: List[pd.DataFrame] = []
		self.redo_stack: List[pd.DataFrame] = []

	def load_csv(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
		"""
		Load a CSV file and validate its contents.

		Args:
		    file_path: Path to the CSV file

		Returns:
		    Tuple of (DataFrame, list of validation errors)

		Raises:
		    Exception: If file cannot be read
		"""
		try:
			self.logger.info(f"Reading CSV file: {file_path}")
			self.df = pd.read_csv(file_path)

			# Store the original DataFrame for reset functionality
			self.original_df = self.df.copy()

			# Convert integer columns to float to handle null values better
			int_columns = self.df.select_dtypes(include="int").columns
			if not int_columns.empty:
				self.logger.info(f"Converting integer columns to nullable integers: {int_columns}")
				self.df[int_columns] = self.df[int_columns].astype(float)

			errors = validate_csv(self.df)

			# Save state for undo
			self._clear_redo_stack()
			self._add_to_undo_stack()

			return self.df, errors

		except Exception as e:
			self.logger.error(f"Error loading CSV file '{file_path}': {e}")
			raise

	def filter_data(self, column_name: str, condition: str, value: str) -> Optional[pd.DataFrame]:
		"""
		Filter the DataFrame based on criteria.

		Args:
		    column_name: Column to filter on
		    condition: Filter condition (Contains, Equals, etc.)
		    value: Value to filter by

		Returns:
		    Filtered DataFrame or None if no rows match

		Raises:
		    ValueError: If filter criteria are invalid
		"""
		try:
			if condition == "Contains":
				mask = self.df[column_name].str.contains(value, na=False)
			elif condition == "Equals":
				# Try numeric comparison first, fall back to string comparison
				try:
					numeric_value = float(value)
					mask = self.df[column_name].astype(float) == numeric_value
				except ValueError:
					mask = self.df[column_name].astype(str) == value
			elif condition == "Starts with":
				mask = self.df[column_name].str.startswith(value, na=False)
			elif condition == "Ends with":
				mask = self.df[column_name].str.endswith(value, na=False)
			elif condition in (">=", "<=", ">", "<"):
				try:
					numeric_column = pd.to_numeric(self.df[column_name], errors="coerce")
					numeric_value = float(value)

					if condition == ">=":
						mask = numeric_column >= numeric_value
					elif condition == "<=":
						mask = numeric_column <= numeric_value
					elif condition == ">":
						mask = numeric_column > numeric_value
					else:  # <
						mask = numeric_column < numeric_value
				except ValueError as e:
					raise ValueError(f"Could not convert column '{column_name}' or value '{value}' to float: {e}")
			else:
				raise ValueError("Invalid filter condition")

			filtered_df = self.df[mask]

			if not filtered_df.empty:
				# Save state for undo
				self._add_to_undo_stack()
				self._clear_redo_stack()

				self.df = filtered_df
				return self.df

			return None

		except Exception as e:
			self.logger.error(f"Error in filter application: {e}")
			raise

	def apply_one_hot_encoding(self, column_name: str, n_distinct: bool = True) -> pd.DataFrame:
		"""
		Apply one-hot encoding to a column.

		Args:
		    column_name: Column to encode
		    n_distinct: Whether to create N distinct columns (True) or N-1 (False)

		Returns:
		    Updated DataFrame

		Raises:
		    ValueError: If encoding fails
		"""
		try:
			self._add_to_undo_stack()
			self._clear_redo_stack()

			self.df = one_hot_encode(self.df, column_name, n_distinct)
			return self.df

		except Exception as e:
			self.logger.error(f"Error in one-hot encoding: {e}")
			raise

	def impute_missing_values(self, column_name: str, strategy: str) -> pd.DataFrame:
		"""
		Impute missing values in a column.

		Args:
		    column_name: Column to impute
		    strategy: Imputation strategy (mean, median, knn, mice)

		Returns:
		    Updated DataFrame

		Raises:
		    ValueError: If imputation fails
		"""
		try:
			if column_name not in self.df.columns:
				raise ValueError(f"Column '{column_name}' not found in DataFrame")

			if not pd.api.types.is_numeric_dtype(self.df[column_name]):
				raise ValueError(f"Column '{column_name}' contains non-numerical values")

			self._add_to_undo_stack()
			self.df = impute_values(self.df, column_name, strategy)
			return self.df

		except Exception as e:
			self.logger.error(f"Error in imputation: {str(e)}")
			raise

	def impute_all_missing_values(self, strategy: str) -> pd.DataFrame:
		"""
		Impute missing values in all columns with missing values.

		Args:
		    strategy: Imputation strategy (mean, median, knn, mice)

		Returns:
		    Updated DataFrame

		Raises:
		    ValueError: If imputation fails
		"""
		try:
			# Check which columns have missing values
			columns_with_missing = [col for col in self.df.columns if self.df[col].isnull().any()]

			if not columns_with_missing:
				return self.df

			# For numerical strategies, check that all columns are numeric
			if strategy in ["mean", "median", "knn", "mice"]:
				non_numeric_cols = [
					col for col in columns_with_missing if not pd.api.types.is_numeric_dtype(self.df[col])
				]

				if non_numeric_cols:
					raise ValueError(
						f"The following columns contain non-numerical values: {', '.join(non_numeric_cols)}"
					)

			self._add_to_undo_stack()

			# Apply imputation to each column
			for column in columns_with_missing:
				self.df = impute_values(self.df, column, strategy)

			return self.df

		except Exception as e:
			self.logger.error(f"Error in imputation: {str(e)}")
			raise

	def undo(self) -> Optional[pd.DataFrame]:
		"""
		Undo the last operation.

		Returns:
		    Restored DataFrame or None if nothing to undo
		"""
		if self.undo_stack:
			self.redo_stack.append(self.df.copy())
			self.df = self.undo_stack.pop()
			return self.df
		return None

	def redo(self) -> Optional[pd.DataFrame]:
		"""
		Redo the last undone operation.

		Returns:
		    Restored DataFrame or None if nothing to redo
		"""
		if self.redo_stack:
			self.undo_stack.append(self.df.copy())
			self.df = self.redo_stack.pop()
			return self.df
		return None

	def canonicalize_smiles_column(self, column_name: str) -> pd.DataFrame:
		"""
		Convert SMILES strings in a column to canonical form.

		Args:
		    column_name: Column containing SMILES strings

		Returns:
		    Updated DataFrame

		Raises:
		    ValueError: If column doesn't contain valid SMILES
		"""
		try:
			# Validate SMILES column
			if not detect_smiles_column(self.df, column_name):
				raise ValueError(f"Column '{column_name}' does not contain valid SMILES strings")

			self._add_to_undo_stack()
			self._clear_redo_stack()

			# Apply canonicalization
			self.df[column_name] = self.df[column_name].apply(canonicalize_smiles)
			return self.df

		except Exception as e:
			self.logger.error(f"Error canonicalizing SMILES: {e}")
			raise

	def add_molecular_descriptors(
		self, column_name: str, descriptor_set: str = "basic", custom_descriptors: List[str] = None
	) -> pd.DataFrame:
		"""
		Add molecular descriptors based on SMILES strings.

		Args:
		    column_name: Column containing SMILES strings
		    descriptor_set: Type of descriptors to add (basic, custom)
		    custom_descriptors: List of descriptor names if descriptor_set is "custom"

		Returns:
		    Updated DataFrame with new descriptor columns

		Raises:
		    ValueError: If SMILES processing fails
		"""
		try:
			# Validate SMILES column
			if not detect_smiles_column(self.df, column_name):
				raise ValueError(f"Column '{column_name}' does not contain valid SMILES strings")

			self._add_to_undo_stack()
			self._clear_redo_stack()

			if descriptor_set == "basic":
				# Add basic descriptors (MW, LogP, TPSA)
				for smiles_idx, smiles in enumerate(self.df[column_name]):
					if pd.isna(smiles):
						continue

					mol = get_mol_from_smiles(smiles)
					if not mol:
						continue

					descriptors = calculate_basic_descriptors(mol)
					for desc_name, value in descriptors.items():
						desc_col = f"{column_name}_{desc_name}"
						if desc_col not in self.df.columns:
							self.df[desc_col] = None
						self.df.at[smiles_idx, desc_col] = value

			elif descriptor_set == "custom" and custom_descriptors:
				# Add custom descriptors
				# Implementation would go here
				pass

			return self.df

		except Exception as e:
			self.logger.error(f"Error adding descriptors: {e}")
			raise

	def add_fingerprints(
		self, column_name: str, fp_type: str = "morgan", radius: int = 2, n_bits: int = 1024
	) -> pd.DataFrame:
		"""
		Add molecular fingerprints based on SMILES strings.

		Args:
		    column_name: Column containing SMILES strings
		    fp_type: Type of fingerprint (morgan, maccs)
		    radius: Radius for Morgan fingerprints
		    n_bits: Number of bits for fingerprints

		Returns:
		    Updated DataFrame with fingerprint columns

		Raises:
		    ValueError: If fingerprint generation fails
		"""
		try:
			# Validate SMILES column
			if not detect_smiles_column(self.df, column_name):
				raise ValueError(f"Column '{column_name}' does not contain valid SMILES strings")

			self._add_to_undo_stack()
			self._clear_redo_stack()

			if fp_type == "morgan":
				# Generate Morgan fingerprints
				fp_prefix = f"{column_name}_ECFP{radius*2}"

				fp_data = []
				for smiles in self.df[column_name]:
					mol = get_mol_from_smiles(smiles) if pd.notna(smiles) else None
					fp = generate_morgan_fingerprint(mol, radius, n_bits)
					fp_data.append(fp)

				# Create DataFrame with fingerprint bits
				fp_df = pd.DataFrame(fp_data, columns=[f"{fp_prefix}_{i}" for i in range(n_bits)])

				# Concatenate with original DataFrame
				self.df = pd.concat([self.df, fp_df], axis=1)

			elif fp_type == "maccs":
				# Generate MACCS fingerprints
				fp_prefix = f"{column_name}_MACCS"

				fp_data = []
				for smiles in self.df[column_name]:
					mol = get_mol_from_smiles(smiles) if pd.notna(smiles) else None
					fp = generate_maccs_fingerprint(mol)
					fp_data.append(fp)

				# Create DataFrame with fingerprint bits
				fp_df = pd.DataFrame(fp_data, columns=[f"{fp_prefix}_{i}" for i in range(167)])

				# Concatenate with original DataFrame
				self.df = pd.concat([self.df, fp_df], axis=1)

			return self.df

		except Exception as e:
			self.logger.error(f"Error adding fingerprints: {e}")
			raise

	def get_dataframe(self) -> pd.DataFrame:
		"""Get the current DataFrame."""
		return self.df

	def _add_to_undo_stack(self) -> None:
		"""Add current DataFrame to undo stack."""
		self.undo_stack.append(self.df.copy())

	def _clear_redo_stack(self) -> None:
		"""Clear the redo stack."""
		self.redo_stack.clear()

	def reset_to_original(self) -> pd.DataFrame:
		"""
		Reset the DataFrame to its original state when first loaded.

		Returns:
		    Original DataFrame

		Raises:
		    ValueError: If no original DataFrame exists
		"""
		try:
			if not hasattr(self, "original_df") or self.original_df is None:
				raise ValueError("No original data available to reset to")

			# Save current state for undo
			self._add_to_undo_stack()
			self._clear_redo_stack()

			# Reset to original
			self.df = self.original_df.copy()
			self.logger.info("Reset to original data")
			return self.df

		except Exception as e:
			self.logger.error(f"Error resetting to original: {e}")
			raise
