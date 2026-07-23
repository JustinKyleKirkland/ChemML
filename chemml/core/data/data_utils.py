import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

VALID_IMPUTE_METHODS = ("mean", "median", "knn", "mice")


def one_hot_encode(df, column_name, n_distinct=True):
	"""One-hot encode a categorical column.

	Args:
		df: Input DataFrame.
		column_name: Column to encode.
		n_distinct: If True, create one column per category (N columns). If
			False, drop the first category to avoid collinearity (N-1 columns).

	Returns:
		A new DataFrame with the encoded columns (boolean dummies cast to int).
	"""
	if column_name not in df.columns:
		raise ValueError(f"Column '{column_name}' not found in DataFrame.")

	one_hot_encoded_df = pd.get_dummies(
		df,
		columns=[column_name],
		prefix=column_name,
		drop_first=not n_distinct,
	)

	# Cast boolean dummy columns to int for downstream numeric processing.
	bool_columns = one_hot_encoded_df.select_dtypes(include="bool").columns
	one_hot_encoded_df[bool_columns] = one_hot_encoded_df[bool_columns].astype(int)

	return one_hot_encoded_df


def validate_csv(df):
	"""Return a list of human-readable validation warnings for a DataFrame."""
	errors = []

	if df.empty:
		errors.append("The CSV file contains no data rows.")

	# Check for duplicate column names
	if df.columns.duplicated().any():
		duplicates = df.columns[df.columns.duplicated()].unique().tolist()
		errors.append(f"Duplicate column names found: {', '.join(map(str, duplicates))}.")

	# Check for null values
	null_columns = df.columns[df.isnull().any()].tolist()
	if null_columns:
		errors.append(f"Null values found in columns: {', '.join(map(str, null_columns))}.")

	return errors


def impute_values(df: pd.DataFrame, column_name: str, method: str = "mean"):
	"""Impute missing values in ``column_name`` using the given strategy.

	``mean`` and ``median`` operate only on the target column. ``knn`` and
	``mice`` are multivariate and impute every numeric column jointly, using
	the other numeric columns as context.

	Args:
		df: Input DataFrame (modified in place and also returned).
		column_name: Column to impute.
		method: One of ``mean``, ``median``, ``knn`` or ``mice``.

	Returns:
		The DataFrame with missing values filled.
	"""
	if column_name not in df.columns:
		raise ValueError(f"Column '{column_name}' not found in DataFrame.")

	if method not in VALID_IMPUTE_METHODS:
		raise ValueError(f"Invalid imputation method '{method}'. Expected one of {', '.join(VALID_IMPUTE_METHODS)}.")

	if method in ("mean", "median"):
		if not pd.api.types.is_numeric_dtype(df[column_name]):
			raise ValueError(f"Cannot compute {method} of non-numeric column '{column_name}'.")
		imputed_value = df[column_name].mean() if method == "mean" else df[column_name].median()
		df.fillna({column_name: imputed_value}, inplace=True)
	else:  # knn or mice
		numerical_df = df.select_dtypes(include=["number"])
		if numerical_df.empty:
			raise ValueError("No numerical columns found in DataFrame.")

		if method == "knn":
			n_neighbors = min(5, max(1, len(numerical_df) - 1))
			imputer = KNNImputer(n_neighbors=n_neighbors)
		else:  # mice
			imputer = IterativeImputer(max_iter=10, random_state=0)

		imputed_data = imputer.fit_transform(numerical_df)
		df.loc[:, numerical_df.columns] = imputed_data

	return df
