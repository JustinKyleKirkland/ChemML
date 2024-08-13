import pandas as pd


def one_hot_encode(df, column_name, n_distinct=True):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    one_hot_encoded_df = pd.get_dummies(
        df,
        columns=[column_name],
        prefix=column_name,
        drop_first=not n_distinct,
    )

    for column in one_hot_encoded_df.columns:
        if one_hot_encoded_df[column].dtype == "bool":
            one_hot_encoded_df[column] = one_hot_encoded_df[column].astype(int)

    return one_hot_encoded_df


def validate_csv(df):
    errors = []

    # Check for duplicate column names
    if df.columns.duplicated().any():
        errors.append("Duplicate column names found in the CSV.")

    # Check for null values
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        errors.append(f"Null values found in columns: {', '.join(null_columns)}.")

    return errors


def impute_values(df: pd.DataFrame, column_name: str, method: str = "mean"):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    imputed_value = (
        df[column_name].mean() if method == "mean" else df[column_name].median()
    )

    # Raise an error if the method is invalid
    if method not in ["mean", "median"]:
        raise ValueError("Invalid imputation method")

    df.fillna({column_name: imputed_value}, inplace=True)
    return df
