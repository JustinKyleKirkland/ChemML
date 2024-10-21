import unittest

import pandas as pd

from utils.data_utils import impute_values, one_hot_encode, validate_csv


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Category": ["A", "B", "C", "D"],
                "Value": [1, 2, None, 4],
            }
        )

    def test_one_hot_encode_with_n_distinct(self):
        result = one_hot_encode(self.df, "Category", n_distinct=False)
        expected = pd.DataFrame(
            {
                "Value": [1, 2, None, 4],
                "Category_B": [0, 1, 0, 0],
                "Category_C": [0, 0, 1, 0],
                "Category_D": [0, 0, 0, 1],
            }
        )
        print(result)
        pd.testing.assert_frame_equal(result, expected)

    def test_one_hot_encode_without_n_distinct(self):
        result = one_hot_encode(self.df, "Category", n_distinct=True)
        expected = pd.DataFrame(
            {
                "Value": [1, 2, None, 4],
                "Category_A": [1, 0, 0, 0],
                "Category_B": [0, 1, 0, 0],
                "Category_C": [0, 0, 1, 0],
                "Category_D": [0, 0, 0, 1],
            }
        )
        print(result)
        pd.testing.assert_frame_equal(result, expected)

    def test_one_hot_encode_column_not_found(self):
        with self.assertRaises(ValueError):
            one_hot_encode(self.df, "NonExistentColumn")

    def test_validate_csv_with_nulls(self):
        errors = validate_csv(self.df)
        self.assertIn("Null values found in columns: Value.", errors)

    def test_validate_csv_without_errors(self):
        df_no_errors = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        errors = validate_csv(df_no_errors)
        self.assertEqual(errors, [])

    def test_impute_values_with_mean(self):
        result = impute_values(self.df, "Value", method="mean")
        expected = pd.DataFrame({"Category": ["A", "B", "C", "D"], "Value": [1, 2, 2.333333, 4]})
        pd.testing.assert_frame_equal(result, expected)

    def test_impute_values_with_median(self):
        result = impute_values(self.df, "Value", method="median")
        expected = pd.DataFrame({"Category": ["A", "B", "C", "D"], "Value": [1, 2, 2.0, 4]})
        pd.testing.assert_frame_equal(result, expected)

    def test_impute_values_column_not_found(self):
        with self.assertRaises(ValueError):
            impute_values(self.df, "NonExistentColumn", method="mean")

    def test_impute_values_invalid_method(self):
        with self.assertRaises(ValueError):
            impute_values(self.df, "Value", method="invalid_method")


if __name__ == "__main__":
    unittest.main()
