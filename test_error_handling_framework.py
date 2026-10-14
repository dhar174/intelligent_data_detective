#!/usr/bin/env python3
"""
Unit tests for the Error Handling and Validation Framework in IntelligentDataDetective_beta_v3_fixed.ipynb

This file tests the validate_dataframe_exists function and handle_tool_errors decorator
to ensure they work correctly and provide robust error handling for data processing tools.
"""

import logging
import os
from pathlib import Path
import tempfile
import unittest
import pandas as pd

# Import the existing test infrastructure
from idd_registry import DataFrameRegistry, override_global_registry
from idd_tools import handle_tool_errors, validate_dataframe_exists


class TestErrorHandlingFramework(unittest.TestCase):
    """Test the error handling and validation framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = DataFrameRegistry()
        self._ctx = override_global_registry(self.registry)
        self._ctx.__enter__()

        # Mock global registry for functions referencing global_df_registry
        global global_df_registry
        global_df_registry = self.registry

        # Create test DataFrames
        self.test_df = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3, None, 5],
                "string_col": ["a", "b", "c", "d", "e"],
                "mixed_col": [1, "b", 3, None, "e"],
            }
        )
        self.empty_df = pd.DataFrame()

        # Register test DataFrames
        self.df_id = self.registry.register_dataframe(self.test_df, "test_df")
        self.empty_df_id = self.registry.register_dataframe(self.empty_df, "empty_df")

        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = Path(self.temp_dir) / "test.csv"
        self.test_df.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self._ctx.__exit__(None, None, None)
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    validate_dataframe_exists = staticmethod(validate_dataframe_exists)
    handle_tool_errors = staticmethod(handle_tool_errors)

    def test_validate_dataframe_exists_valid_df(self):
        """Test validate_dataframe_exists with valid DataFrame."""
        self.assertTrue(self.validate_dataframe_exists(self.df_id))

    def test_validate_dataframe_exists_empty_df(self):
        """Test validate_dataframe_exists with empty DataFrame."""
        self.assertFalse(self.validate_dataframe_exists(self.empty_df_id))

    def test_validate_dataframe_exists_invalid_inputs(self):
        """Test validate_dataframe_exists with invalid inputs."""
        self.assertFalse(self.validate_dataframe_exists("nonexistent_id"))
        self.assertFalse(self.validate_dataframe_exists(""))
        self.assertFalse(self.validate_dataframe_exists(None))
        self.assertFalse(self.validate_dataframe_exists(123))

    def test_validate_dataframe_exists_from_file(self):
        """Test validate_dataframe_exists loading DataFrame from file."""
        file_df_id = self.registry.register_dataframe(
            df=None, df_id="file_df", raw_path=str(self.test_csv_path)
        )
        self.assertTrue(self.validate_dataframe_exists(file_df_id))

    def test_validate_dataframe_exists_invalid_file(self):
        """Test validate_dataframe_exists with invalid file path."""
        invalid_file_id = self.registry.register_dataframe(
            df=None, df_id="invalid_file", raw_path="/nonexistent/path.csv"
        )
        self.assertFalse(self.validate_dataframe_exists(invalid_file_id))

    def test_handle_tool_errors_successful_execution(self):
        """Test handle_tool_errors decorator with successful function execution."""

        @self.handle_tool_errors
        def get_column_names(df_id: str) -> str:
            df = global_df_registry.get_dataframe(df_id)
            return ", ".join(df.columns.tolist())

        result = get_column_names(self.df_id)
        self.assertEqual(result, "numeric_col, string_col, mixed_col")

    def test_handle_tool_errors_invalid_dataframe(self):
        """Test handle_tool_errors decorator with invalid DataFrame ID."""

        @self.handle_tool_errors
        def get_column_names(df_id: str) -> str:
            df = self.registry.get_dataframe(df_id)
            return ", ".join(df.columns.tolist())

        result = get_column_names("invalid_id")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["operation"], "get_column_names")
        self.assertIn("invalid_id", result["reason"])

    def test_handle_tool_errors_keyerror_handling(self):
        """Test handle_tool_errors decorator with KeyError."""

        @self.handle_tool_errors
        def access_nonexistent_column(df_id: str) -> str:
            df = self.registry.get_dataframe(df_id)
            return str(df["nonexistent_column"].dtype)

        result = access_nonexistent_column(self.df_id)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["operation"], "access_nonexistent_column")
        self.assertIn("A requested column or key does not exist.", result["reason"])

    def test_handle_tool_errors_valueerror_handling(self):
        """Test handle_tool_errors decorator with ValueError."""

        @self.handle_tool_errors
        def cause_value_error(df_id: str) -> str:
            raise ValueError("Test value error")

        result = cause_value_error(self.df_id)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["operation"], "cause_value_error")
        self.assertIn("One or more inputs are invalid.", result["reason"])

    def test_handle_tool_errors_general_exception_handling(self):
        """Test handle_tool_errors decorator with general exception."""

        @self.handle_tool_errors
        def cause_general_error(df_id: str) -> str:
            raise RuntimeError("Test runtime error")

        result = cause_general_error(self.df_id)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["operation"], "cause_general_error")
        self.assertEqual(
            result["reason"], "An unexpected data-processing failure occurred."
        )

    def test_realistic_tool_functions(self):
        """Test realistic tool functions that mirror the notebook implementations."""

        @self.handle_tool_errors
        def get_dataframe_schema(df_id: str) -> tuple[str, dict]:
            """Return a summary of the DataFrame's schema and sample data."""
            df = self.registry.get_dataframe(df_id)
            if df is None:
                raw_path = self.registry.get_raw_path_from_id(df_id)
                if raw_path and os.path.exists(raw_path):
                    df = pd.read_csv(raw_path)
                    self.registry.register_dataframe(df, df_id, raw_path)
                else:
                    return (
                        f"Error: DataFrame with ID '{df_id}' not found or raw path is invalid.",
                        {},
                    )
            schema = {
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(3).to_dict(orient="records"),
            }
            return "", {"schema": schema}

        @self.handle_tool_errors
        def check_missing_values(df_id: str) -> str:
            """Checks for missing values in a pandas DataFrame and returns a summary."""
            df = self.registry.get_dataframe(df_id)
            if df is None:
                raw_path = self.registry.get_raw_path_from_id(df_id)
                df = pd.read_csv(raw_path)
                self.registry.register_dataframe(df, df_id, raw_path)
            if df is None:
                return f"Error: DataFrame with ID '{df_id}' not found."
            missing = df.isnull().sum()
            if missing.sum() == 0:
                return f"No missing values in DataFrame '{df_id}'."
            return missing.to_string()

        @self.handle_tool_errors
        def drop_column(df_id: str, column_name: str) -> str:
            """Drops a specified column from the DataFrame."""
            df = self.registry.get_dataframe(df_id)
            if column_name not in df.columns:
                return f"Error: Column '{column_name}' not found in DataFrame '{df_id}'. Available columns: {list(df.columns)}"
            df.drop(columns=[column_name], inplace=True)
            self.registry.register_dataframe(
                df, df_id, self.registry.get_raw_path_from_id(df_id)
            )
            return "Column dropped successfully. New columns: " + ", ".join(
                df.columns.tolist()
            )

        # Test the functions
        schema_result = get_dataframe_schema(self.df_id)
        self.assertEqual(schema_result[0], "")
        self.assertIn("schema", schema_result[1])
        self.assertEqual(len(schema_result[1]["schema"]["columns"]), 3)

        missing_result = check_missing_values(self.df_id)
        self.assertIn("numeric_col", missing_result)

        drop_result = drop_column(self.df_id, "mixed_col")
        self.assertIn("Column dropped successfully", drop_result)

        # Verify column was dropped
        new_schema = get_dataframe_schema(self.df_id)
        self.assertEqual(len(new_schema[1]["schema"]["columns"]), 2)

    def test_decorator_function_signature_preservation(self):
        """Test that the decorator preserves function signatures and metadata."""

        @self.handle_tool_errors
        def sample_function(df_id: str, param1: int = 5) -> str:
            """Sample function docstring."""
            return f"Function called with df_id={df_id}, param1={param1}"

        # Test that function name and docstring are preserved
        self.assertEqual(sample_function.__name__, "sample_function")
        self.assertEqual(sample_function.__doc__, "Sample function docstring.")

        # Test function still works with original parameters
        result = sample_function(self.df_id, param1=10)
        self.assertIn("param1=10", result)

    def test_error_logging(self):
        """Test that errors are properly logged."""
        with self.assertLogs(level="ERROR") as log:

            @self.handle_tool_errors
            def failing_function(df_id: str) -> str:
                raise RuntimeError("Test logging error")

            result = failing_function(self.df_id)
            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["operation"], "failing_function")
            self.assertEqual(
                result["reason"], "An unexpected data-processing failure occurred."
            )

        # Verify error was logged
        self.assertTrue(any("failing_function" in message for message in log.output))
        self.assertTrue(any("Test logging error" in message for message in log.output))

    def test_integration_with_different_function_signatures(self):
        """Test the decorator works with different function signatures."""

        # Function with df_id as first parameter
        @self.handle_tool_errors
        def func_with_df_id_first(df_id: str, other_param: str) -> str:
            return f"df_id: {df_id}, other: {other_param}"

        # Function with df_id as keyword argument
        @self.handle_tool_errors
        def func_with_df_id_kwarg(other_param: str, df_id: str) -> str:
            return f"other: {other_param}, df_id: {df_id}"

        # Function with no df_id (should work without validation)
        @self.handle_tool_errors
        def func_without_df_id(param1: str, param2: int) -> str:
            return f"param1: {param1}, param2: {param2}"

        # Test all variants
        result1 = func_with_df_id_first(self.df_id, "test")
        self.assertIn("df_id: test_df", result1)

        result2 = func_with_df_id_kwarg("test", df_id=self.df_id)
        self.assertIn("df_id: test_df", result2)

        result3 = func_without_df_id("hello", 42)
        self.assertIn("param1: hello", result3)

        # Test error case with invalid df_id
        error_result = func_with_df_id_first("invalid_id", "test")
        self.assertIsInstance(error_result, dict)
        self.assertEqual(error_result["status"], "error")
        self.assertEqual(error_result["operation"], "func_with_df_id_first")
        self.assertIn("invalid_id", error_result["reason"])


class TestFrameworkDocumentation(unittest.TestCase):
    """Test that the framework works as documented in examples."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = DataFrameRegistry()
        self._ctx = override_global_registry(self.registry)
        self._ctx.__enter__()

        self.test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.df_id = self.registry.register_dataframe(self.test_df, "example_df")

    def tearDown(self):
        """Clean up test fixtures."""
        self._ctx.__exit__(None, None, None)

    validate_dataframe_exists = staticmethod(validate_dataframe_exists)
    handle_tool_errors = staticmethod(handle_tool_errors)

    def test_documentation_example_validation(self):
        """Test the example from validate_dataframe_exists docstring."""
        if self.validate_dataframe_exists("example_df"):
            result = "operations proceeded"
        else:
            result = "operations skipped"

        self.assertEqual(result, "operations proceeded")

    def test_documentation_example_decorator(self):
        """Test the example from handle_tool_errors docstring."""

        @self.handle_tool_errors
        def my_tool(df_id: str) -> str:
            # tool implementation
            return "success"

        result = my_tool(self.df_id)
        self.assertEqual(result, "success")

        # Test error case
        error_result = my_tool("invalid_id")
        self.assertIsInstance(error_result, dict)
        self.assertEqual(error_result["status"], "error")
        self.assertEqual(error_result["operation"], "my_tool")
        self.assertIn("invalid_id", error_result["reason"])


if __name__ == "__main__":
    # Configure logging to see error messages during tests
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("Testing Error Handling and Validation Framework...")
    print("=" * 60)

    # Run all tests
    unittest.main(verbosity=2)
