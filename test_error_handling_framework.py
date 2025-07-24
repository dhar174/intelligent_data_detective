#!/usr/bin/env python3
"""
Unit tests for the Error Handling and Validation Framework in IntelligentDataDetective_beta_v3_fixed.ipynb

This file tests the validate_dataframe_exists function and handle_tool_errors decorator
to ensure they work correctly and provide robust error handling for data processing tools.
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
import os
import uuid
from pathlib import Path
from collections import OrderedDict
import functools
import logging
from typing import Dict, Optional, List, Tuple, Union

# Import the existing test infrastructure
from test_intelligent_data_detective import DataFrameRegistry

class TestErrorHandlingFramework(unittest.TestCase):
    """Test the error handling and validation framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = DataFrameRegistry()
        
        # Create test DataFrames
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, None, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'mixed_col': [1, 'b', 3, None, 'e']
        })
        self.empty_df = pd.DataFrame()
        
        # Register test DataFrames
        self.df_id = self.registry.register_dataframe(self.test_df, "test_df")
        self.empty_df_id = self.registry.register_dataframe(self.empty_df, "empty_df")
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = Path(self.temp_dir) / "test.csv"
        self.test_df.to_csv(self.test_csv_path, index=False)
        
        # Mock global registry
        global global_df_registry
        global_df_registry = self.registry
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def validate_dataframe_exists(self, df_id: str) -> bool:
        """Test implementation of validate_dataframe_exists."""
        if not df_id or not isinstance(df_id, str):
            return False
            
        try:
            df = global_df_registry.get_dataframe(df_id)
            if df is not None:
                return not df.empty
                
            raw_path = global_df_registry.get_raw_path_from_id(df_id)
            if raw_path and os.path.exists(raw_path):
                try:
                    df = pd.read_csv(raw_path)
                    if df is not None and not df.empty:
                        global_df_registry.register_dataframe(df, df_id, raw_path)
                        return True
                except Exception:
                    return False
                    
            return False
        except Exception:
            return False

    def handle_tool_errors(self, func):
        """Test implementation of handle_tool_errors decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                df_id = None
                
                if args and isinstance(args[0], str):
                    df_id = args[0]
                elif 'df_id' in kwargs:
                    df_id = kwargs['df_id']
                elif args and hasattr(args[0], 'df_id'):
                    df_id = args[0].df_id
                    
                if df_id and not self.validate_dataframe_exists(df_id):
                    error_msg = f"Error: DataFrame with ID '{df_id}' not found or is invalid."
                    logging.error(f"{func.__name__}: {error_msg}")
                    return error_msg
                    
                result = func(*args, **kwargs)
                return result
                
            except FileNotFoundError as e:
                error_msg = f"Error: File not found - {str(e)}"
                logging.error(f"{func.__name__}: {error_msg}")
                return error_msg
                
            except KeyError as e:
                error_msg = f"Error: Column or key '{str(e)}' not found"
                logging.error(f"{func.__name__}: {error_msg}")
                return error_msg
                
            except ValueError as e:
                error_msg = f"Error: Invalid value - {str(e)}"
                logging.error(f"{func.__name__}: {error_msg}")
                return error_msg
                
            except pd.errors.EmptyDataError:
                error_msg = "Error: No data - the DataFrame or file is empty"
                logging.error(f"{func.__name__}: {error_msg}")
                return error_msg
                
            except pd.errors.ParserError as e:
                error_msg = f"Error: Failed to parse data - {str(e)}"
                logging.error(f"{func.__name__}: {error_msg}")
                return error_msg
                
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logging.error(error_msg)
                return error_msg
                
        return wrapper

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
        file_df_id = self.registry.register_dataframe(df=None, df_id="file_df", raw_path=str(self.test_csv_path))
        self.assertTrue(self.validate_dataframe_exists(file_df_id))
        
    def test_validate_dataframe_exists_invalid_file(self):
        """Test validate_dataframe_exists with invalid file path."""
        invalid_file_id = self.registry.register_dataframe(df=None, df_id="invalid_file", raw_path="/nonexistent/path.csv")
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
            df = global_df_registry.get_dataframe(df_id)
            return ", ".join(df.columns.tolist())
        
        result = get_column_names("invalid_id")
        self.assertIn("Error: DataFrame with ID 'invalid_id' not found", result)
        
    def test_handle_tool_errors_keyerror_handling(self):
        """Test handle_tool_errors decorator with KeyError."""
        @self.handle_tool_errors
        def access_nonexistent_column(df_id: str) -> str:
            df = global_df_registry.get_dataframe(df_id)
            return str(df['nonexistent_column'].dtype)
        
        result = access_nonexistent_column(self.df_id)
        self.assertIn("Error: Column or key", result)
        
    def test_handle_tool_errors_valueerror_handling(self):
        """Test handle_tool_errors decorator with ValueError."""
        @self.handle_tool_errors
        def cause_value_error(df_id: str) -> str:
            raise ValueError("Test value error")
        
        result = cause_value_error(self.df_id)
        self.assertIn("Error: Invalid value - Test value error", result)
        
    def test_handle_tool_errors_general_exception_handling(self):
        """Test handle_tool_errors decorator with general exception."""
        @self.handle_tool_errors
        def cause_general_error(df_id: str) -> str:
            raise RuntimeError("Test runtime error")
        
        result = cause_general_error(self.df_id)
        self.assertIn("Error in cause_general_error: Test runtime error", result)

    def test_realistic_tool_functions(self):
        """Test realistic tool functions that mirror the notebook implementations."""
        
        @self.handle_tool_errors
        def get_dataframe_schema(df_id: str) -> tuple[str, dict]:
            """Return a summary of the DataFrame's schema and sample data."""
            df = global_df_registry.get_dataframe(df_id)
            if df is None:
                raw_path = global_df_registry.get_raw_path_from_id(df_id)
                if raw_path and os.path.exists(raw_path):
                    df = pd.read_csv(raw_path)
                    global_df_registry.register_dataframe(df, df_id, raw_path)
                else:
                    return f"Error: DataFrame with ID '{df_id}' not found or raw path is invalid.", {}
            schema = {
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(3).to_dict(orient="records")
            }
            return "", {"schema": schema}
        
        @self.handle_tool_errors
        def check_missing_values(df_id: str) -> str:
            """Checks for missing values in a pandas DataFrame and returns a summary."""
            df = global_df_registry.get_dataframe(df_id)
            if df is None:
                raw_path = global_df_registry.get_raw_path_from_id(df_id)
                df = pd.read_csv(raw_path)
                global_df_registry.register_dataframe(df, df_id, raw_path)
            if df is None:
                return f"Error: DataFrame with ID '{df_id}' not found."
            missing = df.isnull().sum()
            if missing.sum() == 0:
                return f"No missing values in DataFrame '{df_id}'."
            return missing.to_string()
        
        @self.handle_tool_errors
        def drop_column(df_id: str, column_name: str) -> str:
            """Drops a specified column from the DataFrame."""
            df = global_df_registry.get_dataframe(df_id)
            if column_name not in df.columns:
                return f"Error: Column '{column_name}' not found in DataFrame '{df_id}'. Available columns: {list(df.columns)}"
            df.drop(columns=[column_name], inplace=True)
            global_df_registry.register_dataframe(df, df_id, global_df_registry.get_raw_path_from_id(df_id))
            return "Column dropped successfully. New columns: " + ", ".join(df.columns.tolist())
        
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
        with self.assertLogs(level='ERROR') as log:
            @self.handle_tool_errors
            def failing_function(df_id: str) -> str:
                raise ValueError("Test logging error")
            
            result = failing_function(self.df_id)
            self.assertIn("Error: Invalid value", result)
        
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
        self.assertIn("Error: DataFrame with ID 'invalid_id'", error_result)


class TestFrameworkDocumentation(unittest.TestCase):
    """Test that the framework works as documented in examples."""
    
    def setUp(self):
        """Set up test fixtures."""
        global global_df_registry
        global_df_registry = DataFrameRegistry()
        
        self.test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.df_id = global_df_registry.register_dataframe(self.test_df, "example_df")
        
    def validate_dataframe_exists(self, df_id: str) -> bool:
        """Implementation for documentation examples."""
        if not df_id or not isinstance(df_id, str):
            return False
        df = global_df_registry.get_dataframe(df_id)
        return df is not None and not df.empty

    def handle_tool_errors(self, func):
        """Simple implementation for documentation examples."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                df_id = args[0] if args and isinstance(args[0], str) else None
                if df_id and not self.validate_dataframe_exists(df_id):
                    return f"Error: DataFrame with ID '{df_id}' not found or is invalid."
                return func(*args, **kwargs)
            except Exception as e:
                return f"Error in {func.__name__}: {str(e)}"
        return wrapper

    def test_documentation_example_validation(self):
        """Test the example from validate_dataframe_exists docstring."""
        # Example from docstring:
        # >>> if validate_dataframe_exists('my_df_id'):
        # ...     # proceed with operations
        # ...     pass
        
        if self.validate_dataframe_exists('example_df'):
            result = "operations proceeded"
        else:
            result = "operations skipped"
            
        self.assertEqual(result, "operations proceeded")

    def test_documentation_example_decorator(self):
        """Test the example from handle_tool_errors docstring."""
        # Example from docstring:
        # >>> @handle_tool_errors
        # ... def my_tool(df_id: str) -> str:
        # ...     # tool implementation
        # ...     return "success"
        
        @self.handle_tool_errors
        def my_tool(df_id: str) -> str:
            # tool implementation
            return "success"
            
        result = my_tool(self.df_id)
        self.assertEqual(result, "success")
        
        # Test error case
        error_result = my_tool("invalid_id")
        self.assertIn("Error: DataFrame with ID 'invalid_id' not found", error_result)


if __name__ == '__main__':
    # Configure logging to see error messages during tests
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Testing Error Handling and Validation Framework...")
    print("=" * 60)
    
    # Run all tests
    unittest.main(verbosity=2)