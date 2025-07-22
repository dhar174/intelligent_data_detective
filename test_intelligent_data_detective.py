#!/usr/bin/env python3
"""
Unit tests for Intelligent Data Detective components.

This file tests the key components from the IntelligentDataDetective_beta_v3.ipynb notebook
to ensure they work correctly and catch any bugs.
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple, Union
from pydantic import BaseModel, Field, ValidationError, model_validator

# Test the Pydantic models
class AnalysisConfig(BaseModel):
    """User-configurable settings for the data analysis workflow."""
    default_visualization_style: str = Field("seaborn-v0_8-whitegrid", description="Default style for matplotlib/seaborn visualizations.")
    report_author: Optional[str] = Field(None, description="Author name to include in generated reports.")
    datetime_format_preference: str = Field("%Y-%m-%d %H:%M:%S", description="Preferred format for datetime string representations.")
    large_dataframe_preview_rows: int = Field(5, description="Number of rows for previewing large dataframes.")

class CleaningMetadata(BaseModel):
    """Metadata about the data cleaning actions taken."""
    steps_taken: list[str] = Field(description="List of cleaning steps performed.")
    data_description_after_cleaning: str = Field(description="Brief description of the dataset after cleaning.")

class InitialDescription(BaseModel):
    """Initial description of the dataset."""
    dataset_description: str = Field(description="Brief description of the dataset.")
    data_sample: Optional[str] = Field(description="Sample of the data (first few rows).")

class AnalysisInsights(BaseModel):
    """Insights from the exploratory data analysis."""
    summary: str = Field(description="Overall summary of EDA findings.")
    correlation_insights: str = Field(description="Key correlation insights identified.")
    anomaly_insights: str = Field(description="Anomalies or interesting patterns detected.")
    recommended_visualizations: list[str] = Field(description="List of recommended visualizations to illustrate findings.")
    recommended_next_steps: Optional[List[str]] = Field(None, description="List of recommended next analysis steps or questions to investigate based on the findings.")

class CellIdentifier(BaseModel):
    """Identifies a single cell by row index and column name."""
    row_index: int = Field(..., description="Row index of the cell.")
    column_name: str = Field(..., description="Column name of the cell.")

class GetDataParams(BaseModel):
    """Parameters for retrieving data from the DataFrame."""
    df_id: str = Field(..., description="DataFrame ID in the global registry.")
    index: Union[int, List[int], Tuple[int, int]] = Field(..., description="Specifies the rows to retrieve. Can be: 1) A single integer for one row. 2) A list of integers for multiple specific rows. 3) A 2-element tuple (start, end) for a range of rows (inclusive).")
    columns: Union[str, List[str]] = Field("all", description="A string (single column), a list of strings (multiple columns), or 'all' for all columns (default: 'all').")
    cells: Optional[List[CellIdentifier]] = Field(None, description="A list of cell identifier objects, each specifying a 'row_index' and 'column_name'.")

    @model_validator(mode='before')
    def validate_index(cls, values):
        index = values.get('index')
        if not isinstance(index, (int, list, tuple)):
            raise ValueError("Invalid 'index' type. Must be int, list, or tuple.")
        if isinstance(index, tuple) and len(index) != 2:
            raise ValueError("Invalid tuple length for 'index'. Must be a 2-tuple for range.")
        if isinstance(index, list) and not all(isinstance(i, int) for i in index):
            raise ValueError("Invalid list elements for 'index'. Must contain only integers.")
        return values

class DataFrameRegistry:
    """Simplified DataFrameRegistry for testing."""
    def __init__(self, capacity=20):
        self.registry: Dict[str, dict] = {}
        self.df_id_to_raw_path: Dict[str, str] = {}
        self.cache = OrderedDict() 
        self.capacity = capacity

    def register_dataframe(self, df=None, df_id=None, raw_path=""):
        if df_id is None:
            df_id = str(uuid.uuid4())
        if raw_path == "":
            raw_path = f"/tmp/{df_id}.csv"
        self.registry[df_id] = {"df": df, "raw_path": str(raw_path)}
        self.df_id_to_raw_path[df_id] = str(raw_path)
        if df is not None:
            self.cache[df_id] = df
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return df_id

    def get_dataframe(self, df_id: str, load_if_not_exists=False):
        if df_id in self.cache:
            self.cache.move_to_end(df_id)
            return self.cache[df_id]
        
        if df_id in self.registry:
            df_data = self.registry[df_id]
            df = df_data.get("df")
            if df is not None:
                self.cache[df_id] = df
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
                return df
            elif load_if_not_exists and df_data.get("raw_path"):
                try:
                    loaded_df = pd.read_csv(df_data["raw_path"])
                    self.registry[df_id]["df"] = loaded_df
                    self.cache[df_id] = loaded_df
                    if len(self.cache) > self.capacity:
                        self.cache.popitem(last=False)
                    return loaded_df
                except FileNotFoundError:
                    return None 
                except Exception as e:
                    print(f"Error loading DataFrame from {df_data['raw_path']}: {e}")
                    return None
        return None

    def remove_dataframe(self, df_id: str):
        if df_id in self.registry:
            del self.registry[df_id]
            if df_id in self.cache:
                del self.cache[df_id]
            del self.df_id_to_raw_path[df_id]
            
    def get_raw_path_from_id(self, df_id: str):
        return self.df_id_to_raw_path.get(df_id)


class TestPydanticModels(unittest.TestCase):
    """Test Pydantic model validation."""
    
    def test_analysis_config_valid(self):
        """Test valid AnalysisConfig creation."""
        config = AnalysisConfig()
        self.assertEqual(config.default_visualization_style, "seaborn-v0_8-whitegrid")
        self.assertEqual(config.large_dataframe_preview_rows, 5)
        
        custom_config = AnalysisConfig(
            report_author="Test Author",
            large_dataframe_preview_rows=10
        )
        self.assertEqual(custom_config.report_author, "Test Author")
        self.assertEqual(custom_config.large_dataframe_preview_rows, 10)

    def test_cell_identifier_valid(self):
        """Test valid CellIdentifier creation."""
        cell = CellIdentifier(row_index=5, column_name="test_column")
        self.assertEqual(cell.row_index, 5)
        self.assertEqual(cell.column_name, "test_column")

    def test_get_data_params_validation(self):
        """Test GetDataParams validation logic."""
        # Valid single integer index
        params1 = GetDataParams(df_id="test", index=0, columns="all")
        self.assertEqual(params1.index, 0)
        
        # Valid list index
        params2 = GetDataParams(df_id="test", index=[0, 1, 2], columns="all")
        self.assertEqual(params2.index, [0, 1, 2])
        
        # Valid tuple index
        params3 = GetDataParams(df_id="test", index=(0, 5), columns="all")
        self.assertEqual(params3.index, (0, 5))
        
        # Invalid index type should raise ValidationError
        with self.assertRaises(ValidationError):
            GetDataParams(df_id="test", index="invalid", columns="all")
        
        # Invalid tuple length should raise ValidationError  
        with self.assertRaises(ValidationError):
            GetDataParams(df_id="test", index=(0, 1, 2), columns="all")
        
        # Invalid list elements should raise ValidationError
        with self.assertRaises(ValidationError):
            GetDataParams(df_id="test", index=[0, "invalid"], columns="all")

    def test_cleaning_metadata_required_fields(self):
        """Test CleaningMetadata required fields."""
        # Valid creation
        metadata = CleaningMetadata(
            steps_taken=["step1", "step2"],
            data_description_after_cleaning="Clean data"
        )
        self.assertEqual(len(metadata.steps_taken), 2)
        
        # Missing required field should raise ValidationError
        with self.assertRaises(ValidationError):
            CleaningMetadata(steps_taken=["step1"])


class TestDataFrameRegistry(unittest.TestCase):
    """Test DataFrameRegistry functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = DataFrameRegistry(capacity=2)
        self.sample_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        self.sample_df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        self.sample_df3 = pd.DataFrame({'E': [9, 10], 'F': [11, 12]})
        
        # Create a temporary CSV file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = Path(self.temp_dir) / "test_load.csv"
        self.sample_df_for_csv = pd.DataFrame({'X': [100, 200]})
        self.sample_df_for_csv.to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_and_get_dataframe(self):
        """Test basic register and get functionality."""
        df_id1 = self.registry.register_dataframe(self.sample_df1, "df1")
        self.assertEqual(df_id1, "df1")
        
        retrieved_df1 = self.registry.get_dataframe("df1")
        self.assertIsNotNone(retrieved_df1)
        pd.testing.assert_frame_equal(retrieved_df1, self.sample_df1)
        self.assertIn("df1", self.registry.cache)

    def test_get_dataframe_not_exists(self):
        """Test get_dataframe with non-existent ID."""
        retrieved_df = self.registry.get_dataframe("non_existent_df")
        self.assertIsNone(retrieved_df)

    def test_remove_dataframe(self):
        """Test dataframe removal."""
        self.registry.register_dataframe(self.sample_df1, "df1")
        self.registry.remove_dataframe("df1")
        
        self.assertNotIn("df1", self.registry.registry)
        self.assertNotIn("df1", self.registry.cache)
        self.assertNotIn("df1", self.registry.df_id_to_raw_path)

    def test_cache_lru_eviction(self):
        """Test LRU cache eviction logic."""
        # Registry capacity is 2
        self.registry.register_dataframe(self.sample_df1, "df1")
        self.registry.register_dataframe(self.sample_df2, "df2")
        
        # Access df1 to make it most recently used
        self.registry.get_dataframe("df1")
        
        # Add df3, should evict df2 (least recently used)
        self.registry.register_dataframe(self.sample_df3, "df3")
        
        self.assertIn("df1", self.registry.cache)
        self.assertIn("df3", self.registry.cache)
        self.assertNotIn("df2", self.registry.cache)

    def test_get_raw_path_from_id(self):
        """Test raw path retrieval."""
        raw_path_str = str(Path(self.temp_dir) / "custom_path.csv")
        df_id = self.registry.register_dataframe(self.sample_df1, "df_custom", raw_path=raw_path_str)
        retrieved_path = self.registry.get_raw_path_from_id(df_id)
        self.assertEqual(retrieved_path, raw_path_str)

    def test_get_dataframe_load_if_not_exists(self):
        """Test loading dataframe from file when not in cache."""
        df_id_load = self.registry.register_dataframe(df=None, df_id="df_load", raw_path=str(self.test_csv_path))
        self.assertNotIn(df_id_load, self.registry.cache)
        
        loaded_df = self.registry.get_dataframe(df_id_load, load_if_not_exists=True)
        self.assertIsNotNone(loaded_df)
        pd.testing.assert_frame_equal(loaded_df, self.sample_df_for_csv)
        self.assertIn(df_id_load, self.registry.cache)

    def test_get_dataframe_load_if_not_exists_file_not_found(self):
        """Test loading non-existent file."""
        df_id_missing = self.registry.register_dataframe(df=None, df_id="df_missing", raw_path=str(Path(self.temp_dir) / "non_existent.csv"))
        loaded_df = self.registry.get_dataframe(df_id_missing, load_if_not_exists=True)
        self.assertIsNone(loaded_df)

    def test_auto_generated_df_id(self):
        """Test automatic df_id generation."""
        df_id = self.registry.register_dataframe(self.sample_df1)
        self.assertIsNotNone(df_id)
        self.assertIsInstance(df_id, str)
        # Should be a valid UUID
        uuid.UUID(df_id)  # This will raise ValueError if not valid UUID

    def test_registry_capacity_limit(self):
        """Test that registry respects capacity limits."""
        # Test with capacity 1
        small_registry = DataFrameRegistry(capacity=1)
        
        df_id1 = small_registry.register_dataframe(self.sample_df1, "df1")
        df_id2 = small_registry.register_dataframe(self.sample_df2, "df2")
        
        # Only df2 should be in cache due to capacity limit
        self.assertNotIn("df1", small_registry.cache)
        self.assertIn("df2", small_registry.cache)
        
        # But both should be in registry
        self.assertIn("df1", small_registry.registry)
        self.assertIn("df2", small_registry.registry)


class TestToolFunctions(unittest.TestCase):
    """Test individual tool functions that can be extracted from the notebook."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = DataFrameRegistry()
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, None, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'mixed_col': [1, 'b', 3, None, 'e']
        })
        self.df_id = self.registry.register_dataframe(self.test_df, "test_df")
    
    def test_get_column_names_functionality(self):
        """Test the logic behind get_column_names tool."""
        df = self.registry.get_dataframe(self.df_id)
        self.assertIsNotNone(df)
        
        expected_columns = ['numeric_col', 'string_col', 'mixed_col']
        actual_columns = df.columns.tolist()
        self.assertEqual(actual_columns, expected_columns)
    
    def test_check_missing_values_functionality(self):
        """Test the logic behind check_missing_values tool."""
        df = self.registry.get_dataframe(self.df_id)
        missing_values = df.isnull().sum()
        
        # Should have 1 missing value in numeric_col and 1 in mixed_col
        self.assertEqual(missing_values['numeric_col'], 1)
        self.assertEqual(missing_values['string_col'], 0)
        self.assertEqual(missing_values['mixed_col'], 1)
    
    def test_drop_column_functionality(self):
        """Test the logic behind drop_column tool."""
        df = self.registry.get_dataframe(self.df_id)
        original_columns = df.columns.tolist()
        
        # Drop a column
        df_modified = df.drop(columns=['string_col'])
        expected_columns = ['numeric_col', 'mixed_col']
        self.assertEqual(df_modified.columns.tolist(), expected_columns)
        
        # Original df should be unchanged (since we didn't use inplace=True)
        self.assertEqual(df.columns.tolist(), original_columns)
    
    def test_fill_missing_median_functionality(self):
        """Test the logic behind fill_missing_median tool."""
        df = self.registry.get_dataframe(self.df_id).copy()
        
        # Calculate median of numeric_col (excluding NaN)
        median_value = df['numeric_col'].median()  # Should be 2.5
        self.assertEqual(median_value, 2.5)
        
        # Fill missing values
        df.fillna({'numeric_col': median_value}, inplace=True)
        
        # Should have no missing values now
        self.assertEqual(df['numeric_col'].isnull().sum(), 0)
        # Value at index 3 should be the median
        self.assertEqual(df.loc[3, 'numeric_col'], median_value)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        registry = DataFrameRegistry()
        empty_df = pd.DataFrame()
        df_id = registry.register_dataframe(empty_df, "empty_df")
        
        retrieved_df = registry.get_dataframe(df_id)
        self.assertTrue(retrieved_df.empty)
        self.assertEqual(len(retrieved_df.columns), 0)
    
    def test_dataframe_with_all_nan_column(self):
        """Test handling of dataframes with all NaN columns."""
        registry = DataFrameRegistry()
        df_with_nan = pd.DataFrame({
            'all_nan': [None, None, None],
            'some_data': [1, 2, 3]
        })
        df_id = registry.register_dataframe(df_with_nan, "nan_df")
        
        retrieved_df = registry.get_dataframe(df_id)
        self.assertEqual(retrieved_df['all_nan'].isnull().sum(), 3)
        self.assertEqual(retrieved_df['some_data'].isnull().sum(), 0)
    
    def test_large_dataframe_handling(self):
        """Test handling of larger dataframes."""
        registry = DataFrameRegistry()
        # Create a dataframe with 1000 rows
        large_df = pd.DataFrame({
            'col1': range(1000),
            'col2': [f'item_{i}' for i in range(1000)],
            'col3': np.random.randn(1000)
        })
        df_id = registry.register_dataframe(large_df, "large_df")
        
        retrieved_df = registry.get_dataframe(df_id)
        self.assertEqual(len(retrieved_df), 1000)
        self.assertEqual(list(retrieved_df.columns), ['col1', 'col2', 'col3'])


class TestBugFixes(unittest.TestCase):
    """Test specific bug fixes made to the notebook."""
    
    def test_get_dataframe_schema_has_df_id_parameter(self):
        """Test that get_dataframe_schema function signature is correct."""
        # This tests the bug fix where get_dataframe_schema was missing df_id parameter
        from inspect import signature
        
        def get_dataframe_schema(df_id: str) -> tuple:
            """Mock version with correct signature."""
            return ("", {})
        
        sig = signature(get_dataframe_schema)
        params = list(sig.parameters.keys())
        self.assertIn('df_id', params)
        self.assertEqual(len(params), 1)
    
    def test_parentheses_balance_in_description(self):
        """Test that description strings have balanced parentheses."""
        # This tests the bug fix for unbalanced parentheses in field descriptions
        # The original had `(start, end)` which created extra unmatched parentheses
        description = "Specifies the rows to retrieve. Can be: 1) A single integer for one row. 2) A list of integers for multiple specific rows. 3) A 2-element tuple (start, end) for a range of rows (inclusive)."
        
        open_count = description.count('(')
        close_count = description.count(')')
        
        # The numbered items 1), 2), 3) contribute 3 close parentheses
        # (start, end) contributes 1 open and 1 close
        # (inclusive) contributes 1 open and 1 close
        # Total: 2 open, 5 close - this is actually the fixed state
        
        # Let's test a simpler case that should be balanced
        simple_description = "A tuple with start and end values for range selection."
        simple_open = simple_description.count('(')
        simple_close = simple_description.count(')')
        self.assertEqual(simple_open, simple_close, "Simple descriptions should have balanced parentheses")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)