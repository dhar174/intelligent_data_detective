# Error Handling and Validation Framework

## Overview

The Error Handling and Validation Framework provides robust error handling and runtime validation for the IntelligentDataDetective notebook. It consists of two main components:

1. **`validate_dataframe_exists(df_id: str) -> bool`** - Validates DataFrame existence and validity
2. **`@handle_tool_errors`** - Decorator for consistent error handling across all tool functions

## Core Functions

### validate_dataframe_exists

Validates the existence and validity of a DataFrame by its ID before any operations.

**Signature:**
```python
def validate_dataframe_exists(df_id: str) -> bool
```

**Parameters:**
- `df_id` (str): The ID of the DataFrame to validate

**Returns:**
- `bool`: True if DataFrame exists and is valid, False otherwise

**Examples:**
```python
# Basic validation
if validate_dataframe_exists('my_df_id'):
    # proceed with operations
    df = global_df_registry.get_dataframe('my_df_id')
    print(f"DataFrame has {len(df)} rows")
else:
    print("DataFrame not found or invalid")

# Use in conditional processing
df_ids = ['df1', 'df2', 'df3']
valid_dfs = [df_id for df_id in df_ids if validate_dataframe_exists(df_id)]
print(f"Found {len(valid_dfs)} valid DataFrames")
```

### handle_tool_errors

Decorator that provides standardized error handling, DataFrame validation, and user-friendly error messages for all tool functions.

**Usage:**
```python
@handle_tool_errors
def my_tool(df_id: str) -> str:
    # tool implementation
    df = global_df_registry.get_dataframe(df_id)
    return f"Success: DataFrame has {len(df)} rows"
```

**Features:**
- Automatic DataFrame validation
- Comprehensive exception handling
- Consistent error message formatting
- Logging integration
- Function metadata preservation

## Integration Examples

### Basic Tool Function

```python
@tool("GetColumnNames")
@handle_tool_errors
def get_column_names(df_id: str) -> str:
    """Get the names of the columns in the DataFrame."""
    df = global_df_registry.get_dataframe(df_id)
    if df.empty:
        return f"Warning: DataFrame '{df_id}' is empty. No columns available."
    return ", ".join(df.columns.tolist())
```

### Tool Function with Complex Parameters

```python
@tool("DropColumn")
@handle_tool_errors
def drop_column(df_id: str, column_name: str) -> str:
    """Drop a specified column from the DataFrame."""
    df = global_df_registry.get_dataframe(df_id)
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found. Available: {list(df.columns)}"
    
    df.drop(columns=[column_name], inplace=True)
    global_df_registry.register_dataframe(df, df_id, global_df_registry.get_raw_path_from_id(df_id))
    return f"Column '{column_name}' dropped successfully."
```

### Tool Function with File I/O

```python
@tool("ExportDataFrame")
@handle_tool_errors
def export_dataframe(df_id: str, file_name: str, file_format: str) -> str:
    """Export a DataFrame to a file."""
    df = global_df_registry.get_dataframe(df_id)
    
    if file_format.lower() == 'csv':
        df.to_csv(file_name, index=False)
    elif file_format.lower() == 'json':
        df.to_json(file_name, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return f"DataFrame exported to {file_name} as {file_format.upper()}"
```

## Error Types Handled

The framework automatically handles these error types:

1. **DataFrame Not Found**: Invalid or missing DataFrame IDs
2. **FileNotFoundError**: Missing data files or export paths
3. **KeyError**: Non-existent columns or keys
4. **ValueError**: Invalid parameters or data values
5. **EmptyDataError**: Empty datasets or files
6. **ParserError**: Data parsing/format issues
7. **General Exceptions**: Unexpected runtime errors

## Error Message Format

All error messages follow a consistent format:
- DataFrame validation: `"Error: DataFrame with ID 'df_id' not found or is invalid."`
- File errors: `"Error: File not found - [details]"`
- Column errors: `"Error: Column or key 'column_name' not found"`
- Value errors: `"Error: Invalid value - [details]"`
- General errors: `"Error in function_name: [details]"`

## Logging Integration

The framework automatically logs all errors with timestamps:

```
2025-07-24 18:16:51,365 - ERROR - get_column_names: Error: DataFrame with ID 'invalid_id' not found or is invalid.
```

## Testing

Comprehensive test coverage is provided in `test_error_handling_framework.py`:

- DataFrame validation scenarios
- Decorator functionality
- Error handling for all exception types  
- Integration with realistic tool functions
- Documentation examples verification

**Run tests:**
```bash
python3 test_error_handling_framework.py
```

## Migration Guide

To update existing tool functions:

### Before (Old Pattern):
```python
@tool("MyTool")
def my_tool(df_id: str) -> str:
    try:
        df = global_df_registry.get_dataframe(df_id)
        if df is None:
            raw_path = global_df_registry.get_raw_path_from_id(df_id)
            df = pd.read_csv(raw_path)
            global_df_registry.register_dataframe(df, df_id, raw_path)
        if df is None:
            return f"Error: DataFrame with ID '{df_id}' not found."
        # ... tool logic ...
        return "Success"
    except FileNotFoundError as e:
        return f"Error loading DataFrame: {e}"
    except Exception as e:
        return f"Error in my_tool: {e}"
```

### After (New Pattern):
```python
@tool("MyTool")
@handle_tool_errors
def my_tool(df_id: str) -> str:
    df = global_df_registry.get_dataframe(df_id)
    if df is None:
        raw_path = global_df_registry.get_raw_path_from_id(df_id)
        df = pd.read_csv(raw_path)
        global_df_registry.register_dataframe(df, df_id, raw_path)
    # ... tool logic ...
    return "Success"
```

## Benefits

1. **Consistency**: Standardized error handling across all tools
2. **Reliability**: Comprehensive exception coverage
3. **Debugging**: Clear error messages and logging
4. **Maintenance**: Reduced code duplication
5. **User Experience**: Friendly error messages for end users
6. **Production Ready**: Robust error handling for deployment scenarios

## Implementation Status

✅ **Completed:**
- `validate_dataframe_exists` function implemented and tested
- `handle_tool_errors` decorator implemented and tested  
- Framework integrated into notebook
- Comprehensive test suite created
- Documentation and examples provided

🔄 **Next Steps:**
- Apply `@handle_tool_errors` decorator to all existing tool functions
- Update tool function implementations to use the new framework
- Add framework integration to notebook pipeline workflows