"""
Data validation utilities for Arrow/Parquet datasets.

Provides shared validation functions for checking data format compatibility
and ensuring optimal zero-copy loading performance.

Usage:
    from ava.data.validation import validate_arrow_data_format

    # In dataset __init__:
    validate_arrow_data_format(
        table=arrow_table,
        file_name=file_path.name,
        input_columns=['input_ids', 'token_ids', 'text'],
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pyarrow as pa

logger = logging.getLogger(__name__)


def validate_arrow_data_format(
    table: pa.Table,
    file_name: str,
    input_columns: Optional[List[str]] = None,
) -> bool:
    """
    Validate Arrow table format for zero-copy compatibility.

    Checks that:
    1. Required input column exists
    2. Column types support efficient zero-copy access (fixed-width arrays)

    Logs warnings if variable-length data is detected (which causes 100-1000x
    slowdown due to fallback to slow .as_py() calls).

    Args:
        table: PyArrow table to validate
        file_name: Name of the file (for logging)
        input_columns: List of possible input column names to check
                      (defaults to ['input_ids', 'token_ids', 'text'])

    Returns:
        True if validation passed, False if issues were found

    This is a MEDIUM-risk optimization that prevents silent performance degradation.
    """
    if input_columns is None:
        input_columns = ['input_ids', 'token_ids', 'text']

    if table is None or len(table) == 0:
        return True  # Nothing to validate

    schema_names = table.schema.names

    # Find the input column
    input_col_name = None
    for col_name in input_columns:
        if col_name in schema_names:
            input_col_name = col_name
            break

    if input_col_name is None:
        logger.warning(
            f"Data validation: No input column found in {file_name}. "
            f"Available columns: {schema_names}. "
            f"Expected one of: {input_columns}"
        )
        return False

    # Check column type for zero-copy compatibility
    col_type = table.schema.field(input_col_name).type

    # Zero-copy compatible: fixed-size integers or fixed-size lists
    # Non-zero-copy: variable-length lists (list<int64>), strings
    is_variable_length = (
        pa.types.is_large_list(col_type) or
        pa.types.is_list(col_type)
    )

    if is_variable_length:
        inner_type = col_type.value_type if hasattr(col_type, 'value_type') else None
        if inner_type and (pa.types.is_integer(inner_type) or pa.types.is_floating(inner_type)):
            # Variable-length list of integers - supported but not optimal
            logger.debug(
                f"Data format: {file_name} uses variable-length lists. "
                f"This is supported but fixed-length arrays are 10-20% faster."
            )
            return True
        else:
            # Variable-length list of complex types - warn about slowdown
            logger.warning(
                f"Data format warning: {file_name} uses variable-length data "
                f"(type: {col_type}). This can cause 100-1000x slowdown. "
                f"Consider converting to fixed-size int64 arrays for optimal performance."
            )
            return False
    else:
        logger.debug(f"Data format validation passed: {file_name} uses efficient fixed-width arrays")
        return True


def validate_data_file(
    file_path: Union[str, Path],
    load_table_fn,
    input_columns: Optional[List[str]] = None,
) -> bool:
    """
    Validate a data file's format.

    Convenience wrapper that loads a table and validates it.

    Args:
        file_path: Path to the Arrow/Parquet file
        load_table_fn: Function to load the table (e.g., table_cache.get)
        input_columns: List of possible input column names

    Returns:
        True if validation passed, False if issues were found
    """
    try:
        file_path = Path(file_path)
        table = load_table_fn(file_path)
        return validate_arrow_data_format(
            table=table,
            file_name=file_path.name,
            input_columns=input_columns,
        )
    except Exception as e:
        # Don't fail on validation errors - just log and continue
        logger.debug(f"Data format validation skipped for {file_path}: {e}")
        return True


__all__ = [
    'validate_arrow_data_format',
    'validate_data_file',
]
