"""Utility functions for file handling, data processing, and validation."""

import re
import hashlib
import time
from pathlib import Path
from typing import Any, List, Dict, Optional
import pandas as pd
from contextlib import contextmanager


def safe_filename(name: str, max_length: int = 200) -> str:
    """Convert a string to a safe filename by removing/replacing problematic characters."""
    # Replace problematic characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Replace multiple consecutive underscores with single underscore
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores and whitespace
    safe_name = safe_name.strip('_ ')
    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length].rstrip('_')
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"
    return safe_name


def forward_fill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill merged cells and replace NaN values with empty strings."""
    # Forward fill to handle merged cells
    df_filled = df.ffill()
    # Replace remaining NaN values with empty strings
    df_filled = df_filled.fillna("")
    return df_filled


def sanitize_headers(headers: List[str]) -> List[str]:
    """Clean up column headers by removing special characters and standardizing format."""
    sanitized = []
    for header in headers:
        # Convert to string and strip whitespace
        clean_header = str(header).strip()
        # Replace common problematic patterns
        clean_header = re.sub(r'[*#@!$%^&()+={}[\]|\\:";\'<>?,.`~]', '', clean_header)
        # Replace whitespace and dashes with underscores
        clean_header = re.sub(r'[\s\-]+', '_', clean_header)
        # Remove leading/trailing underscores
        clean_header = clean_header.strip('_')
        # Ensure it's not empty
        if not clean_header:
            clean_header = f"column_{len(sanitized) + 1}"
        sanitized.append(clean_header)
    return sanitized


def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file contents."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]  # First 16 chars for brevity


def detect_header_row(df: pd.DataFrame, max_rows_to_check: int = 10) -> int:
    """
    Attempt to detect which row contains the actual headers.
    Returns the 0-based index of the header row, or 0 if uncertain.
    """
    if df.empty:
        return 0

    rows_to_check = min(max_rows_to_check, len(df))
    best_row = 0
    best_score = 0

    for row_idx in range(rows_to_check):
        score = 0
        row_values = df.iloc[row_idx].astype(str).tolist()

        # Scoring criteria for detecting header rows:

        # 1. Non-empty values
        non_empty_count = sum(1 for val in row_values if val.strip() and val != 'nan')
        score += non_empty_count * 2

        # 2. Unique values (headers should be unique)
        unique_count = len(set(v.strip().lower() for v in row_values if v.strip() and v != 'nan'))
        score += unique_count

        # 3. Text vs numbers (headers are usually text)
        text_count = sum(1 for val in row_values if val.strip() and not _is_numeric(val))
        score += text_count

        # 4. Avoid rows that look like data
        if row_idx > 0:  # Don't penalize the first row
            # Check if this row looks like it contains mostly data values
            numeric_count = sum(1 for val in row_values if _is_numeric(val))
            if numeric_count > len(row_values) * 0.7:  # More than 70% numeric
                score -= 10

        if score > best_score:
            best_score = score
            best_row = row_idx

    return best_row


def _is_numeric(value: str) -> bool:
    """Check if a string represents a numeric value."""
    if not value or value.strip() == '':
        return False
    try:
        float(value.replace(',', ''))  # Handle comma-separated numbers
        return True
    except ValueError:
        return False


def extract_sheet_preview(df: pd.DataFrame, max_rows: int = 30) -> Dict[str, Any]:
    """Extract a preview of the sheet data for display purposes."""
    if df.empty:
        return {
            'headers': [],
            'rows': [],
            'total_rows': 0,
            'preview_rows': 0
        }

    headers = df.columns.tolist()
    preview_rows = min(max_rows, len(df))
    rows = df.head(preview_rows).values.tolist()

    return {
        'headers': headers,
        'rows': rows,
        'total_rows': len(df),
        'preview_rows': preview_rows
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


@contextmanager
def timer():
    """Context manager to measure execution time."""
    start_time = time.time()
    yield
    end_time = time.time()
    return end_time - start_time


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate that a dataframe has usable data."""
    if df is None or df.empty:
        return False

    # Check if all columns are unnamed or contain only NaN/empty values
    has_named_columns = any(
        not str(col).startswith('Unnamed') and str(col).strip()
        for col in df.columns
    )

    has_data = not df.dropna(how='all').empty

    return has_named_columns or has_data


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with optional suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix