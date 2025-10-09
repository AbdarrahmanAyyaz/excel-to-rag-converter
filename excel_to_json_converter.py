#!/usr/bin/env python3
"""
Robust Excel to JSON Converter
Handles large Excel files (4000+ rows) with memory-efficient chunked processing.
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExcelToJSONConverter:
    """
    Convert Excel files to JSON with robust error handling and optimization.
    """

    def __init__(self, input_file: str):
        """
        Initialize converter with input file.

        Args:
            input_file: Path to Excel file (.xlsx or .xls)
        """
        self.input_file = Path(input_file)
        self._validate_input_file()

    def _validate_input_file(self) -> None:
        """Validate input file exists and has correct extension."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        if self.input_file.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError(f"Unsupported file format: {self.input_file.suffix}. Use .xlsx or .xls")

    def _clean_value(self, value: Any) -> Any:
        """
        Clean and normalize a single value.

        Args:
            value: Raw value from Excel

        Returns:
            Cleaned value
        """
        # Handle NaN, None, and pd.NA
        if pd.isna(value):
            return None

        # Handle strings - strip whitespace
        if isinstance(value, str):
            return value.strip()

        # Handle datetime objects - convert to ISO format
        if isinstance(value, (datetime, date, pd.Timestamp)):
            if isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            return value.isoformat()

        # Handle numpy integers and floats
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)

        if isinstance(value, (np.floating, np.float64, np.float32)):
            # Check if it's NaN (shouldn't happen due to pd.isna check, but be safe)
            if np.isnan(value):
                return None
            return float(value)

        # Handle booleans
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        # Return as-is for other types
        return value

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean entire dataframe.

        Args:
            df: Input dataframe

        Returns:
            Cleaned dataframe
        """
        # Create a copy to avoid modifying original
        df = df.copy()

        # Clean column names - strip whitespace
        df.columns = [str(col).strip() for col in df.columns]

        # Handle merged cells (they appear as NaN after first cell)
        # Forward fill merged cells
        for col in df.columns:
            # Only forward fill if there are consecutive NaN values (likely merged cells)
            if df[col].isna().any():
                # Check if column has mixed types indicating potential merged cells
                try:
                    df[col] = df[col].ffill()  # Use ffill() instead of deprecated fillna(method='ffill')
                except Exception as e:
                    logger.warning(f"Could not forward fill column '{col}': {e}")

        return df

    def _process_chunk(self, chunk: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a chunk of data with cleaning and validation.

        Args:
            chunk: DataFrame chunk

        Returns:
            List of records (dictionaries)
        """
        # Clean the chunk
        chunk = self._clean_dataframe(chunk)

        # Convert to dictionary format
        records = []
        for _, row in chunk.iterrows():
            record = {}
            for col in chunk.columns:
                try:
                    record[col] = self._clean_value(row[col])
                except Exception as e:
                    logger.warning(f"Error processing column '{col}': {e}. Setting to None.")
                    record[col] = None
            records.append(record)

        return records

    def _get_sheet_names(self, sheets: Optional[List[str]] = None) -> List[str]:
        """
        Get list of sheet names to process.

        Args:
            sheets: Optional list of specific sheets to process

        Returns:
            List of sheet names
        """
        try:
            # Get all sheet names
            excel_file = pd.ExcelFile(self.input_file, engine='openpyxl')
            all_sheets = excel_file.sheet_names

            if sheets is None:
                return all_sheets

            # Validate requested sheets exist
            invalid_sheets = set(sheets) - set(all_sheets)
            if invalid_sheets:
                logger.warning(f"Sheets not found: {invalid_sheets}. Available: {all_sheets}")

            valid_sheets = [s for s in sheets if s in all_sheets]
            if not valid_sheets:
                raise ValueError(f"None of the requested sheets exist in {self.input_file}")

            return valid_sheets

        except Exception as e:
            raise Exception(f"Error reading Excel file: {e}")

    def _process_sheet(
        self,
        sheet_name: str,
        chunk_size: int = 500
    ) -> Dict[str, Any]:
        """
        Process a single sheet with chunked reading.

        Args:
            sheet_name: Name of the sheet
            chunk_size: Number of rows per chunk

        Returns:
            Dictionary with sheet data and metadata
        """
        logger.info(f"Processing sheet: {sheet_name}")

        try:
            # First, get the total number of rows for progress bar
            df_sample = pd.read_excel(
                self.input_file,
                sheet_name=sheet_name,
                engine='openpyxl',
                nrows=0
            )

            # Get actual row count
            full_df = pd.read_excel(
                self.input_file,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            total_rows = len(full_df)
            columns = list(full_df.columns)

            # Process in chunks
            all_records = []
            num_chunks = (total_rows + chunk_size - 1) // chunk_size

            with tqdm(total=total_rows, desc=f"Processing {sheet_name}", unit="rows") as pbar:
                for chunk_num in range(num_chunks):
                    start_row = chunk_num * chunk_size

                    # Read chunk
                    chunk_df = pd.read_excel(
                        self.input_file,
                        sheet_name=sheet_name,
                        engine='openpyxl',
                        skiprows=range(1, start_row + 1) if start_row > 0 else None,
                        nrows=chunk_size
                    )

                    if chunk_df.empty:
                        break

                    # Process chunk
                    records = self._process_chunk(chunk_df)
                    all_records.extend(records)

                    # Update progress
                    pbar.update(len(chunk_df))

            return {
                'data': all_records,
                'row_count': len(all_records),
                'columns': columns
            }

        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {e}")
            raise

    def _generate_metadata(
        self,
        sheets_data: Dict[str, Dict[str, Any]],
        conversion_start: datetime
    ) -> Dict[str, Any]:
        """
        Generate metadata about the conversion.

        Args:
            sheets_data: Processed sheets data
            conversion_start: When conversion started

        Returns:
            Metadata dictionary
        """
        total_rows = sum(sheet['row_count'] for sheet in sheets_data.values())

        metadata = {
            'source_file': str(self.input_file.name),
            'source_path': str(self.input_file.absolute()),
            'conversion_date': datetime.now().isoformat(),
            'conversion_duration_seconds': (datetime.now() - conversion_start).total_seconds(),
            'total_rows': total_rows,
            'sheets': {
                name: {
                    'row_count': data['row_count'],
                    'columns': data['columns']
                }
                for name, data in sheets_data.items()
            }
        }

        return metadata

    def _validate_json(self, data: Dict[str, Any]) -> bool:
        """
        Validate that data can be serialized to JSON.

        Args:
            data: Data to validate

        Returns:
            True if valid, raises exception otherwise
        """
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError) as e:
            raise ValueError(f"Data cannot be serialized to JSON: {e}")

    def convert(
        self,
        output_file: Optional[str] = None,
        chunk_size: int = 500,
        sheets: Optional[List[str]] = None,
        minified: bool = False,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert Excel to JSON with proper error handling and optimization.

        Args:
            output_file: Path to output JSON file (optional, returns dict if None)
            chunk_size: Number of rows to process at once (default: 500)
            sheets: List of specific sheets to convert (None = all sheets)
            minified: If True, create minified JSON without indentation
            include_metadata: If True, include conversion metadata

        Returns:
            Dictionary containing converted data and metadata
        """
        conversion_start = datetime.now()

        try:
            # Get sheets to process
            sheet_names = self._get_sheet_names(sheets)
            logger.info(f"Converting {len(sheet_names)} sheet(s): {sheet_names}")

            # Process each sheet
            sheets_data = {}
            for sheet_name in sheet_names:
                try:
                    sheet_result = self._process_sheet(sheet_name, chunk_size)
                    sheets_data[sheet_name] = sheet_result
                except Exception as e:
                    logger.error(f"Failed to process sheet '{sheet_name}': {e}")
                    # Continue with other sheets
                    continue

            if not sheets_data:
                raise Exception("No sheets were successfully processed")

            # Build output structure
            if len(sheets_data) == 1:
                # Single sheet - use simpler structure
                sheet_name = list(sheets_data.keys())[0]
                output_data = {
                    'data': sheets_data[sheet_name]['data']
                }
            else:
                # Multiple sheets - nested structure
                output_data = {
                    'sheets': {
                        name: data['data']
                        for name, data in sheets_data.items()
                    }
                }

            # Add metadata if requested
            if include_metadata:
                metadata = self._generate_metadata(sheets_data, conversion_start)
                output_data['metadata'] = metadata

            # Validate JSON
            self._validate_json(output_data)
            logger.info("JSON validation successful")

            # Write to file if output path provided
            if output_file:
                output_path = Path(output_file)

                # Create parent directory if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)

                indent = None if minified else 2
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=indent, ensure_ascii=False)

                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"JSON written to {output_path} ({file_size_mb:.2f} MB)")

            return output_data

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise


def excel_to_json(
    input_file: str,
    output_file: Optional[str] = None,
    chunk_size: int = 500,
    sheets: Optional[List[str]] = None,
    minified: bool = False,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Convert Excel to JSON with proper error handling and optimization.

    Args:
        input_file: Path to input Excel file (.xlsx or .xls)
        output_file: Path to output JSON file (optional, returns dict if None)
        chunk_size: Number of rows to process at once (default: 500)
        sheets: List of specific sheets to convert (None = all sheets)
        minified: If True, create minified JSON without indentation
        include_metadata: If True, include conversion metadata

    Returns:
        Dictionary containing converted data and metadata

    Example:
        >>> # Convert entire Excel file
        >>> result = excel_to_json('data.xlsx', 'output.json')

        >>> # Convert specific sheets only
        >>> result = excel_to_json('data.xlsx', 'output.json', sheets=['Sheet1', 'Sheet2'])

        >>> # Create minified JSON without metadata
        >>> result = excel_to_json('data.xlsx', 'output.json', minified=True, include_metadata=False)

        >>> # Just convert to dict without saving
        >>> data = excel_to_json('data.xlsx')
    """
    converter = ExcelToJSONConverter(input_file)
    return converter.convert(
        output_file=output_file,
        chunk_size=chunk_size,
        sheets=sheets,
        minified=minified,
        include_metadata=include_metadata
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Excel files to JSON with robust error handling'
    )
    parser.add_argument('input_file', help='Input Excel file path')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-c', '--chunk-size', type=int, default=500,
                        help='Rows per chunk (default: 500)')
    parser.add_argument('-s', '--sheets', nargs='+',
                        help='Specific sheets to convert (default: all)')
    parser.add_argument('-m', '--minified', action='store_true',
                        help='Create minified JSON')
    parser.add_argument('--no-metadata', action='store_true',
                        help='Exclude metadata from output')

    args = parser.parse_args()

    # Generate default output filename if not provided
    output_file = args.output
    if not output_file:
        input_path = Path(args.input_file)
        output_file = input_path.with_suffix('.json')

    try:
        result = excel_to_json(
            input_file=args.input_file,
            output_file=output_file,
            chunk_size=args.chunk_size,
            sheets=args.sheets,
            minified=args.minified,
            include_metadata=not args.no_metadata
        )
        print(f"\nConversion successful!")
        if 'metadata' in result:
            print(f"Total rows: {result['metadata']['total_rows']}")
            print(f"Duration: {result['metadata']['conversion_duration_seconds']:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
