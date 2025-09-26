"""File converters for Excel, PDF, CSV, JSON, and ODS formats."""

import time
import json
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pdfplumber
try:
    from odf.opendocument import load
    from odf.table import Table, TableRow, TableCell
    from odf.text import P
    ODF_AVAILABLE = True
except ImportError:
    ODF_AVAILABLE = False

from .types import ProcessingError, SheetResult, FileResult, ExportSettings
from .utils import (
    safe_filename, forward_fill_dataframe, detect_header_row,
    validate_dataframe, extract_sheet_preview
)
from .gemini_client import GeminiClient


def convert_excel_file(file_bytes: bytes, file_name: str, settings: ExportSettings,
                      gemini_client: Optional[GeminiClient] = None) -> FileResult:
    """Convert an Excel file to structured data with optional Gemini enhancements."""
    start_time = time.time()
    file_type = "Excel"
    sheet_results = []
    errors = []

    try:
        # Read the Excel file
        excel_data = pd.ExcelFile(BytesIO(file_bytes))
        sheet_names = excel_data.sheet_names

        for sheet_name in sheet_names:
            sheet_result = _process_excel_sheet(
                excel_data, sheet_name, file_name, settings, gemini_client
            )
            sheet_results.append(sheet_result)

            # Collect any sheet-level errors
            errors.extend(sheet_result.errors)

    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message=str(e),
            error_type="FileReadError"
        )
        errors.append(error)

    processing_time = time.time() - start_time
    successful_sheets = sum(1 for result in sheet_results if result.has_data)

    return FileResult(
        file_name=file_name,
        file_type=file_type,
        total_sheets=len(sheet_results),
        successful_sheets=successful_sheets,
        sheet_results=sheet_results,
        processing_time_seconds=processing_time,
        errors=errors
    )


def _process_excel_sheet(excel_data: pd.ExcelFile, sheet_name: str, file_name: str,
                        settings: ExportSettings, gemini_client: Optional[GeminiClient]) -> SheetResult:
    """Process a single Excel sheet."""
    start_time = time.time()
    errors = []
    original_headers = []
    suggested_headers = None
    summary = None
    fact_sentences = None
    row_count = 0
    has_data = False

    try:
        # Read the sheet
        df = pd.read_excel(excel_data, sheet_name=sheet_name, header=None)

        if df.empty:
            return SheetResult(
                file_name=file_name,
                sheet_name=sheet_name,
                original_headers=[],
                suggested_headers=None,
                row_count=0,
                summary=None,
                fact_sentences=None,
                has_data=False,
                processing_time_seconds=time.time() - start_time,
                errors=errors
            )

        # Detect header row
        header_row_idx = detect_header_row(df)

        # Set the headers and data
        if header_row_idx > 0:
            # Use detected header row
            original_headers = df.iloc[header_row_idx].astype(str).tolist()
            df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
            df.columns = original_headers
        else:
            # Use first row as headers
            original_headers = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = original_headers

        # Clean up the dataframe
        df = forward_fill_dataframe(df)

        # Validate the data
        has_data = validate_dataframe(df)
        row_count = len(df) if has_data else 0

        if has_data and settings.use_gemini and gemini_client:
            try:
                # Get sample rows for Gemini processing
                sample_rows = df.head(5).values.tolist()

                # Generate header suggestions
                if settings.gemini_headers:
                    suggested_headers = gemini_client.suggest_headers(original_headers, sample_rows)

                # Generate sheet summary
                if settings.gemini_summary:
                    summary = gemini_client.generate_sheet_summary(
                        sheet_name, original_headers, row_count, sample_rows
                    )

                # Generate fact sentences (limit to first 100 rows for performance)
                if settings.gemini_facts:
                    max_fact_rows = min(100, row_count)
                    fact_sentences = gemini_client.generate_fact_sentences(
                        original_headers, df.head(max_fact_rows).values.tolist()
                    )

            except Exception as e:
                error = ProcessingError(
                    file_name=file_name,
                    sheet_name=sheet_name,
                    error_message=f"Gemini processing error: {str(e)}",
                    error_type="GeminiError"
                )
                errors.append(error)

    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=sheet_name,
            error_message=str(e),
            error_type="SheetProcessingError"
        )
        errors.append(error)

    processing_time = time.time() - start_time

    return SheetResult(
        file_name=file_name,
        sheet_name=sheet_name,
        original_headers=original_headers,
        suggested_headers=suggested_headers,
        row_count=row_count,
        summary=summary,
        fact_sentences=fact_sentences,
        has_data=has_data,
        processing_time_seconds=processing_time,
        errors=errors
    )


def convert_pdf_file(file_bytes: bytes, file_name: str, settings: ExportSettings,
                    gemini_client: Optional[GeminiClient] = None) -> FileResult:
    """Convert a PDF file to structured data (best effort table extraction)."""
    start_time = time.time()
    file_type = "PDF"
    sheet_results = []
    errors = []

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            # Try to extract tables from all pages
            all_tables = []
            page_texts = []

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Try table extraction first
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables):
                            if table and len(table) > 1:  # Must have headers + data
                                sheet_name = f"Page_{page_num}_Table_{table_num + 1}"
                                sheet_result = _process_pdf_table(
                                    table, sheet_name, file_name, settings, gemini_client
                                )
                                sheet_results.append(sheet_result)

                    # If no tables found, extract text
                    if not tables:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            page_texts.append((page_num, page_text))

                except Exception as e:
                    error = ProcessingError(
                        file_name=file_name,
                        sheet_name=f"Page_{page_num}",
                        error_message=f"Page processing error: {str(e)}",
                        error_type="PDFPageError"
                    )
                    errors.append(error)

            # If we got text but no tables, create a text sheet
            if page_texts and not sheet_results:
                combined_text = "\n\n".join([f"Page {num}:\n{text}" for num, text in page_texts])
                sheet_result = _create_text_sheet(combined_text, "PDF_Text", file_name)
                sheet_results.append(sheet_result)

    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message=f"PDF processing error: {str(e)}",
            error_type="PDFFileError"
        )
        errors.append(error)

    processing_time = time.time() - start_time
    successful_sheets = sum(1 for result in sheet_results if result.has_data)

    return FileResult(
        file_name=file_name,
        file_type=file_type,
        total_sheets=len(sheet_results),
        successful_sheets=successful_sheets,
        sheet_results=sheet_results,
        processing_time_seconds=processing_time,
        errors=errors
    )


def _process_pdf_table(table: List[List[str]], sheet_name: str, file_name: str,
                      settings: ExportSettings, gemini_client: Optional[GeminiClient]) -> SheetResult:
    """Process a table extracted from PDF."""
    start_time = time.time()
    errors = []

    try:
        # Convert table to DataFrame
        headers = [str(cell) if cell else f"Column_{i+1}" for i, cell in enumerate(table[0])]
        data_rows = table[1:]

        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        df = forward_fill_dataframe(df)

        original_headers = headers
        suggested_headers = None
        summary = None
        fact_sentences = None
        row_count = len(df)
        has_data = validate_dataframe(df)

        if has_data and settings.use_gemini and gemini_client:
            try:
                sample_rows = df.head(5).values.tolist()

                if settings.gemini_headers:
                    suggested_headers = gemini_client.suggest_headers(original_headers, sample_rows)

                if settings.gemini_summary:
                    summary = gemini_client.generate_sheet_summary(
                        sheet_name, original_headers, row_count, sample_rows
                    )

                if settings.gemini_facts:
                    max_fact_rows = min(50, row_count)  # Fewer rows for PDF
                    fact_sentences = gemini_client.generate_fact_sentences(
                        original_headers, df.head(max_fact_rows).values.tolist()
                    )

            except Exception as e:
                error = ProcessingError(
                    file_name=file_name,
                    sheet_name=sheet_name,
                    error_message=f"Gemini processing error: {str(e)}",
                    error_type="GeminiError"
                )
                errors.append(error)

    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=sheet_name,
            error_message=str(e),
            error_type="PDFTableError"
        )
        errors.append(error)
        return SheetResult(
            file_name=file_name,
            sheet_name=sheet_name,
            original_headers=[],
            suggested_headers=None,
            row_count=0,
            summary=None,
            fact_sentences=None,
            has_data=False,
            processing_time_seconds=time.time() - start_time,
            errors=errors
        )

    processing_time = time.time() - start_time

    return SheetResult(
        file_name=file_name,
        sheet_name=sheet_name,
        original_headers=original_headers,
        suggested_headers=suggested_headers,
        row_count=row_count,
        summary=summary,
        fact_sentences=fact_sentences,
        has_data=has_data,
        processing_time_seconds=processing_time,
        errors=errors
    )


def _create_text_sheet(text: str, sheet_name: str, file_name: str) -> SheetResult:
    """Create a sheet result from extracted text."""
    start_time = time.time()

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    return SheetResult(
        file_name=file_name,
        sheet_name=sheet_name,
        original_headers=["content"],
        suggested_headers=None,
        row_count=len(paragraphs),
        summary=f"Text content extracted from PDF with {len(paragraphs)} paragraphs",
        fact_sentences=None,
        has_data=len(paragraphs) > 0,
        processing_time_seconds=time.time() - start_time,
        errors=[]
    )


def get_sheet_dataframe(file_bytes: bytes, file_name: str, sheet_name: str) -> Optional[pd.DataFrame]:
    """Get a DataFrame for a specific sheet (used for preview and export)."""
    try:
        file_extension = file_name.lower().split('.')[-1]

        if file_extension in ['xls', 'xlsx']:
            excel_data = pd.ExcelFile(BytesIO(file_bytes))
            if sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_data, sheet_name=sheet_name, header=None)

                if not df.empty:
                    # Detect and set headers
                    header_row_idx = detect_header_row(df)
                    if header_row_idx > 0:
                        headers = df.iloc[header_row_idx].astype(str).tolist()
                        df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
                        df.columns = headers
                    else:
                        headers = df.iloc[0].astype(str).tolist()
                        df = df.iloc[1:].reset_index(drop=True)
                        df.columns = headers

                    return forward_fill_dataframe(df)

        elif file_extension == 'pdf':
            # For PDF, we'd need to re-extract the specific table
            # This is a simplified version
            return None

        elif file_extension in ['csv', 'tsv']:
            # For CSV/TSV, return the main sheet
            if sheet_name == "Data":
                return _get_csv_dataframe(file_bytes, file_extension)

        elif file_extension == 'json':
            # For JSON, return the main sheet
            if sheet_name == "Data":
                return _get_json_dataframe(file_bytes)

        elif file_extension == 'ods':
            # For ODS, handle like Excel
            return _get_ods_sheet_dataframe(file_bytes, sheet_name)

    except Exception:
        return None

    return None


def convert_csv_file(file_bytes: bytes, file_name: str, settings: ExportSettings,
                    gemini_client: Optional[GeminiClient] = None) -> FileResult:
    """Convert a CSV or TSV file to structured data with optional Gemini enhancements."""
    start_time = time.time()
    file_extension = file_name.lower().split('.')[-1]
    file_type = "CSV" if file_extension == 'csv' else "TSV"
    sheet_results = []
    errors = []

    try:
        # Determine separator
        separator = ',' if file_extension == 'csv' else '\t'

        # Read the file
        text_content = file_bytes.decode('utf-8')
        df = pd.read_csv(StringIO(text_content), sep=separator)

        # Process as a single sheet
        sheet_result = _process_dataframe_sheet(
            df, "Data", file_name, settings, gemini_client
        )
        sheet_results.append(sheet_result)

    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
            try:
                text_content = file_bytes.decode(encoding)
                df = pd.read_csv(StringIO(text_content), sep=separator)
                sheet_result = _process_dataframe_sheet(
                    df, "Data", file_name, settings, gemini_client
                )
                sheet_results.append(sheet_result)
                break
            except Exception:
                continue
        else:
            error = ProcessingError(
                file_name=file_name,
                sheet_name=None,
                error_message="Could not decode file with any common encoding",
                error_type="EncodingError"
            )
            errors.append(error)

    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message=str(e),
            error_type="FileReadError"
        )
        errors.append(error)

    processing_time = time.time() - start_time
    successful_sheets = sum(1 for result in sheet_results if result.has_data)

    return FileResult(
        file_name=file_name,
        file_type=file_type,
        total_sheets=len(sheet_results),
        successful_sheets=successful_sheets,
        sheet_results=sheet_results,
        processing_time_seconds=processing_time,
        errors=errors
    )


def convert_json_file(file_bytes: bytes, file_name: str, settings: ExportSettings,
                     gemini_client: Optional[GeminiClient] = None) -> FileResult:
    """Convert a JSON file to structured data with optional Gemini enhancements."""
    start_time = time.time()
    file_type = "JSON"
    sheet_results = []
    errors = []

    try:
        # Parse JSON
        json_content = json.loads(file_bytes.decode('utf-8'))

        # Convert to DataFrame
        if isinstance(json_content, list):
            # Array of objects
            df = pd.DataFrame(json_content)
        elif isinstance(json_content, dict):
            # Single object or nested structure
            if all(isinstance(v, (list, dict)) for v in json_content.values()):
                # Multiple sheets/tables
                for key, value in json_content.items():
                    if isinstance(value, list) and value:
                        try:
                            sub_df = pd.DataFrame(value)
                            sheet_result = _process_dataframe_sheet(
                                sub_df, key, file_name, settings, gemini_client
                            )
                            sheet_results.append(sheet_result)
                        except Exception as e:
                            error = ProcessingError(
                                file_name=file_name,
                                sheet_name=key,
                                error_message=str(e),
                                error_type="JSONProcessingError"
                            )
                            errors.append(error)

                if not sheet_results:
                    # Fallback: treat as single object
                    df = pd.DataFrame([json_content])
                    sheet_result = _process_dataframe_sheet(
                        df, "Data", file_name, settings, gemini_client
                    )
                    sheet_results.append(sheet_result)
            else:
                # Single flat object
                df = pd.DataFrame([json_content])
                sheet_result = _process_dataframe_sheet(
                    df, "Data", file_name, settings, gemini_client
                )
                sheet_results.append(sheet_result)
        else:
            # Unsupported JSON structure
            error = ProcessingError(
                file_name=file_name,
                sheet_name=None,
                error_message="JSON structure not supported (must be object or array)",
                error_type="JSONStructureError"
            )
            errors.append(error)

    except json.JSONDecodeError as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message=f"Invalid JSON: {str(e)}",
            error_type="JSONDecodeError"
        )
        errors.append(error)
    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message=str(e),
            error_type="FileReadError"
        )
        errors.append(error)

    processing_time = time.time() - start_time
    successful_sheets = sum(1 for result in sheet_results if result.has_data)

    return FileResult(
        file_name=file_name,
        file_type=file_type,
        total_sheets=len(sheet_results),
        successful_sheets=successful_sheets,
        sheet_results=sheet_results,
        processing_time_seconds=processing_time,
        errors=errors
    )


def convert_ods_file(file_bytes: bytes, file_name: str, settings: ExportSettings,
                    gemini_client: Optional[GeminiClient] = None) -> FileResult:
    """Convert an ODS file to structured data with optional Gemini enhancements."""
    start_time = time.time()
    file_type = "ODS"
    sheet_results = []
    errors = []

    if not ODF_AVAILABLE:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message="ODS support requires 'odfpy' package. Please install it.",
            error_type="DependencyError"
        )
        errors.append(error)
        return FileResult(
            file_name=file_name,
            file_type=file_type,
            total_sheets=0,
            successful_sheets=0,
            sheet_results=[],
            processing_time_seconds=time.time() - start_time,
            errors=errors
        )

    try:
        # Load ODS document
        doc = load(BytesIO(file_bytes))
        tables = doc.getElementsByType(Table)

        for table in tables:
            table_name = table.getAttribute('name') or f"Sheet_{len(sheet_results) + 1}"

            try:
                # Extract table data
                rows = []
                for row in table.getElementsByType(TableRow):
                    row_data = []
                    for cell in row.getElementsByType(TableCell):
                        # Get cell text content
                        paragraphs = cell.getElementsByType(P)
                        cell_text = ' '.join(p.firstChild.data if p.firstChild else '' for p in paragraphs)
                        row_data.append(cell_text)
                    if row_data:  # Skip empty rows
                        rows.append(row_data)

                if rows:
                    # Convert to DataFrame
                    max_cols = max(len(row) for row in rows) if rows else 0
                    # Pad rows to same length
                    padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]

                    headers = padded_rows[0] if padded_rows else []
                    data_rows = padded_rows[1:] if len(padded_rows) > 1 else []

                    if headers and data_rows:
                        df = pd.DataFrame(data_rows, columns=headers)
                        df = forward_fill_dataframe(df)

                        sheet_result = _process_dataframe_sheet(
                            df, table_name, file_name, settings, gemini_client
                        )
                        sheet_results.append(sheet_result)

            except Exception as e:
                error = ProcessingError(
                    file_name=file_name,
                    sheet_name=table_name,
                    error_message=str(e),
                    error_type="ODSSheetError"
                )
                errors.append(error)

    except Exception as e:
        error = ProcessingError(
            file_name=file_name,
            sheet_name=None,
            error_message=str(e),
            error_type="ODSFileError"
        )
        errors.append(error)

    processing_time = time.time() - start_time
    successful_sheets = sum(1 for result in sheet_results if result.has_data)

    return FileResult(
        file_name=file_name,
        file_type=file_type,
        total_sheets=len(sheet_results),
        successful_sheets=successful_sheets,
        sheet_results=sheet_results,
        processing_time_seconds=processing_time,
        errors=errors
    )


def _process_dataframe_sheet(df: pd.DataFrame, sheet_name: str, file_name: str,
                           settings: ExportSettings, gemini_client: Optional[GeminiClient]) -> SheetResult:
    """Process a DataFrame as a sheet (common logic for CSV, JSON, etc.)."""
    start_time = time.time()
    errors = []

    if df.empty:
        return SheetResult(
            file_name=file_name,
            sheet_name=sheet_name,
            original_headers=[],
            suggested_headers=None,
            row_count=0,
            summary=None,
            fact_sentences=None,
            has_data=False,
            processing_time_seconds=time.time() - start_time,
            errors=errors
        )

    # Clean up the dataframe
    df = forward_fill_dataframe(df)

    original_headers = df.columns.tolist()
    suggested_headers = None
    summary = None
    fact_sentences = None
    row_count = len(df)
    has_data = validate_dataframe(df)

    if has_data and settings.use_gemini and gemini_client:
        try:
            sample_rows = df.head(5).values.tolist()

            # Generate header suggestions
            if settings.gemini_headers:
                suggested_headers = gemini_client.suggest_headers(original_headers, sample_rows)

            # Generate sheet summary
            if settings.gemini_summary:
                summary = gemini_client.generate_sheet_summary(
                    sheet_name, original_headers, row_count, sample_rows
                )

            # Generate fact sentences (limit to first 100 rows for performance)
            if settings.gemini_facts:
                max_fact_rows = min(100, row_count)
                fact_sentences = gemini_client.generate_fact_sentences(
                    original_headers, df.head(max_fact_rows).values.tolist()
                )

        except Exception as e:
            error = ProcessingError(
                file_name=file_name,
                sheet_name=sheet_name,
                error_message=f"Gemini processing error: {str(e)}",
                error_type="GeminiError"
            )
            errors.append(error)

    processing_time = time.time() - start_time

    return SheetResult(
        file_name=file_name,
        sheet_name=sheet_name,
        original_headers=original_headers,
        suggested_headers=suggested_headers,
        row_count=row_count,
        summary=summary,
        fact_sentences=fact_sentences,
        has_data=has_data,
        processing_time_seconds=processing_time,
        errors=errors
    )


def _get_csv_dataframe(file_bytes: bytes, file_extension: str) -> Optional[pd.DataFrame]:
    """Get DataFrame from CSV/TSV file."""
    try:
        separator = ',' if file_extension == 'csv' else '\t'
        text_content = file_bytes.decode('utf-8')
        df = pd.read_csv(StringIO(text_content), sep=separator)
        return forward_fill_dataframe(df)
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
            try:
                text_content = file_bytes.decode(encoding)
                df = pd.read_csv(StringIO(text_content), sep=separator)
                return forward_fill_dataframe(df)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _get_json_dataframe(file_bytes: bytes) -> Optional[pd.DataFrame]:
    """Get DataFrame from JSON file."""
    try:
        json_content = json.loads(file_bytes.decode('utf-8'))
        if isinstance(json_content, list):
            df = pd.DataFrame(json_content)
        elif isinstance(json_content, dict):
            df = pd.DataFrame([json_content])
        else:
            return None
        return forward_fill_dataframe(df)
    except Exception:
        return None


def _get_ods_sheet_dataframe(file_bytes: bytes, sheet_name: str) -> Optional[pd.DataFrame]:
    """Get DataFrame from specific ODS sheet."""
    if not ODF_AVAILABLE:
        return None

    try:
        doc = load(BytesIO(file_bytes))
        tables = doc.getElementsByType(Table)

        for table in tables:
            table_name = table.getAttribute('name') or f"Sheet_{len(tables)}"
            if table_name == sheet_name:
                # Extract table data
                rows = []
                for row in table.getElementsByType(TableRow):
                    row_data = []
                    for cell in row.getElementsByType(TableCell):
                        paragraphs = cell.getElementsByType(P)
                        cell_text = ' '.join(p.firstChild.data if p.firstChild else '' for p in paragraphs)
                        row_data.append(cell_text)
                    if row_data:
                        rows.append(row_data)

                if rows and len(rows) > 1:
                    headers = rows[0]
                    data_rows = rows[1:]
                    max_cols = len(headers)
                    # Pad rows to match header length
                    padded_rows = [row + [''] * (max_cols - len(row)) for row in data_rows]

                    df = pd.DataFrame(padded_rows, columns=headers)
                    return forward_fill_dataframe(df)

    except Exception:
        return None
    return None