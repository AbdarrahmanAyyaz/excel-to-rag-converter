"""Data exporters for JSON, JSONL, Markdown, and ZIP packaging."""

import json
import zipfile
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional
import pandas as pd
import orjson

from .types import ConversionManifest, FileResult, SheetResult, ExportSettings
from .utils import safe_filename
from .converters import get_sheet_dataframe


def export_to_zip(file_results: List[FileResult], manifest: ConversionManifest,
                  file_data: Dict[str, bytes], settings: ExportSettings) -> bytes:
    """Create a ZIP file containing all exported data."""
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create folders for different output types
        all_fact_sentences = []

        for file_result in file_results:
            file_name_base = safe_filename(file_result.file_name.rsplit('.', 1)[0])

            for sheet_result in file_result.sheet_results:
                if not sheet_result.has_data:
                    continue

                sheet_name_safe = safe_filename(sheet_result.sheet_name)
                base_filename = f"{file_name_base}__{sheet_name_safe}"

                # Get the DataFrame for this sheet
                df = get_sheet_dataframe(
                    file_data[file_result.file_name],
                    file_result.file_name,
                    sheet_result.sheet_name
                )

                if df is None or df.empty:
                    continue

                # Export JSON
                if settings.export_json:
                    json_data = _create_json_export(df, sheet_result)
                    json_content = orjson.dumps(json_data, option=orjson.OPT_INDENT_2).decode()
                    zip_file.writestr(f"json/{base_filename}.json", json_content)

                # Export JSONL
                if settings.export_jsonl:
                    jsonl_content = _create_jsonl_export(df, sheet_result)
                    zip_file.writestr(f"jsonl/{base_filename}.jsonl", jsonl_content)

                # Export Markdown
                if settings.export_markdown:
                    markdown_content = _create_markdown_export(df, sheet_result)
                    zip_file.writestr(f"markdown/{base_filename}.md", markdown_content)

                # Collect fact sentences
                if sheet_result.fact_sentences:
                    for sentence in sheet_result.fact_sentences:
                        all_fact_sentences.append({
                            "file": file_result.file_name,
                            "sheet": sheet_result.sheet_name,
                            "fact": sentence
                        })

        # Export fact sentences if any exist
        if all_fact_sentences:
            fact_sentences_jsonl = "\n".join(
                orjson.dumps(fact).decode() for fact in all_fact_sentences
            )
            zip_file.writestr("fact_sentences.jsonl", fact_sentences_jsonl)

        # Export manifest
        manifest_content = orjson.dumps(manifest.to_dict(), option=orjson.OPT_INDENT_2).decode()
        zip_file.writestr("manifest.json", manifest_content)

        # Export settings
        settings_content = orjson.dumps({
            "export_json": settings.export_json,
            "export_jsonl": settings.export_jsonl,
            "export_markdown": settings.export_markdown,
            "use_gemini": settings.use_gemini,
            "gemini_headers": settings.gemini_headers,
            "gemini_summary": settings.gemini_summary,
            "gemini_facts": settings.gemini_facts,
        }, option=orjson.OPT_INDENT_2).decode()
        zip_file.writestr("settings.json", settings_content)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def _create_json_export(df: pd.DataFrame, sheet_result: SheetResult) -> Dict[str, Any]:
    """Create JSON export structure preserving original data."""
    # Convert DataFrame to records (list of dictionaries)
    records = df.to_dict('records')

    # Build the export structure
    export_data = {
        "metadata": {
            "file_name": sheet_result.file_name,
            "sheet_name": sheet_result.sheet_name,
            "row_count": sheet_result.row_count,
            "original_headers": sheet_result.original_headers,
            "processing_time_seconds": sheet_result.processing_time_seconds
        },
        "data": records
    }

    # Add Gemini enhancements if available
    if sheet_result.suggested_headers:
        export_data["metadata"]["suggested_headers"] = sheet_result.suggested_headers

    if sheet_result.summary:
        export_data["metadata"]["summary"] = sheet_result.summary

    # Add header mapping if suggested headers exist
    if sheet_result.suggested_headers and len(sheet_result.suggested_headers) == len(sheet_result.original_headers):
        header_mapping = {
            orig: suggested for orig, suggested in
            zip(sheet_result.original_headers, sheet_result.suggested_headers)
        }
        export_data["metadata"]["header_mapping"] = header_mapping

    return export_data


def _create_jsonl_export(df: pd.DataFrame, sheet_result: SheetResult) -> str:
    """Create JSONL export (one JSON object per line)."""
    lines = []

    # Add metadata as first line
    metadata = {
        "_type": "metadata",
        "file_name": sheet_result.file_name,
        "sheet_name": sheet_result.sheet_name,
        "row_count": sheet_result.row_count,
        "original_headers": sheet_result.original_headers,
        "processing_time_seconds": sheet_result.processing_time_seconds
    }

    if sheet_result.suggested_headers:
        metadata["suggested_headers"] = sheet_result.suggested_headers

    if sheet_result.summary:
        metadata["summary"] = sheet_result.summary

    lines.append(orjson.dumps(metadata).decode())

    # Add each data row
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict["_type"] = "data"
        lines.append(orjson.dumps(row_dict).decode())

    return "\n".join(lines)


def _create_markdown_export(df: pd.DataFrame, sheet_result: SheetResult) -> str:
    """Create Markdown export with table and metadata."""
    lines = []

    # Title and metadata
    lines.append(f"# {sheet_result.sheet_name}")
    lines.append("")
    lines.append(f"**File:** {sheet_result.file_name}")
    lines.append(f"**Sheet:** {sheet_result.sheet_name}")
    lines.append(f"**Rows:** {sheet_result.row_count}")
    lines.append(f"**Processing Time:** {sheet_result.processing_time_seconds:.2f}s")
    lines.append("")

    # Add Gemini summary if available
    if sheet_result.summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(sheet_result.summary)
        lines.append("")

    # Add header mapping if available
    if (sheet_result.suggested_headers and
        len(sheet_result.suggested_headers) == len(sheet_result.original_headers)):
        lines.append("## Column Mapping")
        lines.append("")
        lines.append("| Original Header | Suggested Header |")
        lines.append("|---|---|")
        for orig, suggested in zip(sheet_result.original_headers, sheet_result.suggested_headers):
            lines.append(f"| {orig} | {suggested} |")
        lines.append("")

    # Add data table
    lines.append("## Data")
    lines.append("")

    if not df.empty:
        # Convert DataFrame to markdown table
        try:
            # Use pandas to_markdown if available (pandas >= 1.0.0)
            if hasattr(df, 'to_markdown'):
                table_md = df.to_markdown(index=False, tablefmt='github')
                lines.append(table_md)
            else:
                # Fallback: create table manually
                lines.append(_dataframe_to_markdown_table(df))
        except Exception:
            # Final fallback: simple format
            lines.append(_dataframe_to_markdown_table(df))
    else:
        lines.append("*No data available*")

    lines.append("")

    # Add processing notes
    lines.append("---")
    lines.append("")
    lines.append("*Generated by Excel-to-Everything Converter*")

    if sheet_result.errors:
        lines.append("")
        lines.append("### Processing Warnings")
        for error in sheet_result.errors:
            lines.append(f"- {error.error_type}: {error.error_message}")

    return "\n".join(lines)


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to Markdown table format."""
    if df.empty:
        return "*No data available*"

    lines = []

    # Header row
    headers = [str(col) for col in df.columns]
    lines.append("| " + " | ".join(headers) + " |")

    # Separator row
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows (limit to reasonable number for markdown)
    max_rows = min(1000, len(df))  # Limit to 1000 rows for readability
    for i in range(max_rows):
        row = df.iloc[i]
        row_values = [str(val)[:100] + "..." if len(str(val)) > 100 else str(val) for val in row]
        # Escape pipe characters in cell content
        row_values = [val.replace("|", "\\|") for val in row_values]
        lines.append("| " + " | ".join(row_values) + " |")

    if len(df) > max_rows:
        lines.append("")
        lines.append(f"*... and {len(df) - max_rows} more rows*")

    return "\n".join(lines)


def create_settings_json(settings: ExportSettings) -> str:
    """Create a JSON string of export settings for download."""
    settings_dict = {
        "export_json": settings.export_json,
        "export_jsonl": settings.export_jsonl,
        "export_markdown": settings.export_markdown,
        "use_gemini": settings.use_gemini,
        "gemini_headers": settings.gemini_headers,
        "gemini_summary": settings.gemini_summary,
        "gemini_facts": settings.gemini_facts,
    }
    return orjson.dumps(settings_dict, option=orjson.OPT_INDENT_2).decode()


def load_settings_from_json(json_str: str) -> Optional[ExportSettings]:
    """Load export settings from JSON string."""
    try:
        data = json.loads(json_str)
        return ExportSettings(
            export_json=data.get("export_json", True),
            export_jsonl=data.get("export_jsonl", True),
            export_markdown=data.get("export_markdown", True),
            use_gemini=data.get("use_gemini", False),
            gemini_headers=data.get("gemini_headers", False),
            gemini_summary=data.get("gemini_summary", False),
            gemini_facts=data.get("gemini_facts", False),
        )
    except (json.JSONDecodeError, KeyError):
        return None