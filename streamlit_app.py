"""
Excel-to-Everything Converter
A Streamlit app for converting Excel and PDF files to JSON, JSONL, and Markdown
with optional Gemini AI enhancements. Supports large files (4000+ rows).
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd

from app.types import ExportSettings, ConversionManifest, FileResult
from app.converters import (
    convert_excel_file, convert_pdf_file, convert_csv_file,
    convert_json_file, convert_ods_file, get_sheet_dataframe
)
from app.exporters import export_to_zip, create_settings_json, load_settings_from_json
from app.gemini_client import create_gemini_client
from app.utils import format_file_size, extract_sheet_preview


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Excel to RAG-Ready Converter",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔄 Excel to RAG-Ready Converter")
    st.markdown(
        "Convert **Excel files** (and other formats like PDF, CSV, JSON, TSV, ODS) to **RAG-optimized JSON/JSONL/Markdown** "
        "with optional **Gemini AI** enhancements for better structure and insights."
    )

    # Initialize session state
    if 'settings' not in st.session_state:
        st.session_state.settings = ExportSettings()
    if 'conversion_results' not in st.session_state:
        st.session_state.conversion_results = None
    if 'file_data' not in st.session_state:
        st.session_state.file_data = {}

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        settings = configure_settings()
        st.session_state.settings = settings

        # App info
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Primary Focus: Excel → RAG Pipeline**
            - Upload Excel files (.xlsx, .xls) - all sheets processed
            - RAG-optimized exports: JSON, JSONL, Markdown
            - AI-powered fact sentences for semantic search
            - Smart header normalization
            - Sheet summaries and insights

            **Additional Formats Supported:**
            - PDF: Table extraction + text content
            - CSV/TSV: Delimiter-separated files
            - JSON: Existing structured data
            - ODS: LibreOffice/OpenOffice Calc

            **RAG-Optimized Features:**
            - JSONL format for efficient chunking
            - Fact sentences for semantic search
            - Structured metadata preservation
            - Non-destructive original data retention

            **Privacy:** Files are processed locally and not stored.
            """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 File Upload")
        uploaded_files = st.file_uploader(
            "Choose files to convert",
            type=['xlsx', 'xls', 'pdf', 'csv', 'tsv', 'json', 'ods'],
            accept_multiple_files=True,
            help="Select Excel (.xlsx, .xls), PDF, CSV, TSV, JSON, or ODS files to convert"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

            # Display file info
            total_size = sum(len(f.getvalue()) for f in uploaded_files)
            st.info(f"📊 Total size: {format_file_size(total_size)}")

            # Process files button
            if st.button("🚀 Convert Files", type="primary", use_container_width=True):
                process_files(uploaded_files, settings)

    with col2:
        st.header("⚙️ Quick Actions")

        # Settings export/import
        st.subheader("Settings")

        # Export settings
        if st.button("📥 Export Settings", use_container_width=True):
            settings_json = create_settings_json(settings)
            st.download_button(
                "Download settings.json",
                settings_json,
                "settings.json",
                "application/json",
                use_container_width=True
            )

        # Import settings
        settings_file = st.file_uploader(
            "Import Settings",
            type=['json'],
            help="Upload a previously exported settings.json file"
        )
        if settings_file:
            try:
                settings_json = settings_file.read().decode()
                imported_settings = load_settings_from_json(settings_json)
                if imported_settings:
                    st.session_state.settings = imported_settings
                    st.rerun()
                else:
                    st.error("Invalid settings file")
            except Exception as e:
                st.error(f"Error loading settings: {str(e)}")

    # Results section
    if st.session_state.conversion_results:
        display_results()


def configure_settings() -> ExportSettings:
    """Configure export settings in the sidebar."""
    st.subheader("Output Formats")

    export_json = st.checkbox("📄 JSON", value=st.session_state.settings.export_json,
                             help="Export as JSON arrays with metadata")
    export_jsonl = st.checkbox("📝 JSONL", value=st.session_state.settings.export_jsonl,
                              help="Export as JSON Lines (one object per line, great for RAG)")
    export_markdown = st.checkbox("📖 Markdown", value=st.session_state.settings.export_markdown,
                                 help="Export as Markdown tables with summaries")

    st.subheader("🤖 Gemini AI Features")

    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.settings.gemini_api_key or "",
            help="Enter your Google Gemini API key for AI features"
        )

    use_gemini = st.checkbox(
        "Enable Gemini AI",
        value=st.session_state.settings.use_gemini and bool(api_key),
        disabled=not bool(api_key),
        help="Enable AI-powered enhancements (requires API key)"
    )

    gemini_headers = False
    gemini_summary = False
    gemini_facts = False

    if use_gemini and api_key:
        with st.container():
            st.caption("⚡ AI Features (may incur API costs)")
            gemini_headers = st.checkbox(
                "Smart Headers",
                value=st.session_state.settings.gemini_headers,
                help="Suggest normalized, machine-friendly column names"
            )
            gemini_summary = st.checkbox(
                "Sheet Summaries",
                value=st.session_state.settings.gemini_summary,
                help="Generate 3-5 bullet point summaries of each sheet"
            )
            gemini_facts = st.checkbox(
                "Fact Sentences",
                value=st.session_state.settings.gemini_facts,
                help="Create natural language sentences from data rows (great for RAG)"
            )

            # Test connection
            if st.button("🧪 Test Gemini Connection", use_container_width=True):
                test_gemini_connection(api_key)

    return ExportSettings(
        export_json=export_json,
        export_jsonl=export_jsonl,
        export_markdown=export_markdown,
        use_gemini=use_gemini,
        gemini_headers=gemini_headers,
        gemini_summary=gemini_summary,
        gemini_facts=gemini_facts,
        gemini_api_key=api_key if api_key else None
    )


def test_gemini_connection(api_key: str):
    """Test the Gemini API connection."""
    with st.spinner("Testing Gemini connection..."):
        client = create_gemini_client(api_key)
        if client and client.test_connection():
            st.success("✅ Gemini API connected successfully!")
        else:
            st.error("❌ Failed to connect to Gemini API. Check your API key.")


def process_files(uploaded_files: List, settings: ExportSettings):
    """Process uploaded files and generate results."""
    start_time = time.time()

    # Create Gemini client if enabled
    gemini_client = None
    if settings.use_gemini and settings.gemini_api_key:
        gemini_client = create_gemini_client(settings.gemini_api_key)
        if not gemini_client:
            st.warning("⚠️ Failed to initialize Gemini client. Proceeding without AI features.")

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    file_results = []
    file_data = {}
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = i / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}...")

        # Store file data for later export
        file_bytes = uploaded_file.getvalue()
        file_data[uploaded_file.name] = file_bytes

        # Process based on file type
        file_extension = uploaded_file.name.lower().split('.')[-1]

        try:
            if file_extension in ['xlsx', 'xls']:
                result = convert_excel_file(file_bytes, uploaded_file.name, settings, gemini_client)
            elif file_extension == 'pdf':
                result = convert_pdf_file(file_bytes, uploaded_file.name, settings, gemini_client)
            elif file_extension in ['csv', 'tsv']:
                result = convert_csv_file(file_bytes, uploaded_file.name, settings, gemini_client)
            elif file_extension == 'json':
                result = convert_json_file(file_bytes, uploaded_file.name, settings, gemini_client)
            elif file_extension == 'ods':
                result = convert_ods_file(file_bytes, uploaded_file.name, settings, gemini_client)
            else:
                st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
                continue

            file_results.append(result)

            # Show file progress
            with results_container:
                show_file_progress(result, i + 1, total_files)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

    # Complete processing
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")

    # Create manifest
    processing_time = time.time() - start_time
    manifest = ConversionManifest(
        timestamp=datetime.now(),
        total_files=len(file_results),
        successful_files=sum(1 for r in file_results if r.successful_sheets > 0),
        total_sheets=sum(r.total_sheets for r in file_results),
        successful_sheets=sum(r.successful_sheets for r in file_results),
        total_rows=sum(
            sum(sheet.row_count for sheet in r.sheet_results if sheet.has_data)
            for r in file_results
        ),
        settings=settings,
        file_results=file_results,
        processing_time_seconds=processing_time
    )

    # Store results in session state
    st.session_state.conversion_results = {
        'manifest': manifest,
        'file_results': file_results,
        'file_data': file_data
    }


def show_file_progress(result: FileResult, file_num: int, total_files: int):
    """Display progress for individual file processing."""
    with st.expander(f"📁 {result.file_name} ({file_num}/{total_files})", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sheets", f"{result.successful_sheets}/{result.total_sheets}")
        with col2:
            total_rows = sum(sheet.row_count for sheet in result.sheet_results if sheet.has_data)
            st.metric("Rows", total_rows)
        with col3:
            st.metric("Processing Time", f"{result.processing_time_seconds:.1f}s")

        # Show sheet details
        if result.sheet_results:
            sheet_data = []
            for sheet in result.sheet_results:
                sheet_data.append({
                    "Sheet": sheet.sheet_name,
                    "Rows": sheet.row_count if sheet.has_data else 0,
                    "Status": "✅ Success" if sheet.has_data else "❌ Empty/Failed",
                    "AI Enhanced": "🤖" if (sheet.suggested_headers or sheet.summary or sheet.fact_sentences) else ""
                })

            if sheet_data:
                st.dataframe(pd.DataFrame(sheet_data), use_container_width=True, hide_index=True)

        # Show errors if any
        if result.errors:
            with st.expander("⚠️ Warnings", expanded=False):
                for error in result.errors:
                    st.warning(f"{error.error_type}: {error.error_message}")


def display_results():
    """Display conversion results and download options."""
    results = st.session_state.conversion_results
    manifest = results['manifest']
    file_results = results['file_results']
    file_data = results['file_data']

    st.header("🎉 Conversion Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Files Processed", f"{manifest.successful_files}/{manifest.total_files}")
    with col2:
        st.metric("Sheets Processed", f"{manifest.successful_sheets}/{manifest.total_sheets}")
    with col3:
        st.metric("Total Rows", manifest.total_rows)
    with col4:
        st.metric("Processing Time", f"{manifest.processing_time_seconds:.1f}s")

    # Download section
    st.subheader("📥 Download Results")

    try:
        # Generate ZIP file
        with st.spinner("Creating ZIP file..."):
            zip_data = export_to_zip(file_results, manifest, file_data, manifest.settings)

        # Download button
        zip_filename = f"excel_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        st.download_button(
            label="🗂️ Download All Results (ZIP)",
            data=zip_data,
            file_name=zip_filename,
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

        st.success(f"✅ ZIP file ready! Contains {len([s for f in file_results for s in f.sheet_results if s.has_data])} processed sheets.")

    except Exception as e:
        st.error(f"❌ Error creating ZIP file: {str(e)}")

    # Preview section
    st.subheader("👁️ Data Preview")

    # File selector for preview
    preview_files = [f.file_name for f in file_results if f.successful_sheets > 0]
    if preview_files:
        selected_file = st.selectbox("Select file to preview:", preview_files)

        # Find the selected file result
        file_result = next(f for f in file_results if f.file_name == selected_file)
        preview_sheets = [s.sheet_name for s in file_result.sheet_results if s.has_data]

        if preview_sheets:
            selected_sheet = st.selectbox("Select sheet to preview:", preview_sheets)

            # Show preview
            show_sheet_preview(selected_file, selected_sheet, file_data)

    # Clear results button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.conversion_results = None
        st.session_state.file_data = {}
        st.rerun()


def show_sheet_preview(file_name: str, sheet_name: str, file_data: Dict[str, bytes]):
    """Show a preview of the selected sheet."""
    try:
        df = get_sheet_dataframe(file_data[file_name], file_name, sheet_name)

        if df is not None and not df.empty:
            preview = extract_sheet_preview(df, max_rows=10)

            st.info(f"📊 Showing {preview['preview_rows']} of {preview['total_rows']} rows")

            # Display the preview
            if preview['rows']:
                preview_df = pd.DataFrame(preview['rows'], columns=preview['headers'])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No data to preview")
        else:
            st.warning("Unable to load preview for this sheet")

    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")


if __name__ == "__main__":
    main()