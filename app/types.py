"""Data types and result structures for the Excel conversion pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ProcessingError:
    """Represents an error that occurred during processing."""
    file_name: str
    sheet_name: Optional[str]
    error_message: str
    error_type: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SheetResult:
    """Result of processing a single sheet."""
    file_name: str
    sheet_name: str
    original_headers: List[str]
    suggested_headers: Optional[List[str]]
    row_count: int
    summary: Optional[str]
    fact_sentences: Optional[List[str]]
    has_data: bool
    processing_time_seconds: float
    errors: List[ProcessingError] = field(default_factory=list)


@dataclass
class FileResult:
    """Result of processing an entire file."""
    file_name: str
    file_type: str
    total_sheets: int
    successful_sheets: int
    sheet_results: List[SheetResult]
    processing_time_seconds: float
    errors: List[ProcessingError] = field(default_factory=list)


@dataclass
class ExportSettings:
    """User-configurable export settings."""
    export_json: bool = True
    export_jsonl: bool = True
    export_markdown: bool = True
    use_gemini: bool = False
    gemini_headers: bool = False
    gemini_summary: bool = False
    gemini_facts: bool = False
    gemini_api_key: Optional[str] = None


@dataclass
class ConversionManifest:
    """Manifest describing the entire conversion operation."""
    timestamp: datetime
    total_files: int
    successful_files: int
    total_sheets: int
    successful_sheets: int
    total_rows: int
    settings: ExportSettings
    file_results: List[FileResult]
    errors: List[ProcessingError] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'total_sheets': self.total_sheets,
            'successful_sheets': self.successful_sheets,
            'total_rows': self.total_rows,
            'processing_time_seconds': self.processing_time_seconds,
            'settings': {
                'export_json': self.settings.export_json,
                'export_jsonl': self.settings.export_jsonl,
                'export_markdown': self.settings.export_markdown,
                'use_gemini': self.settings.use_gemini,
                'gemini_headers': self.settings.gemini_headers,
                'gemini_summary': self.settings.gemini_summary,
                'gemini_facts': self.settings.gemini_facts,
            },
            'file_results': [
                {
                    'file_name': fr.file_name,
                    'file_type': fr.file_type,
                    'total_sheets': fr.total_sheets,
                    'successful_sheets': fr.successful_sheets,
                    'processing_time_seconds': fr.processing_time_seconds,
                    'sheet_results': [
                        {
                            'sheet_name': sr.sheet_name,
                            'row_count': sr.row_count,
                            'has_data': sr.has_data,
                            'processing_time_seconds': sr.processing_time_seconds,
                            'original_headers': sr.original_headers,
                            'suggested_headers': sr.suggested_headers,
                            'has_summary': sr.summary is not None,
                            'has_facts': sr.fact_sentences is not None,
                        }
                        for sr in fr.sheet_results
                    ],
                    'errors': [
                        {
                            'sheet_name': err.sheet_name,
                            'error_type': err.error_type,
                            'error_message': err.error_message,
                            'timestamp': err.timestamp.isoformat(),
                        }
                        for err in fr.errors
                    ]
                }
                for fr in self.file_results
            ],
            'errors': [
                {
                    'file_name': err.file_name,
                    'sheet_name': err.sheet_name,
                    'error_type': err.error_type,
                    'error_message': err.error_message,
                    'timestamp': err.timestamp.isoformat(),
                }
                for err in self.errors
            ]
        }