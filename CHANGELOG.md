# Changelog

## [Unreleased] - 2025-10-09

### Added
- Standalone `excel_to_json_converter.py` CLI tool for simple Excel to JSON conversion
- Documentation for standalone CLI tool in README

### Improved
- **Memory-efficient processing for large Excel files (4000+ rows)**
  - Smart two-phase reading: sample first for headers, then full file
  - Reduces memory footprint significantly
  - Uses pandas built-in header detection for better performance
  - Applied to both main processing and preview functions
- Better handling of very large Excel files without running out of memory

### Technical Details
The app now:
1. Reads first 50 rows as sample for header detection
2. Uses pandas `header` parameter for efficient full file reading
3. Maintains all existing features (AI, multi-format, RAG optimization)
4. No breaking changes - fully backward compatible
