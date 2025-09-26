# 🔄 Excel to RAG-Ready Converter

A production-quality Streamlit application that converts Excel files (and other data formats) to RAG-optimized JSON, JSONL, and Markdown formats with optional AI enhancements via Google's Gemini API.

## ✨ Features

### 🎯 Primary Focus: Excel → RAG Pipeline
- **Excel-First**: Optimized for Excel (.xlsx, .xls) with all sheets processed automatically
- **RAG-Optimized Outputs**:
  - **JSONL**: Perfect for embedding and chunking (recommended for RAG)
  - **JSON**: Structured data with metadata and AI insights
  - **Markdown**: Human-readable tables with AI summaries
- **Complete Processing**: Every sheet, every row, with 100% fidelity
- **Single ZIP Download**: All outputs organized and ready for ingestion

### 📂 Additional Format Support
- **PDF**: Table extraction + text content
- **CSV/TSV**: Delimiter-separated files
- **JSON**: Existing structured data enhancement
- **ODS**: LibreOffice/OpenOffice Calc files

### 🤖 AI Enhancements for RAG (Optional)
Powered by Google Gemini 2.5 Flash:

- **Smart Headers**: Suggests normalized, machine-friendly column names without altering original data
- **Sheet Summaries**: Generates 3-5 bullet point descriptions of each sheet's content
- **Fact Sentences**: Creates natural language sentences from data rows (**perfect for RAG semantic search**)

### 🔒 Data Integrity
- **Non-destructive**: Always preserves original column names and data
- **Error Resilient**: Continues processing if individual sheets fail
- **Local Processing**: Files are processed locally and never stored permanently

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
```bash
git clone <repository-url>
cd excel_any_streamlit
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Optional: Configure Gemini API**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
# Or create a .env file with: GOOGLE_API_KEY=your-gemini-api-key
```

3. **Run the Application**
```bash
streamlit run streamlit_app.py
```

4. **Open Browser**
Navigate to `http://localhost:8501`

### 🌐 Streamlit Community Cloud Deployment

1. **Connect Repository**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set `streamlit_app.py` as the main file path

2. **Configure Secrets** (for Gemini features)
   - In your app settings, go to "Secrets"
   - Add: `GOOGLE_API_KEY = "your-gemini-api-key"`

3. **Deploy**
   - Your app will be available at: `https://your-app-name.streamlit.app`

## 📖 Usage Guide

### Basic Workflow

1. **Upload Files**: Drag and drop Excel files (primary) or other supported formats
2. **Configure RAG Settings**: Choose JSONL + AI fact sentences for optimal RAG performance
3. **Process**: Click "Convert Files" to start processing
4. **Download**: Get a ZIP file optimized for RAG ingestion

### Output Structure

The generated ZIP file contains:

```
excel_conversion_20240101_120000.zip
├── json/                          # JSON files with metadata
│   ├── workbook1__sheet1.json
│   └── workbook1__sheet2.json
├── jsonl/                         # JSONL files (one record per line)
│   ├── workbook1__sheet1.jsonl
│   └── workbook1__sheet2.jsonl
├── markdown/                      # Markdown tables with summaries
│   ├── workbook1__sheet1.md
│   └── workbook1__sheet2.md
├── fact_sentences.jsonl           # AI-generated fact sentences (if enabled)
├── manifest.json                  # Processing summary and metadata
└── settings.json                  # Export settings used
```

### Settings Management

- **Export Settings**: Save your configuration for reuse
- **Import Settings**: Load previously saved settings
- **Session Persistence**: Settings are maintained during the current session

## 🛠️ Technical Details

### Architecture

- **Modular Design**: Clean separation between conversion, export, and UI logic
- **Type Safety**: Full type annotations with dataclasses
- **Error Handling**: Graceful failure handling with detailed error reporting
- **Performance**: Streaming JSONL export and efficient pandas operations

### Dependencies

```
streamlit          # Web application framework
pandas             # Data manipulation and analysis
openpyxl           # Excel file reading (.xlsx)
xlrd               # Legacy Excel file reading (.xls)
pdfplumber         # PDF text and table extraction
google-generativeai # Gemini AI API client
orjson             # Fast JSON serialization
tqdm               # Progress bars
python-dotenv      # Environment variable management
```

### File Processing

- **Header Detection**: Automatically detects header rows in Excel sheets
- **Merged Cell Handling**: Forward-fills merged cells for proper data structure
- **Data Validation**: Ensures data quality and skips empty/invalid sheets
- **Memory Efficient**: Processes large files without loading everything into memory

## 🤖 Gemini AI Integration

### API Key Setup

**Option 1: Environment Variable**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

**Option 2: .env File**
```
GOOGLE_API_KEY=your-api-key-here
```

**Option 3: Streamlit Secrets** (for cloud deployment)
```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your-api-key-here"
```

### AI Features

1. **Smart Headers**:
   - Analyzes existing headers and sample data
   - Suggests machine-friendly column names
   - Preserves original headers alongside suggestions

2. **Sheet Summaries**:
   - Generates concise, factual descriptions
   - Identifies data types and potential use cases
   - Uses neutral, domain-agnostic language

3. **Fact Sentences**:
   - Converts tabular data to natural language
   - Optimized for semantic search and RAG systems
   - One sentence per data row

### Cost Considerations

Gemini API usage incurs costs based on:
- Number of input tokens (your data)
- Number of output tokens (generated content)
- Model version (Gemini 2.5 Flash is cost-efficient)

Estimated costs for typical usage:
- Small file (100 rows): ~$0.01-0.05
- Medium file (1,000 rows): ~$0.10-0.50
- Large file (10,000 rows): ~$1-5

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Required for Gemini features
GOOGLE_API_KEY=your-gemini-api-key

# Optional: Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Custom Prompts

To modify AI behavior, edit the prompts in `app/prompts.py`:

- `HEADER_SUGGESTION_PROMPT`: Controls header normalization
- `SHEET_SUMMARY_PROMPT`: Controls summary generation
- `ROW_FACT_SENTENCE_PROMPT`: Controls fact sentence creation

## 🐛 Troubleshooting

### Common Issues

**"No module named 'app'"**
```bash
# Ensure you're running from the project root directory
cd excel_any_streamlit
python -m streamlit run streamlit_app.py
```

**Gemini API Errors**
- Verify your API key is correct
- Check your Google Cloud billing is enabled
- Ensure the Gemini API is enabled in your project

**Large File Processing**
- Files over 100MB may cause memory issues
- Consider processing fewer files at once
- PDF processing is more memory-intensive than Excel

**Excel Reading Issues**
- Ensure files are not password-protected
- Some very old Excel formats may not be supported
- Corrupted files will be skipped with warnings

### Performance Tips

- **Disable AI features** for faster processing of large datasets
- **Process files in batches** rather than uploading everything at once
- **Use JSONL format** for the most efficient output for large datasets

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest

# Format code
black .
```

## 📞 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Made with ❤️ and Streamlit**