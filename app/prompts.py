"""Gemini AI prompts for Excel data processing."""

from typing import List, Dict, Any


HEADER_SUGGESTION_PROMPT = """You receive spreadsheet column headers and a few sample rows.
Return ONLY a JSON array of suggested, short, machine-friendly column names, same length and order as input.
If you cannot safely suggest names without changing meaning, return the original headers unchanged.
Do not include additional commentary.

Original headers: {headers}
Sample rows (first 3):
{sample_rows}

Return only the JSON array of suggested column names:"""


SHEET_SUMMARY_PROMPT = """In 3–5 concise bullet points, summarize what this sheet contains (types of entities/attributes) and how someone might use it.
Be strictly factual and neutral.

Sheet name: {sheet_name}
Column headers: {headers}
Number of rows: {row_count}
Sample data (first 3 rows):
{sample_rows}

Provide your summary as bullet points:"""


ROW_FACT_SENTENCE_PROMPT = """Write a single, concise sentence that restates this row's content clearly for search and retrieval.
Keep it factual and brief. Do not speculate.

Column headers: {headers}
Row data: {row_data}

Fact sentence:"""


def format_headers_for_prompt(headers: List[str]) -> str:
    """Format headers list for inclusion in prompts."""
    return ", ".join(f'"{header}"' for header in headers)


def format_sample_rows_for_prompt(rows: List[List[Any]], max_rows: int = 3) -> str:
    """Format sample rows for inclusion in prompts."""
    if not rows:
        return "No data available"

    sample_rows = rows[:max_rows]
    formatted_rows = []

    for i, row in enumerate(sample_rows, 1):
        # Convert row values to strings and truncate long values
        row_strs = [str(val)[:50] + "..." if len(str(val)) > 50 else str(val) for val in row]
        formatted_rows.append(f"Row {i}: [{', '.join(repr(val) for val in row_strs)}]")

    return "\n".join(formatted_rows)


def format_row_for_fact_prompt(headers: List[str], row: List[Any]) -> Dict[str, str]:
    """Format a single row for the fact sentence prompt."""
    # Create a mapping of headers to values
    row_data = {}
    for i, header in enumerate(headers):
        value = row[i] if i < len(row) else ""
        # Skip empty values
        if str(value).strip():
            row_data[header] = str(value)

    return {
        'headers': format_headers_for_prompt(headers),
        'row_data': ", ".join(f"{k}: {repr(v)}" for k, v in row_data.items())
    }


def build_header_prompt(headers: List[str], sample_rows: List[List[Any]]) -> str:
    """Build the complete header suggestion prompt."""
    return HEADER_SUGGESTION_PROMPT.format(
        headers=format_headers_for_prompt(headers),
        sample_rows=format_sample_rows_for_prompt(sample_rows)
    )


def build_summary_prompt(sheet_name: str, headers: List[str], row_count: int, sample_rows: List[List[Any]]) -> str:
    """Build the complete sheet summary prompt."""
    return SHEET_SUMMARY_PROMPT.format(
        sheet_name=sheet_name,
        headers=format_headers_for_prompt(headers),
        row_count=row_count,
        sample_rows=format_sample_rows_for_prompt(sample_rows)
    )


def build_fact_prompt(headers: List[str], row: List[Any]) -> str:
    """Build the complete fact sentence prompt for a single row."""
    row_info = format_row_for_fact_prompt(headers, row)
    return ROW_FACT_SENTENCE_PROMPT.format(**row_info)