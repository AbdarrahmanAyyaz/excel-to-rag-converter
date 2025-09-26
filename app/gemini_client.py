"""Gemini AI client wrapper for Excel data processing."""

import json
import time
from typing import List, Optional, Any, Dict
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .prompts import build_header_prompt, build_summary_prompt, build_fact_prompt
from .types import ProcessingError


class GeminiClient:
    """Wrapper for Gemini AI API interactions."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """Initialize the Gemini client."""
        self.api_key = api_key
        self.model_name = model_name
        self._model = None
        self._setup_client()

    def _setup_client(self):
        """Configure the Gemini API client."""
        try:
            genai.configure(api_key=self.api_key)

            # Configure safety settings to be permissive for data processing
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Initialize the model
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=safety_settings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def suggest_headers(self, headers: List[str], sample_rows: List[List[Any]]) -> Optional[List[str]]:
        """
        Suggest normalized header names for the given headers and sample data.
        Returns None if suggestion fails or if suggested headers don't match input length.
        """
        if not headers or not sample_rows:
            return None

        try:
            prompt = build_header_prompt(headers, sample_rows)
            response = self._model.generate_content(prompt)

            if not response.text:
                return None

            # Try to parse the JSON response
            suggested_headers = json.loads(response.text.strip())

            # Validate the response
            if (isinstance(suggested_headers, list) and
                len(suggested_headers) == len(headers)):
                return [str(header) for header in suggested_headers]
            else:
                return None

        except json.JSONDecodeError:
            # If JSON parsing fails, return None
            return None
        except Exception as e:
            # Log the error but don't raise it
            print(f"Header suggestion error: {str(e)}")
            return None

    def generate_sheet_summary(self, sheet_name: str, headers: List[str],
                              row_count: int, sample_rows: List[List[Any]]) -> Optional[str]:
        """Generate a summary of what the sheet contains."""
        if not headers or not sample_rows:
            return None

        try:
            prompt = build_summary_prompt(sheet_name, headers, row_count, sample_rows)
            response = self._model.generate_content(prompt)

            if response.text and response.text.strip():
                return response.text.strip()
            else:
                return None

        except Exception as e:
            print(f"Sheet summary error: {str(e)}")
            return None

    def generate_fact_sentences(self, headers: List[str], rows: List[List[Any]],
                               max_rows: Optional[int] = None) -> List[str]:
        """
        Generate fact sentences for rows of data.
        Process in batches to avoid API rate limits.
        """
        if not headers or not rows:
            return []

        fact_sentences = []
        rows_to_process = rows[:max_rows] if max_rows else rows

        for i, row in enumerate(rows_to_process):
            try:
                # Skip empty rows
                if not any(str(val).strip() for val in row):
                    continue

                prompt = build_fact_prompt(headers, row)
                response = self._model.generate_content(prompt)

                if response.text and response.text.strip():
                    fact_sentences.append(response.text.strip())

                # Small delay to avoid rate limits
                if i > 0 and i % 10 == 0:
                    time.sleep(1)

            except Exception as e:
                print(f"Fact sentence error for row {i}: {str(e)}")
                continue

        return fact_sentences

    def test_connection(self) -> bool:
        """Test if the Gemini API connection is working."""
        try:
            test_prompt = "Return the word 'test' as JSON: {\"result\": \"test\"}"
            response = self._model.generate_content(test_prompt)
            return response.text is not None
        except Exception:
            return False

    @property
    def is_configured(self) -> bool:
        """Check if the client is properly configured."""
        return self._model is not None and bool(self.api_key)


def create_gemini_client(api_key: Optional[str]) -> Optional[GeminiClient]:
    """Factory function to create a Gemini client if API key is provided."""
    if not api_key or not api_key.strip():
        return None

    try:
        client = GeminiClient(api_key.strip())
        if client.test_connection():
            return client
        else:
            return None
    except Exception:
        return None