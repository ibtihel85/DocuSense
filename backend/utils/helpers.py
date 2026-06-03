"""
backend/utils/helpers.py
────────────────────────
General-purpose utility functions used across the codebase.
"""

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any


def compute_file_hash(file_path: str | Path) -> str:
    """
    Compute SHA-256 hash of a file.
    Used to detect duplicate uploads and as document ID.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of a string (for chunk deduplication)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFC form.
    Important for German umlauts (ä, ö, ü) and other special characters.
    """
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Remove excessive whitespace, normalize line endings."""
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace 3+ newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars for display/logging."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def extract_doc_id_from_filename(filename: str) -> str:
    """
    Generate a clean document ID from filename.
    Removes special characters, lowercases, replaces spaces with underscores.
    """
    stem = Path(filename).stem
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
    clean = re.sub(r"_+", "_", clean).lower()
    return clean[:64]  # Limit length


def safe_json_loads(text: str) -> dict[str, Any] | None:
    """
    Attempt to parse JSON from LLM output.
    Handles common issues like markdown code fences.
    Returns None if parsing fails.
    """
    import json

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object/array in text
    json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def chunk_list(lst: list, n: int) -> list[list]:
    """Split a list into chunks of size n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"
