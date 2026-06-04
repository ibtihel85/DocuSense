"""
backend/ocr/cleaner.py
──────────────────────
Text normalization and cleaning for OCR and parsed document text.
"""
import re
import unicodedata


def clean_ocr_text(text: str) -> str:
    """Full cleaning pipeline for OCR-extracted text."""
    text = unicodedata.normalize("NFC", text)
    text = fix_ocr_artifacts(text)
    text = normalize_whitespace(text)
    text = remove_control_chars(text)
    return text.strip()


def fix_ocr_artifacts(text: str) -> str:
    """Fix common OCR misreads."""
    # Common OCR errors
    replacements = {
        r'\bl\b(?=\s*\d)': '1',   # lowercase l before digit → 1
        r'(?<=\d)\bO\b': '0',     # O between digits → 0
        r'\|': 'I',               # pipe → I (common OCR error)
        r'(?<=[a-z])1(?=[a-z])': 'l',  # 1 between lowercase → l
    }
    for pattern, replacement in replacements.items():
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            pass
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces/tabs, limit newlines."""
    text = re.sub(r'[ \t]+', ' ', text)         # multiple spaces → one
    text = re.sub(r' *\n *', '\n', text)         # spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)       # 3+ newlines → 2
    return text


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters (except newline/tab)."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def clean_pdf_text(text: str) -> str:
    """Cleaning specifically for pypdf-extracted text."""
    text = unicodedata.normalize("NFC", text)
    # Fix broken hyphenation across lines
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Fix ligatures that pypdf sometimes breaks
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = normalize_whitespace(text)
    text = remove_control_chars(text)
    return text.strip()


def extract_language_hint(text: str) -> str:
    """
    Heuristic language detection for short texts.
    Returns 'de' for German, 'en' for English.
    """
    german_markers = ['der', 'die', 'das', 'und', 'ist', 'von', 'mit', 'für', 'auf', 'nicht']
    english_markers = ['the', 'and', 'is', 'of', 'to', 'in', 'for', 'with', 'that', 'this']
    words = set(text.lower().split()[:100])
    de_count = sum(1 for w in german_markers if w in words)
    en_count = sum(1 for w in english_markers if w in words)
    return 'de' if de_count > en_count else 'en'
