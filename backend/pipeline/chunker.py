"""
backend/pipeline/chunker.py
────────────────────────────
Sentence-aware text chunker with configurable size and overlap.
Respects paragraph boundaries where possible for cleaner chunks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_number: int | None
    char_start: int
    char_end: int
    token_count: int


def estimate_tokens(text: str) -> int:
    """Fast token count estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK needed)."""
    # Split on sentence boundaries: ., !, ? followed by space + capital
    pattern = r'(?<=[.!?])\s+(?=[A-ZÜÖÄ])'
    sentences = re.split(pattern, text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for sent in sentences:
        parts = sent.split('\n\n')
        result.extend(p.strip() for p in parts if p.strip())
    return result if result else [text]


class SemanticChunker:
    """
    Chunks text by accumulating sentences until target token size,
    then overlapping by `overlap` tokens into the next chunk.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size

    def chunk_text(
        self,
        text: str,
        page_number: int | None = None,
        start_chunk_index: int = 0,
    ) -> list[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk
            page_number: Source page number (for citation)
            start_chunk_index: Starting index for chunk numbering

        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []

        sentences = split_into_sentences(text)
        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        char_cursor = 0
        chunk_idx = start_chunk_index

        for sentence in sentences:
            sent_tokens = estimate_tokens(sentence)

            # If adding this sentence exceeds chunk size, flush current chunk
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                if estimate_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        page_number=page_number,
                        char_start=char_cursor,
                        char_end=char_cursor + len(chunk_text),
                        token_count=current_tokens,
                    ))
                    chunk_idx += 1
                    char_cursor += len(chunk_text)

                # Overlap: keep last N tokens worth of sentences
                overlap_sentences = []
                overlap_tokens = 0
                for prev_sent in reversed(current_sentences):
                    t = estimate_tokens(prev_sent)
                    if overlap_tokens + t <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += t
                    else:
                        break

                current_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sent_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sent_tokens

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    page_number=page_number,
                    char_start=char_cursor,
                    char_end=char_cursor + len(chunk_text),
                    token_count=current_tokens,
                ))

        logger.debug("Text chunked", page=page_number, chunk_count=len(chunks))
        return chunks

    def chunk_pages(self, pages: list[tuple[int, str]]) -> list[TextChunk]:
        """
        Chunk multiple pages, maintaining page numbers in metadata.
        Args: pages = [(page_number, text), ...]
        """
        all_chunks: list[TextChunk] = []
        chunk_idx = 0
        for page_num, text in pages:
            page_chunks = self.chunk_text(text, page_number=page_num, start_chunk_index=chunk_idx)
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)
        return all_chunks
