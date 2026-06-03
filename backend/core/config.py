"""
backend/core/config.py
──────────────────────
Centralised configuration using Pydantic-Settings.
All values can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 120
    ollama_temperature: float = 0.1

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "docusense_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_retrieval: int = 10
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 5
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50

    # ── OCR ───────────────────────────────────────────────────────────────────
    tesseract_cmd: str = "tesseract"
    tesseract_lang: str = "eng+deu"
    ocr_dpi: int = 300

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    max_upload_size_mb: int = 50

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_file: str = "./data/logs/docusense.log"

    # ── Data Paths ───────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    bm25_index_path: str = "./data/bm25_index.pkl"
    chroma_export_path: str = "./data/chroma_export"

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_output_dir: str = "./data/eval_results"

    # ── Feature Flags ────────────────────────────────────────────────────────
    enable_compliance_agent: bool = True
    enable_extraction_agent: bool = True
    enable_reranking: bool = True
    debug_mode: bool = False

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.chroma_persist_dir,
            self.upload_dir,
            self.eval_output_dir,
            Path(self.log_file).parent,
            Path(self.bm25_index_path).parent,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
