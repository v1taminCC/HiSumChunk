from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent.parent
EXAMPLE_DATA_DIR = REPO_ROOT / "example_data"
EXAMPLE_BOOK_DIR = EXAMPLE_DATA_DIR / "Far From the Madding Crowd"

BOOKSUM_SPLIT_JSON_DIR = EXAMPLE_BOOK_DIR
BOOKSUM_RAW_DIR = EXAMPLE_BOOK_DIR
SOURCE_RETRIEVAL_DIR = REPO_ROOT / "src" / "retrieval"
MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "<MODEL_DIR>/Qwen3-Embedding-0.6B")

DATA_DIR = BASE_DIR / "data"
SOURCE_CHAPTER_DIR = DATA_DIR / "source_chapters"
QA_DRAFT_DIR = DATA_DIR / "qa_drafts"
QA_REVIEW_DIR = DATA_DIR / "qa_review"
QA_FINAL_DIR = DATA_DIR / "qa_final"
PROMPT_DIR = BASE_DIR / "prompts"

SOURCE_CHAPTER_BY_BOOK_DIR = SOURCE_CHAPTER_DIR / "by_book"
QA_DRAFT_BY_BOOK_DIR = QA_DRAFT_DIR / "by_book"
QA_FINAL_BY_BOOK_DIR = QA_FINAL_DIR / "by_book"

# API config now in utils_qwen_api.py (single model, single key)
QWEN_API_BASE = os.getenv("API_BASE_URL", "")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen3.6-plus"

DEFAULT_QA_PER_CHAPTER = 4
DEFAULT_MAX_CHAPTER_CHARS = 16000
DEFAULT_REVIEW_SAMPLE_SIZE = 80
ALLOWED_QUESTION_TYPES = [
    "fact",
    "causal",
    "relation",
    "event_transition",
    "evidence_summary",
]
ALLOWED_DIFFICULTIES = ["easy", "medium", "hard"]
