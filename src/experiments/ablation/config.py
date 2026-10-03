#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation experiment configuration"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent.parent
EXAMPLE_DATA_DIR = REPO_ROOT / "example_data"
EXAMPLE_BOOK_DIR = EXAMPLE_DATA_DIR / "Far From the Madding Crowd"

# Model path
MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "<MODEL_DIR>/Qwen3-Embedding-0.6B")

# Data directories
DATA_DIR = BASE_DIR / "data"
SOURCE_SPLITS_DIR = DATA_DIR / "source_splits"
VARIANT_SPLITS_DIR = DATA_DIR / "variant_splits"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_BY_BOOK_DIR = CHUNKS_DIR / "by_book"
QUERIES_DIR = DATA_DIR / "queries"
QUERIES_BY_BOOK_DIR = QUERIES_DIR / "by_book"
QA_DRAFTS_DIR = DATA_DIR / "qa_drafts"
ANSWERS_DIR = DATA_DIR / "answers"
ANSWERS_BY_BOOK_DIR = ANSWERS_DIR / "by_book"

# Embedding directories
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_BY_BOOK_DIR = EMBEDDINGS_DIR / "by_book"

# Retrieval result directories
RUNS_DIR = BASE_DIR / "runs"
RUNS_IN_BOOK_DIR = RUNS_DIR / "in_book"

# Results directories
RESULTS_DIR = BASE_DIR / "results"
JUDGMENTS_DIR = RESULTS_DIR / "judgments"
JUDGMENTS_BY_BOOK_DIR = JUDGMENTS_DIR / "by_book_eval"

# Prompt directories
PROMPTS_DIR = BASE_DIR / "prompts"

# Source data (read-only)
OURS_SPLIT_JSON_DIR = EXAMPLE_BOOK_DIR
OURS_L1OUTLINE_GLOB = "*_L1Outline.json"
OURS_SPLIT_GLOB = "*_split.json"

SOURCE_QA_DIR = EXAMPLE_BOOK_DIR
SOURCE_QUERIES_BY_BOOK_DIR = SOURCE_QA_DIR / "queries"
SOURCE_EMBEDDINGS_BY_BOOK_DIR = BASE_DIR / "embeddings" / "source_queries"
SOURCE_QA_DRAFTS_ALL = SOURCE_QA_DIR / "qa_drafts" / "Far From the Madding Crowd.qa_drafts.json"

# Ablation variants
VARIANTS = ["wo_h1", "wo_h2", "wo_h1h2", "noise_50"]

# Embedding parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cuda"
NORMALIZE_EMBEDDINGS = True

# Retrieval parameters
DEFAULT_TOP_K = 10

# Answer generation parameters
ANSWER_TOP_K = 5
