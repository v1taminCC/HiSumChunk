# -*- coding: utf-8 -*-
"""Summary faithfulness audit - configuration.

- Input: main-experiment artifacts (read-only). Use environment variable HISUMCHUNK_ROOT to point at the repository root;
  defaults to the parent directory of this package.
- Output: the `results/` directory of this package.
- API: official DeepSeek platform; the key is read from environment variable DEEPSEEK_API_KEY (never write it to a file).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- Package paths ----
ROOT = Path(__file__).resolve().parent               # src/experiments/summary_audit/
CODE_DIR = ROOT / 'scripts'
RESULTS_DIR = ROOT / 'results'
INVENTORY_DIR = RESULTS_DIR / 'inventory'
FRAMES_DIR = RESULTS_DIR / 'frames'
ANNOT_DIR = RESULTS_DIR / 'annotations'
HUMAN_DIR = RESULTS_DIR / 'human'
EXPOSURE_DIR = RESULTS_DIR / 'exposure'
LOGS_DIR = RESULTS_DIR / 'logs'
REPORTS_DIR = RESULTS_DIR
PROMPTS_DIR = CODE_DIR / 'prompts'

# ---- Read-only inputs (main-experiment artifacts) ----
MAIN_ROOT = Path(os.environ.get('HISUMCHUNK_ROOT', str(ROOT.parent.parent.parent)))
SPLIT_DIR = MAIN_ROOT / 'example_data'
CHUNKS_FILE = MAIN_ROOT / 'src' / 'retrieval' / 'data' / 'chunks' / 'chunks_all.ours.json'
ANSWERS_FILE = MAIN_ROOT / 'src' / 'retrieval' / 'data' / 'answers' / 'answers_all.in_book.ours.top5.json'
MAIN_JUDG_DIR = MAIN_ROOT / 'src' / 'retrieval' / 'results' / 'answer_judgments' / 'by_book_3dim'

# ---- API (DeepSeek official, OpenAI-compatible) ----
API_BASE = 'https://api.deepseek.com/v1'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
ENABLE_THINKING = False                      # thinking={"type":"disabled"}
MODEL_AUDIT = 'deepseek-v4-pro'              # claim extraction and labeling
MODEL_FALLBACK = 'deepseek-v4-pro'           # fallback on moderation/parse failure
AUDIT_TEMPERATURE = 0.1
AUDIT_MAX_TOKENS = 8192
DEFAULT_WORKERS = 6

# ---- Audit protocol ----
PROMPT_VERSION = 'v3'                        # prompts/audit_*.v3.txt
FULL_RUN_TAG = 'full_v3'
FULL_RUN_ANN = ANNOT_DIR / f'{FULL_RUN_TAG}.jsonl'
SEED = 2026
COMPONENT_A_SIZE = 400                       # non-retrieved-layer sample size
COMPONENT_A_MIN_PER_STRATUM = 25
