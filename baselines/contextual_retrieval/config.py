# -*- coding: utf-8 -*-
"""Contextual Retrieval baseline configuration

Code: this directory
Data: ./data (a one-time export snapshot; this package only writes its own logs).
All paths below are resolved relative to this file.
"""
import os
from pathlib import Path

BASE_CODE = Path(__file__).resolve().parent
BASE_DATA = BASE_CODE
REPO_ROOT = BASE_CODE.parent.parent

DATA = BASE_DATA / 'data'
SOURCE = DATA / 'source'
SPLIT_OUT = BASE_DATA / 'split_output' / 'json'
CHUNKS_DIR = DATA / 'chunks'
EMB_DIR = DATA / 'embeddings'
RUNS_DIR = DATA / 'runs'
CONTEXTS_DIR = DATA / 'contexts'
ANSWERS_DIR = DATA / 'answers'
JUDG_DIR = DATA / 'judgments'
REPORTS_DIR = BASE_DATA / 'reports'

PROMPTS_DIR = BASE_CODE / 'prompts'
LOGS_DIR = BASE_CODE / 'logs'

# ---- One-off export sources (used only by 00_build_source_chunks.py; never touched afterwards) ----
MAIN_DIR = REPO_ROOT / 'src' / 'retrieval'
MAIN_CHUNKS_DIR = MAIN_DIR / 'data' / 'chunks' / 'by_book'
L1_OUTLINE_DIR = REPO_ROOT / 'example_data'

# ---- Scheme identifier ----
SCHEME = 'ctxretrieval'

# ---- SGLang (local Qwen3-4B-Instruct-2507, same framework as the proposed method) ----
SGLANG_ENDPOINT = 'http://127.0.0.1:30000'
SGLANG_MODEL_PATH = os.environ.get('CHUNKING_MODEL_PATH', '<MODEL_DIR>/Qwen3-4B-Instruct-2507')
SGLANG_CONCURRENCY = 16          # aligned with the server-side --max-running-requests
CR_MAX_NEW_TOKENS = 256          # description <=100 words; headroom left
CR_TEMPERATURE = 0.1
CR_TOP_P = 0.5
CR_TOP_K = 15
DOC_EXCERPT_CHARS = 8000         # chapter excerpt cap (<document> part)
PROMPT_MAX_CHARS = 20000         # client-side guard (normal prompts stay under 12k chars; never triggers)
CR_WORDS_SOFT_MAX = 120          # above this is treated as an anomaly (retry once)

# ---- Embeddings (local Qwen3-Embedding-0.6B) ----
EMBED_MODEL_PATH = Path(os.environ.get('EMBEDDING_MODEL_PATH', '<MODEL_DIR>/Qwen3-Embedding-0.6B'))
EMBED_DEVICE = 'cuda'
EMBED_BATCH = 32
TOPK = 10
ANSWER_TOPK = 5

# ---- Generation/judging (API, same protocol as the main experiment) ----
MODEL_GEN = 'qwen3.6-plus'
MODEL_JUDGE = 'deepseek-v4-pro'
GEN_TEMP = 0.2
JUDGE_TEMP = 0.0
WORKERS = 3
DIMS = ['grounding', 'correctness', 'hallucination_absence']

# ---- Miscellaneous ----
SEED = 2026
PUNCT = '。！？.!?'
