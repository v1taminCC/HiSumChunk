# -*- coding: utf-8 -*-
"""Placebo experiment configuration: source snapshots (read-only, all inside this directory) + outputs (written only to this directory)"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
SOURCE = DATA / 'source'

# ---- Source snapshots (inside this directory, read-only) ----
CONTEXTS_A = {
    'main': SOURCE / 'contexts_A' / 'main' / 'contexts_A.json',
    'remaining': SOURCE / 'contexts_A' / 'remaining' / 'contexts_A.json',
    'fill18': SOURCE / 'contexts_A' / 'fill18' / 'contexts_A.json',
}
JUDG_DIRS = {
    'main': SOURCE / 'judgments_ABCD' / 'main' / 'by_book',
    'remaining': SOURCE / 'judgments_ABCD' / 'remaining' / 'by_book',
    'fill18': SOURCE / 'judgments_ABCD' / 'fill18' / 'by_book',
}
OLD_OURS_DIR = SOURCE / 'old_ours' / 'by_book'
H2_POOL = SOURCE / 'h2_pool.jsonl'
CHUNK_PREFIX = SOURCE / 'chunk_prefix.json'
SAMPLE_200 = SOURCE / 'sample_200.json'
REMAINING = SOURCE / 'remaining.json'
QA_DRAFTS = SOURCE / 'qa_drafts_all.json'

# ---- Outputs (written only to this directory) ----
CONTEXTS_DIR = DATA / 'contexts'
ANSWERS_DIR = DATA / 'answers'
JUDG_DIR = DATA / 'judgments'
REPORTS_DIR = BASE / 'reports'
LOGS_DIR = BASE / 'logs'
PROMPTS_DIR = BASE / 'prompts'

# ---- Constants (same as the four arms) ----
ARM = 'P'
DIMS = ['grounding', 'correctness', 'hallucination_absence']
MODEL_GEN = 'qwen3.6-plus'
MODEL_JUDGE = 'deepseek-v4-pro'
GEN_TEMP = 0.2
JUDGE_TEMP = 0.0
WORKERS = 3
SEED = 2026

# ---- Placebo construction parameters ----
PLACEBO_LEN_TOL = 0.20   # length tolerance +/-20%
PLACEBO_CAND_N = 20      # pick randomly among the N length-closest candidates
PUNCT = '。！？.!?'
