# -*- coding: utf-8 -*-
"""Mechanism (context-variant) experiment configuration: paths and constants.
Source directories are read-only; all outputs are written under this package."""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent      # src/experiments/mechanism/
FOUR = BASE
REPO_ROOT = BASE.parent.parent.parent              # repository root
MAIN = REPO_ROOT / 'src' / 'retrieval'

# ---- Sources (read-only allowlist) ----
QA_DRAFTS = MAIN / 'qa_build' / 'data' / 'qa_drafts' / 'qa_drafts_all.json'
RUNS_DIR = MAIN / 'runs' / 'in_book'
CHUNKS_DIR = MAIN / 'data' / 'chunks' / 'by_book'
OLD_ANSWERS_DIR = MAIN / 'data' / 'answers' / 'by_book'
OLD_JUDG_DIR = MAIN / 'results' / 'answer_judgments' / 'by_book_3dim'
SPLIT_DIR = REPO_ROOT / 'example_data'

# ---- Outputs (written only to this directory) ----
MANIFEST_DIR = FOUR / '00-manifest'
SAMPLE_DIR = FOUR / '01-sample'
CONTEXTS_DIR = FOUR / '02-contexts'
ANSWERS_OUT_DIR = FOUR / '03-answers'
JUDG_OUT_DIR = FOUR / '04-judgments'
REPORTS_DIR = FOUR / '05-reports'
SCRIPTS_DIR = FOUR / 'scripts'
PROMPTS_DIR = FOUR / 'prompts'
LOGS_DIR = FOUR / 'logs'
STATS_PKG = FOUR / 'stats_package'

# ---- Extended outputs (remaining 359, separate directory; never touches the 200-question main outputs) ----
CONTEXTS_EXT_DIR = FOUR / '02-contexts-remaining'
ANSWERS_EXT_DIR = FOUR / '03-answers-remaining'
JUDG_EXT_DIR = FOUR / '04-judgments-remaining'

# ---- 18 fill-question outputs (fill18; separate directory) ----
RUNS_TOP20_DIR = FOUR / 'runs_top20'
CONTEXTS_FILL18_DIR = FOUR / '02-contexts-fill18'
ANSWERS_FILL18_DIR = FOUR / '03-answers-fill18'
JUDG_FILL18_DIR = FOUR / '04-judgments-fill18'

# ---- Result mirrors (aligned with the ablation directory convention; read-only sources, writes only these two trees) ----
RESULTS_JUDG_DIR = FOUR / 'results' / 'judgments' / 'by_book_3dim'
RESULTS_ANSWERS_DIR = FOUR / 'data' / 'answers' / 'by_book'

# ---- Constants ----
ARMS = ['A', 'B', 'C', 'D', 'P']
SEED = 2026
SAMPLE_SIZE = 200
MIN_BOOKS = 18
MIN_BODY_CHUNKS = 5      # minimum number of body blocks in top-10 (questions below this leave the sampling pool)
TOPK = 5
TOPN = 10
PUNCT = '。！？.!?'
MODEL_GEN = 'qwen3.6-plus'
MODEL_JUDGE = 'deepseek-v4-pro'
WORKERS = 3
GEN_TEMP = 0.2
JUDGE_TEMP = 0.0

DIMS = ['grounding', 'correctness', 'hallucination_absence']
SCHEMES_MAIN = ['ours', 'fixed', 'lumberchunker', 'lbdsemantic', 'semanticembed', 'raptor']
