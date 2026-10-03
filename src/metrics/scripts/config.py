from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR_CURRENT = Path(__file__).resolve().parent
WORK_DIR = SCRIPTS_DIR_CURRENT.parent
REPO_ROOT = WORK_DIR.parent.parent
SOURCE_ROOT = REPO_ROOT / "example_data" / "Far From the Madding Crowd"

SOURCE_ANSWERS_DIR = SOURCE_ROOT / "answers"
SOURCE_JUDGMENTS_DIR = SOURCE_ROOT / "judgments"
SOURCE_QA_DRAFTS = SOURCE_ROOT / "qa_drafts" / "Far From the Madding Crowd.qa_drafts.json"

DATA_DIR = WORK_DIR / "data"
ANSWERS_DIR = DATA_DIR / "answers" / "by_book"
JUDGMENTS_DIR = DATA_DIR / "judgments_eval" / "by_book"
QA_DRAFTS_DIR = DATA_DIR / "qa_drafts"
METADATA_DIR = DATA_DIR / "metadata"
PREPARED_DIR = DATA_DIR / "prepared"

SCRIPTS_DIR = WORK_DIR / "scripts"
RESULTS_DIR = WORK_DIR / "results"
PER_ITEM_DIR = RESULTS_DIR / "per_item"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = WORK_DIR / "logs"

SCHEMES = ["ours", "raptor", "fixed", "lumberchunker", "lbdsemantic", "semanticembed"]
ANSWER_FILE_PATTERN = "*.in_book.{scheme}.top5.json"
JUDGMENT_FILE_PATTERN = "*.eval.json"

INPUT_MANIFEST_PATH = METADATA_DIR / "input_manifest.json"
PREPARED_RECORDS_PATH = PREPARED_DIR / "eval_records.json"
PREPARE_STATS_PATH = REPORTS_DIR / "prepare_stats.txt"

CONTEXT_METRICS_PATH = PER_ITEM_DIR / "context_metrics.json"
ENTITY_METRICS_PATH = PER_ITEM_DIR / "entity_metrics.json"
AUTO_METRICS_ALL_PATH = PER_ITEM_DIR / "auto_metrics_all.json"
AUTO_METRICS_BY_SCHEME_JSON_PATH = TABLES_DIR / "auto_metrics_by_scheme.json"
AUTO_METRICS_BY_SCHEME_CSV_PATH = TABLES_DIR / "auto_metrics_by_scheme.csv"
AUTO_METRICS_SUMMARY_PATH = REPORTS_DIR / "auto_metrics_summary.txt"

BERTSCORE_MODEL = "roberta-base"
BERTSCORE_LANG = "en"
BERTSCORE_BATCH_SIZE = 32
BERTSCORE_RESCALE_WITH_BASELINE = False
BERTSCORE_DEVICE = None

SPACY_MODEL = "en_core_web_sm"
ENTITY_LABELS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]

MAX_CONTEXT_SENTENCES = 120
MIN_SENTENCE_CHARS = 8
TEXT_TRUNCATION_CHARS = 20000

ROUGE_USE_STEMMER = True

CSV_FLOAT_PRECISION = 6

DIRECTORIES_TO_CREATE = [
    ANSWERS_DIR,
    JUDGMENTS_DIR,
    QA_DRAFTS_DIR,
    METADATA_DIR,
    PREPARED_DIR,
    PER_ITEM_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]
