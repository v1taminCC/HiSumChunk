from __future__ import annotations

from datetime import datetime

from config import (
    ANSWER_FILE_PATTERN,
    ANSWERS_DIR,
    INPUT_MANIFEST_PATH,
    JUDGMENT_FILE_PATTERN,
    JUDGMENTS_DIR,
    QA_DRAFTS_DIR,
    SCHEMES,
    SOURCE_ANSWERS_DIR,
    SOURCE_JUDGMENTS_DIR,
    SOURCE_QA_DRAFTS,
)
from utils.io_utils import ProgressLogger, ensure_directories, log_message, safe_copy_file, safe_write_json, set_log_file

OVERWRITE_EXISTING_COPIES = True
PROGRESS_EVERY_FILES = 5


def list_sources() -> tuple[list[tuple[str, str | None, object]], list[tuple[str, str | None, object]], list[tuple[str, str | None, object]]]:
    answer_sources: list[tuple[str, str | None, object]] = []
    judgment_sources: list[tuple[str, str | None, object]] = []
    qa_sources: list[tuple[str, str | None, object]] = []

    for scheme in SCHEMES:
        pattern = ANSWER_FILE_PATTERN.format(scheme=scheme)
        for source in sorted(SOURCE_ANSWERS_DIR.glob(pattern)):
            answer_sources.append(("answers", scheme, source))

    for source in sorted(SOURCE_JUDGMENTS_DIR.glob(JUDGMENT_FILE_PATTERN)):
        judgment_sources.append(("judgments_eval", None, source))

    summary_path = SOURCE_JUDGMENTS_DIR / "evaluation_summary.txt"
    if summary_path.exists():
        judgment_sources.append(("judgments_eval_summary", None, summary_path))

    if SOURCE_QA_DRAFTS.exists():
        qa_sources.append(("qa_drafts", None, SOURCE_QA_DRAFTS))

    return answer_sources, judgment_sources, qa_sources


def target_for(category: str, source) -> object:
    if category == "answers":
        return ANSWERS_DIR / source.name
    if category in {"judgments_eval", "judgments_eval_summary"}:
        return JUDGMENTS_DIR / source.name
    if category == "qa_drafts":
        return QA_DRAFTS_DIR / source.name
    raise ValueError(f"Unsupported category: {category}")


def copy_sources(sources: list[tuple[str, str | None, object]], task_name: str) -> list[dict]:
    copied: list[dict] = []
    progress = ProgressLogger(task_name, total=len(sources), every=PROGRESS_EVERY_FILES)
    for index, (category, scheme, source) in enumerate(sources, start=1):
        target = target_for(category, source)
        if target.exists() and not OVERWRITE_EXISTING_COPIES:
            progress.update(index, detail=f"skipped {source.name}")
            continue
        info = safe_copy_file(source, target)
        info["category"] = category
        if scheme:
            info["scheme"] = scheme
        copied.append(info)
        progress.update(index, detail=f"copied {source.name}")
    progress.finish(detail=f"copied={len(copied)}")
    return copied


def main() -> None:
    log_path = set_log_file(__file__)
    log_message(f"Log file: {log_path}")
    log_message("Step 00 started: copy input files into isolated work directory")
    ensure_directories()
    answer_sources, judgment_sources, qa_sources = list_sources()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_answers_dir": str(SOURCE_ANSWERS_DIR),
        "source_judgments_dir": str(SOURCE_JUDGMENTS_DIR),
        "source_qa_drafts": str(SOURCE_QA_DRAFTS),
        "files": [],
    }
    manifest["files"].extend(copy_sources(answer_sources, "copy answer files"))
    manifest["files"].extend(copy_sources(judgment_sources, "copy judgment files"))
    manifest["files"].extend(copy_sources(qa_sources, "copy qa draft files"))
    safe_write_json(manifest, INPUT_MANIFEST_PATH)
    log_message(f"Copied {len(manifest['files'])} files into isolated work directory")
    log_message(f"Manifest: {INPUT_MANIFEST_PATH}")
    log_message("Step 00 finished")


if __name__ == "__main__":
    main()
