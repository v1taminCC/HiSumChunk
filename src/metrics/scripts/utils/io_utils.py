from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from config import DIRECTORIES_TO_CREATE, LOGS_DIR, WORK_DIR


def ensure_directories() -> None:
    for directory in DIRECTORIES_TO_CREATE:
        safe_mkdir(directory)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_inside_workdir(path: Path) -> Path:
    resolved = path.resolve()
    if not is_relative_to(resolved, WORK_DIR.resolve()):
        raise ValueError(f"Refuse to write outside WORK_DIR: {resolved}")
    return resolved


def safe_mkdir(path: Path) -> None:
    assert_inside_workdir(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_write_json(data: Any, path: Path) -> None:
    assert_inside_workdir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_write_text(text: str, path: Path) -> None:
    assert_inside_workdir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    assert_inside_workdir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        field_set: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in field_set:
                    field_set.append(key)
        fieldnames = field_set
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy_file(source: Path, target: Path) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    assert_inside_workdir(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source": str(source),
        "target": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences(text: str, min_chars: int = 1) -> list[str]:
    normalized = normalize_space(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])", normalized)
    sentences = [part.strip() for part in parts if len(part.strip()) >= min_chars]
    return sentences or [normalized]


def get_context_text(context_chunks: Any, truncation_chars: int | None = None) -> str:
    if not isinstance(context_chunks, list):
        return ""
    texts: list[str] = []
    for chunk in context_chunks:
        if isinstance(chunk, dict):
            text = normalize_space(chunk.get("text", ""))
        else:
            text = normalize_space(chunk)
        if text:
            texts.append(text)
    combined = "\n".join(texts)
    if truncation_chars and len(combined) > truncation_chars:
        return combined[:truncation_chars]
    return combined


def record_key(record: dict[str, Any]) -> str:
    book_title = normalize_space(record.get("book_title"))
    scheme = normalize_space(record.get("scheme"))
    query_id = normalize_space(record.get("query_id"))
    return f"{book_title}||{scheme}||{query_id}"


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


ACTIVE_LOG_FILE: Path | None = None


def set_log_file(script_file: str) -> Path:
    global ACTIVE_LOG_FILE
    script_name = Path(script_file).stem
    log_path = LOGS_DIR / f"{script_name}_log.txt"
    assert_inside_workdir(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_LOG_FILE = log_path
    separator = "=" * 80
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n")
        f.write(f"Log session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: {script_file}\n")
        f.write(f"Log file: {log_path}\n")
        f.write(f"{separator}\n\n")
    return log_path


def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"\n[{timestamp}] {message}\n"
    print(line, flush=True)
    if ACTIVE_LOG_FILE is not None:
        with ACTIVE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


class ProgressLogger:
    def __init__(self, task_name: str, total: int, every: int = 1) -> None:
        self.task_name = task_name
        self.total = max(int(total), 0)
        self.every = max(int(every), 1)
        self.start_time = time.time()
        self.last_count = 0
        log_message(f"START {self.task_name} | total={self.total}")

    def update(self, current: int, detail: str = "", force: bool = False) -> None:
        current = min(max(int(current), 0), self.total if self.total else int(current))
        if not force and current != self.total and current - self.last_count < self.every:
            return
        self.last_count = current
        elapsed = time.time() - self.start_time
        if current > 0 and self.total > 0:
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (self.total - current) / rate if rate > 0 else None
            percent = current / self.total * 100
        else:
            rate = 0
            remaining = None
            percent = 0
        detail_text = f" | {detail}" if detail else ""
        log_message(
            f"PROGRESS {self.task_name} | {current}/{self.total} ({percent:.2f}%)"
            f" | elapsed={format_duration(elapsed)} | eta={format_duration(remaining)}"
            f" | speed={rate:.2f}/s{detail_text}"
        )

    def finish(self, detail: str = "") -> None:
        elapsed = time.time() - self.start_time
        detail_text = f" | {detail}" if detail else ""
        log_message(f"DONE {self.task_name} | total={self.total} | elapsed={format_duration(elapsed)}{detail_text}")
