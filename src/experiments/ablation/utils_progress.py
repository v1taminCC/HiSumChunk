# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import sys
import threading
import time


class ProgressTracker:
    def __init__(self, total_books: int):
        self.total_books = total_books
        self.book_start_time = time.time()
        self.book_idx = 0
        self.book_title = ""
        self.scheme = ""
        self.scheme_start_time = time.time()
        self.query_total = 0
        self.query_done = 0
        self.ok_count = 0
        self.fail_count = 0
        self.book_ok = 0
        self.book_fail = 0
        self._lock = threading.Lock()
        self._last_print = 0.0

    def _elapsed_str(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def start_book(self, book_idx: int, book_title: str) -> None:
        with self._lock:
            self.book_idx = book_idx
            self.book_title = book_title
            self.book_start_time = time.time()
            self.book_ok = 0
            self.book_fail = 0
        bar = self._total_bar()
        label = f"\u25b8 \u4e66\u672c [{book_idx}/{self.total_books}]"
        print(f"\n{'─'*_term_width()}", flush=True)
        print(f"{bar} {label} {book_title}", flush=True)

    def start_scheme(self, scheme: str, query_total: int, skipped: int = 0) -> None:
        with self._lock:
            self.scheme = scheme
            self.query_total = query_total
            self.query_done = skipped
            self.ok_count = skipped
            self.fail_count = 0
            self.scheme_start_time = time.time()
            self._last_print = 0.0
            self.book_ok += skipped
        if skipped > 0 and skipped < query_total:
            print(f"    [{scheme}] {skipped}/{query_total} already done, {query_total - skipped} remaining", flush=True)
        self._render()

    def update(self, ok: bool = True) -> None:
        with self._lock:
            self.query_done += 1
            if ok:
                self.ok_count += 1
                self.book_ok += 1
            else:
                self.fail_count += 1
                self.book_fail += 1
        now = time.time()
        if now - self._last_print > 0.3:
            self._render()

    def finish_scheme(self) -> None:
        self._render(force=True)
        print(flush=True)

    def finish_book(self) -> None:
        elapsed = time.time() - self.book_start_time
        elapsed_str = self._elapsed_str(elapsed)
        bar = self._total_bar()
        label = f"\u25b8 \u4e66\u672c [{self.book_idx}/{self.total_books}]"
        status = f"\u2713{self.book_ok} \u2717{self.book_fail}"
        print(f"{bar} {label} {self.book_title}  {status}  \u8017\u65f6 {elapsed_str}", flush=True)

    def _total_bar(self) -> str:
        bar_width = 12
        filled = int(bar_width * self.book_idx / max(self.total_books, 1))
        return "\u2588" * filled + "\u2591" * (bar_width - filled)

    def _render(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_print < 0.3:
            return
        self._last_print = now

        with self._lock:
            total = self.query_total
            done = self.query_done
            ok = self.ok_count
            fail = self.fail_count
            scheme = self.scheme
            start = self.scheme_start_time

        pct = done / max(total, 1) * 100
        elapsed = now - start
        if done > 0 and done < total:
            eta = elapsed / done * (total - done)
            eta_str = self._elapsed_str(eta)
        elif done >= total:
            eta_str = self._elapsed_str(now - start) + " done"
        else:
            eta_str = "..."

        bar_width = 20
        filled = int(bar_width * done / max(total, 1))
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        scheme_label = f"[{scheme}]"
        status = f"\u2713{ok} \u2717{fail}"

        line = (
            f"  {scheme_label:<8} {bar} {done:>4}/{total:<4} ({pct:5.1f}%)"
            f"  {status:<12}  \u5269\u4f59 {eta_str}"
        )
        print(f"\r{line:<120}", end="", flush=True)
        sys.stdout.flush()

    def total_summary(self, total_books: int, total_queries: int, start_time: float) -> str:
        elapsed = time.time() - start_time
        return f"\n{'═'*_term_width()}\n\u5168\u90e8\u5b8c\u6210\uff01{total_books} \u672c\u4e66, {total_queries} \u6761\u67e5\u8be2, \u603b\u8017\u65f6 {self._elapsed_str(elapsed)}\n{'═'*_term_width()}"


def _term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 120
