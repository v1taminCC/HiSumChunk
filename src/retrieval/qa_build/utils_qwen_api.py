# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests

API_BASE = os.getenv("API_BASE_URL", "")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3.6-plus"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.8
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120
DEFAULT_RETRY = 5
DEFAULT_RETRY_SLEEP = 3.0
DEFAULT_WORKERS = 5

_request_timestamps: list[float] = []
_ratelimit_lock = threading.Lock()


def _rate_limit(max_rps: int = DEFAULT_WORKERS) -> None:
    global _request_timestamps
    with _ratelimit_lock:
        now = time.time()
        _request_timestamps = [t for t in _request_timestamps if now - t < 1.0]
        if len(_request_timestamps) >= max_rps:
            sleep_time = 1.0 - (now - _request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        _request_timestamps.append(time.time())


def qwen_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retry: int = DEFAULT_RETRY,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    max_rps: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if not API_KEY:
        return {"ok": False, "raw_response": None, "content": "", "parsed": None, "error": "DASHSCOPE_API_KEY is not set"}
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    last_error = ""
    for attempt in range(retry):
        try:
            _rate_limit(max_rps)
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return {
                "ok": True,
                "raw_response": data,
                "content": content,
                "parsed": parsed,
                "error": "",
            }
        except Exception as exc:
            last_error = str(exc)
            if "429" in last_error:
                retry_sleep = max(retry_sleep, 3.0)
            if attempt < retry - 1:
                time.sleep(retry_sleep * (attempt + 1))

    return {
        "ok": False,
        "raw_response": {},
        "content": "",
        "parsed": {},
        "error": last_error or "unknown_error",
    }


def qwen_chat_batch(
    calls: list[dict],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retry: int = DEFAULT_RETRY,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    workers: int = DEFAULT_WORKERS,
    on_result: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = [{}] * len(calls)

    def _call_one(idx: int, c: dict) -> None:
        result = qwen_chat(
            system_prompt=c["system"],
            user_prompt=c["user"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            retry=retry,
            retry_sleep=retry_sleep,
            max_rps=workers,
        )
        results[idx] = result
        if on_result:
            on_result(idx, result)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_call_one, i, c): i for i, c in enumerate(calls)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                idx = futures[future]
                results[idx] = {"ok": False, "parsed": {}, "content": "", "error": "thread exception"}

    return results
