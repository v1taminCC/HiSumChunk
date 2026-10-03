# -*- coding: utf-8 -*-
"""Dual-endpoint API client (generation=Qwen/DashScope; judging=DeepSeek official)
- Record the actual model name and token usage from responses (for record-keeping and cost auditing)
- Retry each request `retry` times with exponential backoff; longer wait on 429
"""
from __future__ import annotations

import os

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests

# Generation (Qwen, DashScope)
API_BASE_QWEN = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_QWEN = os.environ.get("DASHSCOPE_API_KEY", "")

# Judging (DeepSeek via Bailian platform; off-peak CNY 4.5/13.5 per M tokens)
API_BASE_DEEPSEEK = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_DEEPSEEK = os.environ.get("DASHSCOPE_API_KEY", "")

MODEL_GENERATE = "qwen3.6-plus"
MODEL_JUDGE = "deepseek-v4-pro"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 120
DEFAULT_RETRY = 5
DEFAULT_RETRY_SLEEP = 5.0
DEFAULT_WORKERS = 3
MIN_REQUEST_INTERVAL = 0.5

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


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = MODEL_GENERATE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retry: int = DEFAULT_RETRY,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    max_rps: int = DEFAULT_WORKERS,
    api_base: str = API_BASE_QWEN,
    api_key: str = API_KEY_QWEN,
    extra_payload: dict | None = None,
) -> dict[str, Any]:
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_payload:
        payload.update(extra_payload)

    last_error = ""
    for attempt in range(retry):
        try:
            _rate_limit(max_rps)
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "ok": True, "content": content, "error": "",
                "model_actual": data.get("model", ""),
                "usage": data.get("usage", {}),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if "429" in last_error:
                retry_sleep = max(retry_sleep, 3.0)
            if attempt < retry - 1:
                time.sleep(retry_sleep * (attempt + 1))

    return {"ok": False, "content": "", "error": last_error, "model_actual": "", "usage": {}}


def chat_batch(
    calls: list[dict],
    *,
    model: str = MODEL_GENERATE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    workers: int = DEFAULT_WORKERS,
    api_base: str = API_BASE_QWEN,
    api_key: str = API_KEY_QWEN,
    extra_payload: dict | None = None,
    on_result: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = [{}] * len(calls)

    def _call_one(idx: int, c: dict) -> None:
        result = chat(
            system_prompt=c["system"],
            user_prompt=c["user"],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_rps=workers,
            api_base=api_base,
            api_key=api_key,
            extra_payload=extra_payload,
        )
        results[idx] = result
        if on_result:
            on_result(idx, result)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_call_one, i, c): i for i, c in enumerate(calls)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:  # noqa: BLE001
                idx = futures[future]
                results[idx] = {"ok": False, "content": "", "error": "thread exception",
                                "model_actual": "", "usage": {}}

    return results
