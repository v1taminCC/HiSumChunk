# -*- coding: utf-8 -*-
"""Summary faithfulness audit - standalone API client (Tencent Cloud TokenHub, OpenAI-compatible).

No code dependency on utils_simple_api.py of the other experiments; this file is a self-contained implementation for this directory.
API docs: https://cloud.tencent.com/document/product/1823/130058
Endpoint and key are configured in config.py; thinking is explicitly disabled by default.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import requests

from config import API_BASE, API_KEY, ENABLE_THINKING, MODEL_AUDIT

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 300
DEFAULT_RETRY = 5
DEFAULT_RETRY_SLEEP = 5.0
DEFAULT_WORKERS = 3

_lock = threading.Lock()
_recent: list[float] = []


def _rate_limit(max_rps: int) -> None:
    global _recent
    with _lock:
        now = time.time()
        _recent = [t for t in _recent if now - t < 1.0]
        if len(_recent) >= max_rps:
            sleep_time = 1.0 - (now - _recent[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        _recent.append(time.time())


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = MODEL_AUDIT,
    base_url: str = API_BASE,
    api_key: str = API_KEY,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retry: int = DEFAULT_RETRY,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    max_rps: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    url = f"{base_url}/chat/completions"
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
        # TokenHub deep-reasoning toggle (disabled to keep JSON output and cost under control)
        "thinking": {"type": "enabled" if ENABLE_THINKING else "disabled"},
    }

    last_error = ""
    for attempt in range(retry):
        try:
            _rate_limit(max_rps)
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                body = resp.text[:800]
                last_error = f"HTTP {resp.status_code}: {body}"
                # Fatal errors such as account arrears / exhausted free quota / auth failure: fail immediately without retry
                fatal_markers = ("Arrearage", "overdue-payment", "invalid_api_key",
                                 "not enough quota", "invalid api key", "not_authorized",
                                 "unauthorized", "authentication",
                                 "free trial quota", "postpaid billing", "401008")
                if any(m.lower() in body.lower() for m in fatal_markers):
                    return {"ok": False, "content": "", "usage": {}, "error": last_error,
                            "fatal": True}
                if "429" in body:
                    retry_sleep = max(retry_sleep, 3.0)
                if attempt < retry - 1:
                    time.sleep(retry_sleep * (attempt + 1))
                continue
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {"ok": True, "content": content, "usage": usage, "error": ""}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retry - 1:
                time.sleep(retry_sleep * (attempt + 1))

    return {"ok": False, "content": "", "usage": {}, "error": last_error}


def chat_json(
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> dict[str, Any]:
    result = chat(system_prompt, user_prompt, **kwargs)
    fatal = bool(result.get("fatal"))
    if not result["ok"]:
        return {"ok": False, "parsed": {}, "content": result["content"], "usage": {},
                "error": result["error"], "fatal": fatal}
    content = result["content"]
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        parsed = json.loads(cleaned)
        return {"ok": True, "parsed": parsed, "content": content, "usage": result.get("usage", {}), "error": ""}
    except Exception:  # noqa: BLE001
        return {"ok": False, "parsed": {}, "content": content, "usage": result.get("usage", {}),
                "error": f"JSON parse failed: {content[:200]}", "fatal": False}


def chat_batch(
    calls: list[dict],
    *,
    model: str = MODEL_AUDIT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    workers: int = DEFAULT_WORKERS,
    on_result: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """calls: [{"system":..., "user":..., "meta":...}, ...] -> results aligned by index."""
    results: list[dict[str, Any]] = [{} for _ in calls]

    def _call_one(idx: int, c: dict) -> None:
        result = chat_json(
            c["system"],
            c["user"],
            model=c.get("model", model),
            base_url=c.get("base_url", API_BASE),
            api_key=c.get("api_key", API_KEY),
            temperature=temperature,
            max_tokens=max_tokens,
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
            except Exception:  # noqa: BLE001
                idx = futures[future]
                results[idx] = {"ok": False, "parsed": {}, "content": "", "usage": {}, "error": "thread exception"}

    return results
