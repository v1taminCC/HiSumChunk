# -*- coding: utf-8 -*-
"""SGLang local-model client (same framework as the proposed method).

- RuntimeEndpoint connects to the local SGLang server (Qwen3-4B-Instruct-2507)
- @function + gen for one-shot dialogue; ThreadPoolExecutor for batched concurrency
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CR_MAX_NEW_TOKENS, CR_TEMPERATURE, CR_TOP_K, CR_TOP_P, SGLANG_ENDPOINT,
)

from sglang import (  # noqa: E402
    RuntimeEndpoint,
    assistant,
    function,
    gen,
    set_default_backend,
    system,
    user,
)

_backend_initialized = False


def init_backend() -> None:
    global _backend_initialized
    if _backend_initialized:
        return
    set_default_backend(RuntimeEndpoint(SGLANG_ENDPOINT))
    _backend_initialized = True


@function
def _single_chat(s, prompt: str, max_tokens: int):
    s += system("")
    s += user(prompt)
    s += assistant(gen(
        "response",
        max_tokens=max_tokens,
        temperature=CR_TEMPERATURE,
        top_p=CR_TOP_P,
        top_k=CR_TOP_K,
    ))


def _extract(result) -> str:
    if isinstance(result, dict):
        raw = result.get("response", "")
    elif hasattr(result, "response"):
        raw = result.response
    elif isinstance(result, str):
        raw = result
    else:
        try:
            raw = result["response"]
        except Exception:  # noqa: BLE001
            raw = str(result)
    if not isinstance(raw, str):
        raw = str(raw)
    return raw.strip()


def call_model(prompt: str, max_tokens: int | None = None) -> str:
    """Single call; failures raise and are caught upstream"""
    init_backend()
    result = _single_chat(prompt=prompt, max_tokens=max_tokens or CR_MAX_NEW_TOKENS)
    return _extract(result)


def chat_batch(prompts: list[str], *, workers: int = 16,
               max_tokens: int | None = None,
               on_result: Callable[[int, str], None] | None = None) -> list[str]:
    """Batches concurrent calls by `workers`; returns a list of texts as long as prompts (failure = empty string)"""
    results = [''] * len(prompts)
    for start in range(0, len(prompts), workers):
        batch = list(range(start, min(start + workers, len(prompts))))
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = {
                ex.submit(call_model, prompts[j], max_tokens): j for j in batch
            }
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    results[j] = fut.result()
                except Exception:  # noqa: BLE001
                    results[j] = ''
                if on_result:
                    on_result(j, results[j])
    return results
