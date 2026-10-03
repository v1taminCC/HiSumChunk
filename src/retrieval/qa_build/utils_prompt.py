from __future__ import annotations

from pathlib import Path
from typing import Any

from utils_io import read_text


def load_prompt(name: str, prompt_dir: Path) -> str:
    return read_text(prompt_dir / name)


def render_prompt(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)
