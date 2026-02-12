"""Logging helpers for paper execution."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile


def log_line(msg: str, path: str | Path = "logs/paper_trades.log") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(msg)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def write_state(state: dict, path: str | Path = "logs/paper_state.json") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent) as handle:
        json.dump(state, handle, sort_keys=True)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(target)
