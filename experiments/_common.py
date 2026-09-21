"""Shared bootstrap for experiment scripts: import path and config loading."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())
