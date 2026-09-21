"""Seed management and the experiment logging schema (plan section 28)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results" / "raw"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def peak_rss_mb() -> float:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        # Windows exposes a true peak working set; elsewhere fall back to current RSS.
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except ImportError:
        return 0.0


@dataclass
class RunRecord:
    model: str
    condition: str
    dataset: str
    seed: int
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    git_commit: str = field(default_factory=git_commit)
    context_length: int = 0
    parameters: int = 0
    training_tokens: int = 0
    note_tokens: float = 0.0
    note_count: float = 0.0
    fallback_rate: float = 0.0
    accuracy: float = 0.0
    task_success: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_host_memory_mb: float = 0.0
    kv_cache_bytes: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    note_compile_ms: float = 0.0
    total_task_ms: float = 0.0
    train_wall_s: float = 0.0
    estimated_flops: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)

    def save(self, experiment: str) -> Path:
        out = RAW_DIR / experiment
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.condition}_seed{self.seed}_{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def load_runs(experiment: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((RAW_DIR / experiment).glob("*.json"))]


def pick_device(requested: str = "auto") -> torch.device:
    """"auto" uses CUDA when present. Everything here runs on CPU or GPU unchanged."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def gpu_memory_mb() -> float:
    """Peak CUDA memory for this process, 0 on CPU (plan Rule 6 wants the real number)."""
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
