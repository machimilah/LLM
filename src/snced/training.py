"""Training utilities for free-tier GPUs (Kaggle, Colab).

Sessions there are capped at a few hours and can be killed without warning, so
every long run must checkpoint and resume. `Trainer state` keeps everything
needed to continue exactly where a killed session stopped: weights, optimizer,
step, curriculum stage and RNG streams.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainState:
    """Everything needed to continue a run in a fresh session."""

    step: int = 0
    stage: int = 0
    final_start: int | None = None
    best_metric: float = -1.0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"step": self.step, "stage": self.stage, "final_start": self.final_start,
                "best_metric": self.best_metric, "history": self.history}


def save_resumable(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   state: TrainState, rng: random.Random) -> None:
    """Atomic save: a session killed mid-write must not corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.to_dict(),
        "rng": {"python": rng.getstate(), "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_resumable(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                   rng: random.Random | None = None) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if rng is not None and payload.get("rng"):
        rng.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch"].cpu() if torch.is_tensor(payload["rng"]["torch"])
                            else payload["rng"]["torch"])
        np.random.set_state(payload["rng"]["numpy"])
    s = payload["state"]
    return TrainState(step=s["step"], stage=s["stage"], final_start=s["final_start"],
                      best_metric=s["best_metric"], history=s.get("history", []))


class SessionBudget:
    """Stop cleanly before the platform kills the session.

    Kaggle gives ~9 hours and Colab less; a run that ignores the limit loses
    whatever it did not checkpoint.
    """

    def __init__(self, hours: float = 8.5) -> None:
        self.deadline = time.time() + hours * 3600

    @property
    def expired(self) -> bool:
        return time.time() >= self.deadline

    @property
    def remaining_min(self) -> float:
        return max(0.0, (self.deadline - time.time()) / 60)


def load_quantized_causal_lm(model_id: str, bits: int = 4, device_map: str = "auto"):
    """Load a large model in 4-bit so a 7B fits on a 16 GB T4 for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map=device_map)
    return model, tok


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")):
    """Trainable adapters over a frozen (optionally quantised) base model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM", target_modules=list(targets))
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable of {total:,} ({trainable / total:.2%})")
    return model


def write_manifest(path: Path, **fields) -> None:
    """A small JSON beside the checkpoint so a later session knows what it holds."""
    path.write_text(json.dumps(fields, indent=2), encoding="utf-8")
