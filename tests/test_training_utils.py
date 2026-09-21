import random
from pathlib import Path

import torch

from snced.training import SessionBudget, TrainState, load_resumable, save_resumable


def test_checkpoint_round_trip_restores_everything(tmp_path: Path):
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    rng = random.Random(7)
    [rng.random() for _ in range(5)]
    state = TrainState(step=1234, stage=3, final_start=900, best_metric=0.87)

    path = tmp_path / "ckpt.pt"
    save_resumable(path, model, opt, state, rng)

    fresh, fresh_opt, fresh_rng = torch.nn.Linear(4, 3), None, random.Random(0)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    restored = load_resumable(path, fresh, fresh_opt, fresh_rng)

    assert (restored.step, restored.stage, restored.final_start) == (1234, 3, 900)
    assert restored.best_metric == 0.87
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)
    assert fresh_rng.random() == rng.random()      # the data stream continues identically


def test_save_is_atomic(tmp_path: Path):
    """A session killed mid-write must not leave a corrupt checkpoint."""
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.pt"
    save_resumable(path, model, opt, TrainState(), random.Random(0))
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()


def test_session_budget_expires():
    assert SessionBudget(hours=0).expired
    budget = SessionBudget(hours=1)
    assert not budget.expired and 59 < budget.remaining_min <= 60
