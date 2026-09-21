import torch

from snced.models.layers import MoEFeedForward, sliding_window_mask, topk_sparse_mask


def test_moe_activates_only_the_routed_experts():
    torch.manual_seed(0)
    moe = MoEFeedForward(d_model=32, d_ff=64, n_experts=8, top_k=2)
    x = torch.randn(2, 5, 32)
    out = moe(x)
    assert out.shape == x.shape
    assert moe.active_fraction == 0.25          # 2 of 8 experts per token
    out.sum().backward()
    assert moe.gate.weight.grad is not None
    assert torch.isfinite(moe.last_balance_loss)


def test_sliding_window_mask_forgets_old_tokens():
    m = sliding_window_mask(6, window=3, device=torch.device("cpu"))
    assert not m[5, 3] and not m[5, 5]   # inside the window
    assert m[5, 2]                        # older than the window
    assert m[0, 1]                        # still causal


def test_topk_sparse_mask_keeps_only_k_keys_per_query():
    torch.manual_seed(0)
    scores = torch.randn(1, 2, 6, 6)
    mask = topk_sparse_mask(scores, k=2, causal=True)
    kept = (~mask).sum(-1)
    # Each query keeps at most k keys, and never more than its causal prefix.
    assert int(kept.max()) <= 2
    assert int(kept[0, 0, 0]) == 1        # first token can only see itself
