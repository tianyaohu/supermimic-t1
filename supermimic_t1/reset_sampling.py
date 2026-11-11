# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Return-based reset sampling helpers."""

import torch


def update_bin_returns(bin_G: torch.Tensor, bins: torch.Tensor, returns: torch.Tensor, old_weight: float = 0.8):
    sums = torch.zeros_like(bin_G)
    counts = torch.zeros_like(bin_G)
    sums.index_add_(0, bins, returns)
    counts.index_add_(0, bins, torch.ones_like(returns))
    updated = counts > 0
    bin_G[updated] = old_weight * bin_G[updated] + (1 - old_weight) * sums[updated] / counts[updated]
    return bin_G
