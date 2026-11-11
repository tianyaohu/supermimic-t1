# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Fixed-size GPU FIFO replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, state_dim: int, act_dim: int, xi_dim: int):
        self.T, self.N, self.device = T, N, device
        self.fields = {
            "s": (state_dim, torch.float32), "s1": (state_dim, torch.float32), "a": (act_dim, torch.float32),
            "a_prev": (act_dim, torch.float32), "xi": (xi_dim, torch.float32), "r": ((), torch.float32),
            "f": ((), torch.long), "ep_step": ((), torch.int32), "done": ((), torch.bool),
        }  # fmt: skip
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.data.values())

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int, newest: bool = False):
        """k-step windows; dict of (B, k, ...) tensors and a validity mask (B, k).

        ``newest=True`` takes windows that end at the most recently added row (held-out data when called
        right after collection and before the model update)."""
        if k < 1 or k > self.size:
            raise ValueError(f"Cannot sample {k}-step windows from {self.size} buffered steps")
        if newest:
            t0 = torch.full((B,), self.size - k, device=self.device)
        else:
            t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=self.device))
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the rows of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
