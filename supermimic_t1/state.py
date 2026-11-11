# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""State layout shared by the simulator wrapper, replay buffer, world model and MDP heads.

A state ``S`` is a flat tuple of tensors made of *blocks*. The robot uses two blocks:

  RootBlock   (p, R, v, w)  root position, rotation matrix, world linear/angular velocity of the
                            root link origin
  JointBlock  (q, qd)       joint positions and velocities

so ``S = (p, R, v, w, q, qd)``. Each block knows how to (un)flatten itself, which heading- and
position-invariant features it contributes to the world-model input, how its one-step increment is
expressed in a local frame (the regression target), how to apply a predicted increment, and its
per-component error floors. Everything downstream iterates over ``StateLayout.blocks``, so the trainer
does not depend on the state dimensions.
"""

from __future__ import annotations

import torch

from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat


class RootBlock:
    n_tensors = 4
    flat_dim = 3 + 4 + 3 + 3
    feat_dim = 1 + 6 + 3 + 3
    inc_dim = 3 + 3 + 3 + 3
    names = ("pos", "rot", "linvel", "angvel")

    def flatten(self, t):
        p, R, v, w = t
        return torch.cat((p, mat_to_quat(R), v, w), -1)

    def unflatten(self, x):
        return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13]

    def features(self, t, frame):
        """Height, tilt w.r.t. heading (6D), body-frame linear/angular velocity."""
        p, R, v, w = t
        H = frame["H"]
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (p[..., 2:3], mat_to_6d(H.transpose(-1, -2) @ R), (Rt @ v.unsqueeze(-1)).squeeze(-1),
             (Rt @ w.unsqueeze(-1)).squeeze(-1)), -1)  # fmt: skip

    def increment(self, t, t1, frame):
        """Position delta in the heading frame, rotation vector R^T R1, velocity deltas in the body frame."""
        p, R, v, w = t
        p1, R1, v1, w1 = t1
        Ht, Rt = frame["H"].transpose(-1, -2), R.transpose(-1, -2)
        return torch.cat(
            ((Ht @ (p1 - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ R1),
             (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1), (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1)), -1)  # fmt: skip

    def apply(self, t, d, frame):
        p, R, v, w = t
        H = frame["H"]
        dp, dth, dv, dw = d.split((3, 3, 3, 3), -1)
        return (p + (H @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dth), v + (R @ dv.unsqueeze(-1)).squeeze(-1),
                w + (R @ dw.unsqueeze(-1)).squeeze(-1))  # fmt: skip

    def err_floor(self):
        # floors on the error normalisation: pos 3 mm, rot 5e-3 rad, lin vel 2 cm/s, ang vel 5e-2 rad/s
        return torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2)))

    def component_slices(self, start: int):
        return {n: slice(start + 3 * i, start + 3 * i + 3) for i, n in enumerate(self.names)}


class JointBlock:
    n_tensors = 2
    names = ("q", "qd")

    def __init__(self, nj: int):
        self.nj = nj
        self.flat_dim = self.feat_dim = self.inc_dim = 2 * nj

    def flatten(self, t):
        return torch.cat(t, -1)

    def unflatten(self, x):
        return x[..., : self.nj], x[..., self.nj : 2 * self.nj]

    def features(self, t, frame):
        return torch.cat(t, -1)

    def increment(self, t, t1, frame):
        return torch.cat((t1[0] - t[0], t1[1] - t[1]), -1)

    def apply(self, t, d, frame):
        return t[0] + d[..., : self.nj], t[1] + d[..., self.nj :]

    def err_floor(self):
        return torch.cat((torch.full((self.nj,), 3e-3), torch.full((self.nj,), 5e-2)))

    def component_slices(self, start: int):
        return {"q": slice(start, start + self.nj), "qd": slice(start + self.nj, start + 2 * self.nj)}


class StateLayout:
    """Ordered blocks; block 0 must be the robot root (it defines the heading frame)."""

    def __init__(self, blocks):
        self.blocks = list(blocks)
        assert isinstance(self.blocks[0], RootBlock)
        self.flat_dim = sum(b.flat_dim for b in self.blocks)
        self.feat_dim = sum(b.feat_dim for b in self.blocks)
        self.inc_dim = sum(b.inc_dim for b in self.blocks)
        self.inc_splits = tuple(b.inc_dim for b in self.blocks)

    # ---- tuple <-> blocks ----
    def split(self, S):
        out, i = [], 0
        for b in self.blocks:
            out.append(tuple(S[i : i + b.n_tensors]))
            i += b.n_tensors
        return out

    @staticmethod
    def frame(S):
        return {"H": yaw_mat(S[1]), "p": S[0]}

    def flatten(self, S):
        return torch.cat([b.flatten(t) for b, t in zip(self.blocks, self.split(S))], -1)

    def unflatten(self, x):
        out, i = [], 0
        for b in self.blocks:
            out.extend(b.unflatten(x[..., i : i + b.flat_dim]))
            i += b.flat_dim
        return tuple(out)

    def features(self, S):
        fr = self.frame(S)
        return torch.cat([b.features(t, fr) for b, t in zip(self.blocks, self.split(S))], -1)

    def increment(self, S, S1):
        fr = self.frame(S)
        return torch.cat([b.increment(t, t1, fr) for b, t, t1 in zip(self.blocks, self.split(S), self.split(S1))], -1)

    def apply(self, S, d):
        fr = self.frame(S)
        out = []
        for b, t, di in zip(self.blocks, self.split(S), d.split(self.inc_splits, -1)):
            out.extend(b.apply(t, di, fr))
        return tuple(out)

    def error(self, S_hat, S):
        """Frame-local error of S_hat w.r.t. S, same layout as ``increment`` (frame of the true state)."""
        return self.increment(S, S_hat)

    def err_floor(self):
        return torch.cat([b.err_floor() for b in self.blocks])

    def component_slices(self):
        out, i = {}, 0
        for b in self.blocks:
            out.update(b.component_slices(i))
            i += b.inc_dim
        return out

    def stack(self, seq, dim: int):
        """Stack a list of states along ``dim`` component-wise."""
        n = len(seq[0])
        return tuple(torch.stack([s[i] for s in seq], dim) for i in range(n))


def robot_layout(nj: int) -> StateLayout:
    return StateLayout([RootBlock(), JointBlock(nj)])
