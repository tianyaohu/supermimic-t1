# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Reference motion library: Booster CSV clips (root pos, root quat xyzw, joints) at 50 Hz, with FK.

CSV layout (booster_assets README): x y z, qx qy qz qw, then the joints in T1_JOINT_NAMES order.
Clips at another rate are resampled to 50 Hz (lerp/slerp, as booster_train's csv_to_npz.py).
Velocities are central finite differences (one-sided at the ends), as torch.gradient in csv_to_npz.py.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .kinematics import Kinematics
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


def goal_dim_per_frame(num_joints: int) -> int:
    return 3 + 6 + 3 + 3 + 2 * num_joints


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, device: str, src_fps: float = 50.0):
        self.device = device
        self.nj = kin.num_joints
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
            assert raw.shape[1] == 7 + self.nj, f"{f}: {raw.shape[1]} columns, expected {7 + self.nj}"
            p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:]
            quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)  # -> wxyz
            T_src = raw.shape[0]
            if abs(src_fps - 1 / CONTROL_DT) > 1e-6:
                dur = (T_src - 1) / src_fps
                n = int(np.floor(dur / CONTROL_DT + 1e-9)) + 1
                t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
                i0 = torch.clamp(t.floor().long(), max=T_src - 2)
                a = (t - i0.float()).clamp(0, 1)
                p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
                j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
                quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j), lens.append(p.shape[0])
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.tensor([0] + lens[:-1], dtype=torch.long), 0).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            w = torch.zeros_like(p)  # world angular velocity by central differences of rotations
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            out = [[], [], [], []]
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                for k, x in enumerate(kin.forward(*(t[s:e] for t in self.ref_state_all()))):
                    out[k].append(x)
            self.body_p, self.body_R, self.body_v, self.body_w = (torch.cat(x) for x in out)
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            for b in range(int(np.ceil(lens[c] / BIN_FRAMES))):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[bin_clip[b]]) + 1)
            self.frame_bin[s:e] = b
        self.goal_frame_dim = goal_dim_per_frame(self.nj)

    def ref_state_all(self):
        return self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """K future reference frames, holding the final pose with zero velocity beyond the clip end."""
        fut = self.future_frames(f, K)
        requested = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        recorded = (requested <= self.clip_end[self.clip_id[f]].unsqueeze(-1)).unsqueeze(-1)
        v = torch.where(recorded, self.root_v_local[fut], 0.0)
        w = torch.where(recorded, self.root_w_local[fut], 0.0)
        qd = torch.where(recorded, self.qd[fut], 0.0)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat((dp, rot, v, w, self.q[fut], qd * 0.1), -1)  # (B, K, 15 + 2 nj)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
