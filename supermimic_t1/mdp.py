# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State ``S = (p, R, v, w, q, qd)`` of the T1 root (trunk = BeyondMimic-T1 anchor) and joints; see state.py.
Sim context xi = [ground friction (1), joint default offsets (nj), torso CoM offset (3)].
All dimensions derive from the joint count nj and the tracked-body count nb.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, rot_angle_sq, yaw_mat
from .state import StateLayout

MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT  # every positive term at 1 -> 0.1


def xi_dim(nj: int) -> int:
    return 1 + nj + 3


def obs_dim(nj: int) -> int:
    return nj + nj + 3 + 3 + nj + 6


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0  # paper App. A (BeyondMimic-T1 uses -10)
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, layout: StateLayout, default_q: torch.Tensor,
                 cfg: RewardCfg = RewardCfg()):  # fmt: skip
        self.kin, self.motion, self.layout, self.cfg = kin, motion, layout, cfg
        nj = self.nj = kin.num_joints
        self.nb = len(kin.tracked_idx)
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        # Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation (+-0.05)
        self.obs_noise = torch.cat((torch.full((nj,), 0.01), torch.full((nj,), 0.5), torch.full((3,), 0.2),
                                    torch.full((3,), 0.05), torch.zeros(nj), torch.full((6,), 0.05))).to(dev)  # fmt: skip
        self.obs_dim = obs_dim(nj)
        self.xi_dim = xi_dim(nj)
        self.critic_extra = nj + self.nb * 3 + 3 + 6 + self.xi_dim

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioception (+ reference root orientation relative to the robot root)."""
        p, R, v, w, q, qd = S[:6]
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1 : 1 + self.nj]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(Rt @ self.motion.root_R[f])
        return torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def _ref_bodies(self, S, f):
        """Reference tracked bodies re-anchored to the robot root (xy + yaw), as in BeyondMimic."""
        m = self.motion
        p, R = S[0], S[1]
        rp, rR = m.root_p[f], m.root_R[f]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        return ref_bp, dR

    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S[:6]
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        ref_bp, dR = self._ref_bodies(S, f)
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if terms:
            return r, {"track": r_track * CONTROL_DT, "e_body_pos": e_pos, "e_body_ori": e_ori, "e_root_pos": e_rp,
                       "action_rate": r_rate, "joint_limit": r_lim}  # fmt: skip
        return r

    def _errs(self, S, f):
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        return (rp[..., 2] - p[..., 2]).abs(), (rR[..., 2, 2] - R[..., 2, 2]).abs()

    def termination(self, S, f) -> torch.Tensor:
        """|e_z| > 0.25 m or e_tilt = |g_z,ref - g_z| > 0.8 (BeyondMimic anchor terms, anchor = trunk)."""
        e_z, e_tilt = self._errs(S, f)
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        e_z, e_tilt = self._errs(S, f)
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi_n) -> torch.Tensor:
        """Critic input: invariant state, robot-vs-reference errors, (normalised) sim context."""
        m = self.motion
        p, R, v, w, q, qd = S[:6]
        Ht = yaw_mat(R).transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        ref_bp, _ = self._ref_bodies(S, f)
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((self.layout.features(S), m.q[f] - q, e_body, e_root, rel, xi_n), -1)

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        nj = self.nj
        return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1 : 1 + nj] / 0.01, xi[..., 1 + nj : 4 + nj] / 0.05), -1)
