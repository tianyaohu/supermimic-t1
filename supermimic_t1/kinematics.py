# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Differentiable batched forward kinematics of the T1 in torch, built from the T1 MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for simulator
states and for world-model predictions, so the analytic reward head matches the reward the simulator
transitions are labelled with. Gate 1 (tools/test_fk.py) checks it against Newton's body states.

Velocities are those of each link-frame origin (Isaac Lab's ``body_link_vel_w``). The root is the
free ``trunk`` body; every other moving body has one hinge whose anchor is the body origin (asserted).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import MJCF_PATH, T1_JOINT_NAMES, TRACKED_BODIES
from .rotations import axis_angle_to_mat, quat_to_mat


class Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES, joint_names: list[str] = T1_JOINT_NAMES,
                 mjcf: str = MJCF_PATH):  # fmt: skip
        m = mujoco.MjModel.from_xml_path(mjcf)
        self.device = device
        self.joint_names = list(joint_names)
        self.num_joints = len(joint_names)
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed, massless marker bodies (camera frame); none are tracked
            assert m.body_jntnum[b] == 1, name
            jid = m.body_jntadr[b]
            assert np.allclose(m.jnt_pos[jid], 0.0), f"joint anchor of {name} is not the body origin"
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, name
                pb = m.body_parentid[b]
                if pb not in mj_to_local:
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        assert sorted(i for i in joint_idx if i >= 0) == list(range(self.num_joints)), "joint set mismatch"
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked = list(tracked)
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        lo, hi = np.zeros(self.num_joints), np.zeros(self.num_joints)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions (B,nb,3), rotations (B,nb,3,3), link-origin linear and angular velocities.

        Args:
            p, R, v, w: root (trunk) position (B,3), rotation (B,3,3), world lin/ang velocity of the link origin.
            q, qd: joint positions/velocities (B,23) in T1_JOINT_NAMES order.
        """
        n = self.num_bodies
        P, Rs, V, W = [None] * n, [None] * n, [None] * n, [None] * n
        for i in range(n):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            r = (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            pi = P[par] + r
            R_off = Rp @ self.off_R[i]
            Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
            axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], r, dim=-1)  # hinge at the child origin: no joint-induced term
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(n) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
