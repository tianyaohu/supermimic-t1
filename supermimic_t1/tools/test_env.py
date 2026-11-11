"""Simulator sanity checks: reset readback without refresh, reference playback, step timing.

Run: python supermimic_t1/tools/test_env.py --num_envs 2048
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
add_launcher_args(parser)
args = parser.parse_args()

import torch

from supermimic_t1.env import TrackingSim, make_sim_cfg
from supermimic_t1.kinematics import Kinematics
from supermimic_t1.mdp import MAX_STEP_REWARD, MDP
from supermimic_t1.motion import MotionLib
from supermimic_t1.robot import MOTION_CSV, MOTION_FPS, default_joint_pos_list
from supermimic_t1.state import robot_layout

dev = "cuda:0"
kin = Kinematics(dev)
motion = MotionLib([MOTION_CSV], kin, dev, src_fps=MOTION_FPS)
print(f"motion: {motion.num_frames} frames ({motion.num_frames * 0.02:.1f} s), bins {motion.num_bins}")
mdp = MDP(kin, motion, robot_layout(kin.num_joints), torch.tensor(default_joint_pos_list(), device=dev))
q_out = ((motion.q < kin.q_lo) | (motion.q > kin.q_hi)).float().mean().item()
print(f"reference joints outside hard limits: {100 * q_out:.2f}% of (frame, joint) entries")
sim_cfg = make_sim_cfg(dev)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True)
    N = args.num_envs
    ids = torch.arange(N, device=dev)
    # 1) reset readback without refresh, after the env has been stepped (stale-buffer check)
    for _ in range(3):
        env.step(torch.zeros(N, kin.num_joints, device=dev))
    frames = torch.randint(0, motion.num_frames - 600, (N,), device=dev)
    env.reset(ids, frames)
    S = env.state()
    ref = motion.ref_state(frames)
    for n, a, b in zip(("p", "R", "v", "w", "q", "qd"), S, ref):
        print(f"reset readback (no refresh) {n}: max err {(a - b).abs().max().item():.2e}")
    # 2) reference playback with joint setpoints = reference joints (no policy)
    env.reset(ids, frames)
    rs, t0 = [], None
    for t in range(150):
        if t == 20:
            torch.cuda.synchronize()
            t0 = time.time()
        a = (motion.q[env.f + 1] - env.default_q) / env.alpha
        out = env.step(a)
        rs.append(out["r"].mean().item() / MAX_STEP_REWARD)
        if t % 25 == 0:
            print(f"t {t}: norm reward {rs[-1]:.3f} | failed frac {env.failed_eval.float().mean().item():.3f} | "
                  f"e_body_pos {out['terms']['e_body_pos'].mean().item():.4f}", flush=True)  # fmt: skip
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 130
    print(f"step time {1e3 * dt:.2f} ms per 50 Hz step at {N} envs ({N / dt:.0f} env-steps/s)")
    print(f"peak torch VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
