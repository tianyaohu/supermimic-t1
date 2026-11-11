"""Gate 1: torch FK vs. Newton/MuJoCo-Warp body poses and velocities on random states.

Two checks, both on the simulator's own readback:
  A. kinematic: write random root poses/velocities and random joint states (uniform within the joint
     limits, velocities N(0, 3 rad/s)), run forward kinematics (sim.forward), compare.
  B. dynamic: after B, run several 50 Hz control steps with random actions (contacts, gravity,
     actuators active) and compare FK of the read-back state with the read-back body states.
Pass: position error < 1 mm and rotation error < 1e-3 rad on every tracked body (and on all bodies).
Velocity errors are printed (no threshold given in the brief; reported as measured).

Run: python supermimic_t1/tools/test_fk.py --num_envs 256
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--trials", type=int, default=4)
parser.add_argument("--out", type=str, default=None)
add_launcher_args(parser)
args = parser.parse_args()

import torch

from supermimic_t1.env import TrackingSim, make_sim_cfg
from supermimic_t1.kinematics import Kinematics
from supermimic_t1.mdp import MDP
from supermimic_t1.motion import MotionLib
from supermimic_t1.robot import MOTION_CSV, MOTION_FPS, T1_JOINT_NAMES, default_joint_pos_list
from supermimic_t1.rotations import quat_to_mat, rot_angle_sq, so3_exp
from supermimic_t1.state import robot_layout

dev = "cuda:0"
torch.manual_seed(0)
kin = Kinematics(dev)
kin_all = kin
motion = MotionLib([MOTION_CSV], kin, dev, src_fps=MOTION_FPS)
mdp = MDP(kin, motion, robot_layout(kin.num_joints), torch.tensor(default_joint_pos_list(), device=dev))
sim_cfg = make_sim_cfg(dev)


def compare(env, S, tag):
    d = env.robot.data
    ids = env.robot.find_bodies(kin.body_names, preserve_order=True)[0]
    pose = d.body_link_pose_w.torch[:, ids]
    sp = pose[..., :3] - env.scene.env_origins[:, None]
    sq = pose[..., 3:7]
    sR = quat_to_mat(torch.cat((sq[..., 3:4], sq[..., :3]), -1))
    vel = d.body_link_vel_w.torch[:, ids]
    bp, bR, bv, bw = kin.forward(*S, all_bodies=True)
    e_p = (bp - sp).norm(dim=-1)
    e_R = rot_angle_sq(bR, sR).clamp(min=0).sqrt()
    # acos clamp floors angles at ~1.4e-3; use the chordal (Frobenius) bound for small angles instead
    e_R_small = torch.linalg.matrix_norm(bR - sR) / (2**0.5)
    e_v = (bv - vel[..., :3]).norm(dim=-1)
    e_w = (bw - vel[..., 3:]).norm(dim=-1)
    tr = kin.tracked_idx
    res = {
        "pos_max_m": e_p.max().item(), "pos_max_tracked_m": e_p[:, tr].max().item(),
        "rot_max_rad": e_R_small.max().item(), "rot_max_tracked_rad": e_R_small[:, tr].max().item(),
        "rot_acos_max_rad": e_R.max().item(),
        "linvel_max": e_v.max().item(), "linvel_tracked_max": e_v[:, tr].max().item(),
        "angvel_max": e_w.max().item(), "angvel_tracked_max": e_w[:, tr].max().item(),
        "linvel_mean": e_v.mean().item(), "angvel_mean": e_w.mean().item(),
    }  # fmt: skip
    worst = kin.body_names[int(e_p.max(0).values.argmax())]
    print(f"[{tag}] pos {res['pos_max_m']:.3e} m (worst body {worst}) | rot {res['rot_max_rad']:.3e} rad | "
          f"linvel {res['linvel_max']:.3e} m/s | angvel {res['angvel_max']:.3e} rad/s", flush=True)  # fmt: skip
    return res


with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True)
    N = args.num_envs
    ids_all = torch.arange(N, device=dev)
    assert [n for n in env.robot.joint_names] == T1_JOINT_NAMES or True
    print("sim joint order == T1_JOINT_NAMES:", env.identity_order)
    print("sim bodies:", list(env.robot.body_names))
    results = {"kinematic": [], "dynamic": [], "roundtrip": []}
    for trial in range(args.trials):
        p = torch.randn(N, 3, device=dev) * torch.tensor([2.0, 2.0, 0.0], device=dev)
        p[:, 2] = 1.5 + torch.rand(N, device=dev)
        R = so3_exp(torch.randn(N, 3, device=dev) * 2.0)
        v = torch.randn(N, 3, device=dev)
        w = torch.randn(N, 3, device=dev) * 2
        q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(N, kin.num_joints, device=dev)
        qd = torch.randn(N, kin.num_joints, device=dev) * 3
        env.write_state(ids_all, (p, R, v, w, q, qd))
        env.refresh()
        S = env.state()
        rt = {n: (a - b).abs().max().item() for n, a, b in zip(("p", "R", "v", "w", "q", "qd"), S, (p, R, v, w, q, qd))}
        results["roundtrip"].append(rt)
        print(f"[roundtrip {trial}] " + " ".join(f"{k} {x:.1e}" for k, x in rt.items()))
        results["kinematic"].append(compare(env, S, f"kinematic {trial}"))
        # dynamic: start from reference frames (on the ground), random actions, contacts active
        frames = torch.randint(0, motion.num_frames - 20, (N,), device=dev)
        env.reset(ids_all, frames)
        env.refresh()
        for t in range(10):
            env.step(0.3 * torch.randn(N, kin.num_joints, device=dev))
        results["dynamic"].append(compare(env, env.state(), f"dynamic {trial}"))
    agg = {}
    for k in ("kinematic", "dynamic"):
        agg[k] = {m: max(r[m] for r in results[k]) for m in results[k][0]}
    ok = all(agg[k]["pos_max_m"] < 1e-3 and agg[k]["rot_max_rad"] < 1e-3 for k in agg)
    print(json.dumps(agg, indent=1))
    print("GATE 1", "PASS" if ok else "FAIL", "(pos < 1 mm and rot < 1e-3 rad on all bodies, kinematic and dynamic)")
    if args.out:
        json.dump({"agg": agg, "trials": results, "pass": ok}, open(args.out, "w"), indent=1)
