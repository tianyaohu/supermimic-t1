# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Checkpoint evaluation in the simulator (paper App. F.1, extended to every start-frame class).

Deterministic policy, no action noise, no observation noise, friction 1.0, no domain randomisation
(xi = [1, 0...0, 0, 0, 0]), no reset noise. One rollout starts at the first frame of every 1 s bin of
every clip (bin 0 of a clip is the paper's protocol: play the clip once from its first frame) and runs
until the clip's last frame or the training termination rule fires.

Per start: normalised return = sum of per-step reward / (steps to clip end * max per-step reward 0.1);
episode-length fraction = steps survived / steps to clip end; failed = termination fired.

Run: python supermimic_t1/evaluate.py --ckpt logs/supermimic/run/last.pt --out logs/supermimic/run/eval_last
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="output prefix; writes <out>.csv and <out>.json")
parser.add_argument("--repeats", type=int, default=1, help="rollouts per start frame (physics is deterministic)")
parser.add_argument("--record", type=str, default=None,
                    help="save an npz of the rollouts that start at a clip's first frame (root pose, joints, reference frame)")
add_launcher_args(parser)
args = parser.parse_args()

import torch

import supermimic_t1.networks as networks
from supermimic_t1.env import TrackingSim, make_sim_cfg
from supermimic_t1.kinematics import Kinematics
from supermimic_t1.mdp import MAX_STEP_REWARD, MDP
from supermimic_t1.motion import MotionLib
from supermimic_t1.networks import Policy
from supermimic_t1.robot import MOTION_CSV, MOTION_FPS, T1_JOINT_NAMES, default_joint_pos_list
from supermimic_t1.rotations import mat_to_quat
from supermimic_t1.state import robot_layout

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = Kinematics(dev)
motion = MotionLib([MOTION_CSV], kin, dev, src_fps=MOTION_FPS)
mdp = MDP(kin, motion, robot_layout(kin.num_joints), torch.tensor(default_joint_pos_list(), device=dev))
ck = torch.load(args.ckpt, map_location=dev)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
pi = Policy(mdp.obs_dim, motion.goal_frame_dim * K, kin.num_joints, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()

starts = motion.bin_start.repeat(args.repeats)
N = len(starts)
sim_cfg = make_sim_cfg(dev, njmax=a_.get("njmax", 300), nconmax=a_.get("nconmax", 64))
with launch_simulation(sim_cfg, args), torch.no_grad():
    env = TrackingSim(sim_cfg, N, motion, mdp, eval_mode=True)
    ids = torch.arange(N, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, starts.clone())
    clip_end = motion.clip_end[motion.clip_id[starts]]
    horizon = (clip_end - starts).float()  # transitions from the start frame to the clip's last frame
    ret = torch.zeros(N, device=dev)
    steps = torch.zeros(N, device=dev)
    hist = None
    rec_ids = torch.isin(starts, motion.clip_start).nonzero().flatten()
    err = {k: torch.zeros(N, device=dev) for k in ("body_pos_rms_m", "root_pos_m", "joint_abs_rad")}
    rec = {"root_p": [], "root_quat_wxyz": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if args.record:
            rec["root_p"].append(S[0][rec_ids].cpu()), rec["root_quat_wxyz"].append(mat_to_quat(S[1][rec_ids]).cpu())
            rec["q"].append(S[4][rec_ids].cpu()), rec["f"].append(env.f[rec_ids].cpu()), rec["alive"].append(alive[rec_ids].cpu())
        out = env.step(a)
        S1 = env.state()
        am = alive.float()
        err["body_pos_rms_m"] += am * out["terms"]["e_body_pos"].sqrt()
        err["root_pos_m"] += am * out["terms"]["e_root_pos"].sqrt()
        err["joint_abs_rad"] += am * (S1[4] - motion.q[env.f]).abs().mean(-1)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
    norm = ret / (horizon * MAX_STEP_REWARD)
    frac = steps / horizon
    failed = env.failed_eval.float()
    rows = []
    for i in range(N):
        b = int(motion.frame_bin[starts[i]])
        rows.append({"start_frame": int(starts[i]), "bin": b, "clip": motion.names[int(motion.bin_clip[b])],
                     "steps_possible": int(horizon[i]), "steps_survived": int(steps[i]),
                     "norm_return": round(norm[i].item(), 5), "ep_len_frac": round(frac[i].item(), 5),
                     "failed": int(failed[i].item())})  # fmt: skip
    first = [r for r, s in zip(rows, starts.tolist()) if s in motion.clip_start.tolist()]
    res = {
        "ckpt": os.path.abspath(args.ckpt), "iteration": ck["iteration"], "train_wall_h": ck["wall_time"] / 3600,
        "n_starts": N, "start_classes": "first frame of every 1 s bin of every clip",
        "norm_return_mean": norm.mean().item(),
        "norm_return_stepweighted": (ret.sum() / (horizon.sum() * MAX_STEP_REWARD)).item(),
        "ep_len_frac_mean": frac.mean().item(), "failure_rate": failed.mean().item(),
        "tracking_error_step_mean": {k: (v.sum() / steps.sum()).item() for k, v in err.items()},
        "tracking_error_note": "body_pos_rms_m: RMS over tracked bodies of re-anchored (xy+yaw) position error; "
                               "root_pos_m: global trunk position error (includes drift); joint_abs_rad: mean |q - q_ref| "
                               "incl. the arm frames where q_ref is outside the joint limits",
        "paper_protocol_from_first_frame": {"norm_return": sum(r["norm_return"] for r in first) / len(first),
                                            "ep_len_frac": sum(r["ep_len_frac"] for r in first) / len(first),
                                            "failed": sum(r["failed"] for r in first)},
    }  # fmt: skip
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(res, open(args.out + ".json", "w"), indent=1)
    if args.record:
        import numpy as np

        np.savez_compressed(args.record, **{k: torch.stack(v).numpy() for k, v in rec.items()},
                            start_frame=starts[rec_ids].cpu().numpy(), ref_root_p=motion.root_p.cpu().numpy(),
                            ref_root_quat_wxyz=mat_to_quat(motion.root_R).cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            joint_names=np.array(T1_JOINT_NAMES), ckpt=os.path.abspath(args.ckpt))  # fmt: skip
        print(f"recorded {len(rec_ids)} rollout(s), {len(rec['f'])} steps -> {args.record}")
    print(json.dumps(res, indent=1))
