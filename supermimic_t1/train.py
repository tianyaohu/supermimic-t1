# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""SuperMimic training for the Booster T1 (Algorithm A.1): collect -> fit world model -> BPTT policy + critic.

Isaac Lab 3 kit-less, Newton/MuJoCo-Warp. Example:
  python supermimic_t1/train.py --num_envs 2048 --pi_batch 2048 --log_dir logs/supermimic/run
Gate 2 (world model only, zero policy + action noise):
  python supermimic_t1/train.py --policy_mode zero --iterations 3000 --log_dir logs/supermimic/gate2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--wall_minutes", type=float, default=0.0, help="stop after this much wall-clock (0 = off)")
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorised simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=2048)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--policy_mode", choices=["train", "zero"], default="train",
                    help="zero: never update the policy and act with zero action + noise (Gate 2)")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=1000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--max_episode_steps", type=int, default=500)
parser.add_argument("--njmax", type=int, default=300)
parser.add_argument("--nconmax", type=int, default=64)
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "supermimic", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--compile", type=int, default=1, help="torch.compile the hot functions")
parser.add_argument("--profile_every", type=int, default=0, help="time the phases every n iterations (syncs)")
parser.add_argument("--torch_profile_at", type=int, default=0,
                    help="run torch.profiler over iterations [n, n+3) and write kernel tables to log_dir (0 = off)")
add_launcher_args(parser)
args = parser.parse_args()

import torch
import torch.nn as nn

import supermimic_t1.networks as networks
from supermimic_t1.buffer import ReplayBuffer
from supermimic_t1.env import TrackingSim, make_sim_cfg
from supermimic_t1.kinematics import Kinematics
from supermimic_t1.mdp import MAX_STEP_REWARD, MDP
from supermimic_t1.motion import MotionLib
from supermimic_t1.networks import Critic, Policy, WorldModel
from supermimic_t1.robot import MOTION_CSV, MOTION_FPS, default_joint_pos_list
from supermimic_t1.state import robot_layout

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = Kinematics(dev)
NJ = kin.num_joints
layout = robot_layout(NJ)
motion = MotionLib([MOTION_CSV], kin, dev, src_fps=MOTION_FPS)
mdp = MDP(kin, motion, layout, torch.tensor(default_joint_pos_list(), device=dev))
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_frame_dim * K
CRITIC_IN = layout.feat_dim + mdp.critic_extra + GOAL_DIM
OBS_DIM = mdp.obs_dim

wm = WorldModel(layout, NJ, mdp.xi_dim).to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, NJ, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
print(f"[dims] nj {NJ} state {layout.flat_dim} feat {layout.feat_dim} inc {layout.inc_dim} obs {OBS_DIM} "
      f"goal {GOAL_DIM} xi {mdp.xi_dim} critic_in {CRITIC_IN} | params wm {sum(p.numel() for p in wm.parameters())/1e6:.1f}M "
      f"pi {sum(p.numel() for p in pi.parameters())/1e6:.1f}M", flush=True)  # fmt: skip

ERR_FLOOR = layout.err_floor().to(dev)
COMP = layout.component_slices()  # name -> slice into the increment/error vector


def _wm_rollout(s, a, xi):
    S = layout.unflatten(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], mdp.normalize_xi(xi[:, j]))
        preds.append(S)
    return layout.stack(preds, 1)  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    raw = layout.error(S_hat, layout.unflatten(s1))  # (B, k, D) physical units, frame-local
    e = ((raw / std) ** 2).mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1], raw


def _imag_step(S, o_hist, f, xi, xi_n):
    a = pi(o_hist, motion.goal(f, K))
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n):
    return torch.cat((mdp.critic_features(S, f, xi_n), motion.goal(f, K)), -1)


def _sim_reward(S1, f1, a, a_prev):
    return mdp.reward(S1, f1, a, a_prev, terms=True), mdp.termination(S1, f1)


maybe = (lambda fn: torch.compile(fn)) if args.compile else (lambda fn: fn)  # noqa: E731
wm_rollout, wm_err, imag_step = maybe(_wm_rollout), maybe(_wm_err), maybe(_imag_step)
reward_term, critic_input = maybe(_reward_term), maybe(_critic_input)
mdp_obs = maybe(mdp.obs)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


class Timer:
    """Phase timer; synchronises only on profiled iterations."""

    def __init__(self):
        self.on, self.acc, self.n, self._t = False, {}, 0, None

    def start(self, on: bool, count: bool = True):
        self.on = on
        if on:
            torch.cuda.synchronize()
            self._t = time.time()
            self.n += int(count)

    def mark(self, name: str):
        if self.on:
            torch.cuda.synchronize()
            t = time.time()
            self.acc[name] = self.acc.get(name, 0.0) + (t - self._t)
            self._t = t

    def summary(self):
        tot = sum(self.acc.values())
        return {k: (1e3 * v / max(self.n, 1), 100 * v / max(tot, 1e-9)) for k, v in self.acc.items()}


timer = Timer()


# ------------------------------------------------------------------------------------------------
# GPU-KERNEL-CANDIDATE: world-model fit, 250 ms/iteration = 43% (Gate 3 phase timer, 2048 windows x 8 steps x 10
#   batches). GPU-bound: 243 ms of kernel time per iteration, mostly bf16 cuBLAS GEMMs (3x2048 MLP), which are near
#   peak already. A kernel could fuse the LayerNorm+SiLU epilogue into the GEMMs and fuse the per-step state
#   integration (so3_exp, yaw frame, quaternion conversions; ~100 small launches per step) into one kernel; expected
#   gain is modest (~10-20%), because the GEMMs dominate.
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last, _ = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def wm_heldout(B: int = 2048):
    """k-step open-loop error on the newest windows (collected this iteration, not yet trained on)."""
    w = buffer.sample_windows(B, args.wm_window, newest=True)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    _, e_last, raw = wm_err(S_hat, w["s1"], w["mask"])
    m = w["mask"][:, -1] > 0
    out = {"e8_norm": e_last[m].mean().item(), "n": int(m.sum())}
    r = raw[:, -1][m]  # error after k steps, physical units
    for name, sl in COMP.items():
        x = r[:, sl]
        # vector components: RMS of the per-sample vector norm; joints: RMS over joints and samples
        out[name] = (x.norm(dim=-1).pow(2).mean().sqrt() if name in ("pos", "rot", "linvel", "angvel")
                     else x.pow(2).mean().sqrt()).item()  # fmt: skip
    # zero-model baseline: S_hat = S_0 held constant (no motion), same metric, for scale
    S0 = layout.unflatten(w["s"][:, 0])
    rawc = layout.error(S0, layout.unflatten(w["s1"][:, -1]))[m]
    for name, sl in COMP.items():
        x = rawc[:, sl]
        out["hold_" + name] = (x.norm(dim=-1).pow(2).mean().sqrt() if name in ("pos", "rot", "linvel", "angvel")
                               else x.pow(2).mean().sqrt()).item()  # fmt: skip
    return out


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        ok = s["f"] + HZ <= motion.clip_end[motion.clip_id[s["f"]]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


# GPU-KERNEL-CANDIDATE: imagined rollout, forward 88 ms (15%) + backward 123 ms (21%) per iteration (H=32, B=2048).
#   Besides the MLP GEMMs, the analytic heads run the 25-body FK chain, reward and observation every step as hundreds
#   of tiny ops; tiny batched 3x3 matmuls alone are 42 ms/iteration (8% of all kernel time, torch.profiler). A Warp
#   kernel computing FK + reward + soft termination (+ its hand-written adjoint) per sample in one launch would remove
#   most of these launches and the saved intermediates (which also limit pi_batch on 12 GB). Most promising candidate.
def policy_update():
    s, h = sample_valid_starts(args.pi_batch)
    S = layout.unflatten(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = mdp.normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.add_noise(mdp_obs(layout.unflatten(h["s"]), h["a_prev"], h["xi"], h["f"]))
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    S0, f0 = S, f
    Ss, As, Aps, Fs = [], [], [], []
    for _ in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f = imag_step(S, o_hist, f, xi, xi_n)
        Ss.append(S), As.append(a_prev), Fs.append(f)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(t.flatten(0, 1) for t in layout.stack(Ss, 0))
    r, d = reward_term(S_all, flat(Fs), flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(critic_input(S, f, xi_n))
    loss = -J.mean()
    timer.mark("imagine_fwd")
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    timer.mark("imagine_bwd")
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    v_loss = torch.zeros((), device=dev)
    # GPU-KERNEL-CANDIDATE: critic fit, 49 ms/iteration (8%): 20 Adam steps on a (256,128) MLP over 65k samples.
    #   Launch-bound (tiny GEMMs). A CUDA-graph capture of the 20 steps (or one fused small-MLP training kernel)
    #   should cut most of it; a custom kernel is not needed first.
    if not args.no_critic:  # critic: GAE(lambda) targets over the imagined transitions
        with torch.no_grad():
            S_c = tuple(torch.cat((a.unsqueeze(0), b.detach()), 0).flatten(0, 1) for a, b in zip(S0, layout.stack(Ss, 0)))
            f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input(S_c, f_c, xi_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
    timer.mark("critic")
    return loss.detach(), v_loss.detach(), (r_sum / HZ / MAX_STEP_REWARD).mean(), gn.detach(), c.mean().detach()


def gpu_mem_smi() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout  # fmt: skip
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim_cfg(dev, njmax=args.njmax, nconmax=args.nconmax)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, max_episode_steps=args.max_episode_steps)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, layout.flat_dim, NJ, mdp.xi_dim)
    print(f"[mem] replay buffer {buffer.nbytes() / 2**30:.2f} GiB", flush=True)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp_obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist),
                                   torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))  # fmt: skip
            g = motion.goal(env.f, K)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g) if args.policy_mode == "train" else torch.zeros(args.num_envs, NJ, device=dev)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
            out = env.step(a)
            buffer.add(out)
            S0, S1 = layout.unflatten(out["s"]), layout.unflatten(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], mdp.normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].sum()), stats["ep_n"].append(len(out["ep_len"]))
                stats["ep_ret"].append(out["ep_ret_norm"].sum())
                stats["term"].append(out["ep_term"].float().sum())

    stats = {"r": [], "ep_len": [], "ep_n": [], "ep_ret": [], "term": []}
    if buffer.size < args.wm_window:
        collect(max(args.warmup_steps, args.wm_window), stats)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    new_log = not os.path.exists(log_path) or start_it == 0
    log_f = open(log_path, "w" if new_log else "a")
    cols = ["iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "ho_e8_norm", "ho_pos", "ho_rot", "ho_linvel",
            "ho_angvel", "ho_q", "ho_qd", "hold_pos", "hold_rot", "hold_linvel", "hold_angvel", "hold_q", "hold_qd",
            "pi_loss", "v_loss", "imag_rew", "pi_gradnorm", "imag_survival", "sim_rew", "ep_len", "ep_ret_norm",
            "term_frac", "episodes", "it_per_s", "lr", "vram_torch_peak_gib", "vram_smi_mib"]  # fmt: skip
    if new_log:
        log_f.write(",".join(cols) + "\n")
    prof_f = open(os.path.join(args.log_dir, "profile.jsonl"), "a")
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
    it = start_it
    for it in range(start_it, args.iterations):
        lr = lr_at(it)
        for o in (opt_wm, opt_pi, opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = args.profile_every > 0 and (it + 1) % args.profile_every == 0 and it > args.wm_only_iters + 5
        if args.torch_profile_at and it == args.torch_profile_at:
            from torch.profiler import ProfilerActivity, profile, record_function

            tprof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            tprof.__enter__()
        tp_on = bool(args.torch_profile_at) and args.torch_profile_at <= it < args.torch_profile_at + 3
        rf = (lambda name: record_function(name)) if tp_on else (lambda name: __import__("contextlib").nullcontext())  # noqa: E731
        timer.start(prof)
        with rf("phase_collect"):
            collect(n, stats)
        timer.mark("collect")
        ho = wm_heldout() if (it + 1) % args.log_every == 0 else None
        timer.start(prof, count=False)  # held-out eval excluded from the profile
        with rf("phase_wm_update"):
            for _ in range(args.wm_batches):
                l, e8 = wm_update()
        timer.mark("wm_update")
        agg["wm"].append(l), agg["wm8"].append(e8)
        if it >= args.wm_only_iters and args.policy_mode == "train":
            with rf("phase_policy_critic"):
                pl, vl, ir, gn, surv = policy_update()
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
        if args.torch_profile_at and it == args.torch_profile_at + 2:
            torch.cuda.synchronize()
            tprof.__exit__(None, None, None)
            ka = tprof.key_averages()
            open(os.path.join(args.log_dir, "torch_profile_cuda.txt"), "w").write(
                ka.table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=90))
            tprof.export_chrome_trace(os.path.join(args.log_dir, "torch_profile_trace.json"))
            print(f"[torch_profile] wrote tables for iterations {args.torch_profile_at}-{it}", flush=True)
        if prof:
            prof_f.write(json.dumps({"iteration": it + 1, **{k: v[0] for k, v in timer.summary().items()}}) + "\n")
            prof_f.flush()
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            n_ep = sum(int(v) for v in stats["ep_n"])
            ep_len = float(sum(float(v) for v in stats["ep_len"])) / max(n_ep, 1)
            ep_ret = float(sum(float(v) for v in stats["ep_ret"])) / max(n_ep, 1)
            term_frac = float(sum(float(v) for v in stats["term"])) / max(n_ep, 1)
            row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), ho["e8_norm"], ho["pos"], ho["rot"],
                   ho["linvel"], ho["angvel"], ho["q"], ho["qd"], ho["hold_pos"], ho["hold_rot"], ho["hold_linvel"],
                   ho["hold_angvel"], ho["hold_q"], ho["hold_qd"], mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                   mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), ep_len, ep_ret, term_frac, n_ep, ips, lr,
                   torch.cuda.max_memory_allocated() / 2**30, gpu_mem_smi()]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} ho_e8 {ho['e8_norm']:.4f} "
                  f"(pos {ho['pos']:.4f} rot {ho['rot']:.4f} q {ho['q']:.4f}) | pi {row[18]:.3f} imag_r {row[20]:.3f} "
                  f"surv {row[22]:.3f} | sim_r {row[23]:.3f} ep_len {ep_len:.1f} ep_ret {ep_ret:.3f} term {term_frac:.2f} "
                  f"| {ips:.2f} it/s | vram {row[30]:.2f} GiB / smi {row[31]:.0f} MiB", flush=True)  # fmt: skip
            if prof and timer.n:
                print("[prof] " + " | ".join(f"{k} {v[0]:.0f} ms ({v[1]:.0f}%)" for k, v in timer.summary().items()), flush=True)
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        done_wall = args.wall_minutes > 0 and (time.time() - t_start) / 60 >= args.wall_minutes
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or done_wall:
            ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                  "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
        if done_wall:
            print(f"[stop] wall-clock limit {args.wall_minutes} min reached at iteration {it + 1}", flush=True)
            break
    if timer.n:
        json.dump(timer.summary(), open(os.path.join(args.log_dir, "profile_summary.json"), "w"), indent=1)
        print("[prof summary] " + json.dumps(timer.summary()), flush=True)
