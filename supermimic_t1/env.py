# Adapted from supermimic_vibed (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""T1 motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

A thin batched wrapper: applies the joint setpoint, steps MuJoCo-Warp 4 x 5 ms per 50 Hz control step
(the explicit PD actuator is evaluated at every 5 ms physics step, as in booster_train), labels the
transition with the analytic reward/termination of mdp.py, and resets episodes to reference states
drawn by return-based bin sampling (paper App. D.1) with BeyondMimic's reset noise.

The sim context xi = [ground friction, joint default offsets (nj), torso CoM offset (3)] is a tensor
(``self.xi``) that is written into the simulator and stored with every transition.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from .mdp import MAX_STEP_REWARD, MDP
from .motion import CONTROL_DT, MotionLib
from .reset_sampling import update_bin_returns
from .robot import GROUND_USD, T1_JOINT_NAMES, TORSO_COM_BODY, action_scale, make_robot_cfg
from .rotations import mat_to_quat, quat_to_mat, so3_exp

PHYSICS_DT = 0.005
DECIMATION = round(CONTROL_DT / PHYSICS_DT)  # 4


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.UsdFileCfg(usd_path=GROUND_USD))
    robot: ArticulationCfg = make_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")


def make_sim_cfg(device: str, njmax: int = 300, nconmax: int = 64):
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=njmax, nconmax=nconmax, cone="pyramidal", integrator="implicitfast",
                                   use_mujoco_contacts=True),  # fmt: skip
        num_substeps=1,
    )
    # use_newton_actuators=False: the T1 T-N-curve PD is a custom explicit actuator that Newton cannot run
    # natively, so Isaac Lab evaluates it in write_data_to_sim(), which physics_step() calls every 5 ms.
    return sim_utils.SimulationCfg(dt=PHYSICS_DT, device=device, physics=physics, use_newton_actuators=False)


class TrackingSim:
    """Batched T1 tracking simulator. Construct inside ``launch_simulation(sim_cfg, args)``."""

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99,
                 eval_mode: bool = False, max_episode_steps: int = 500, reset_noise: bool = True):  # fmt: skip
        self.device = dev = sim_cfg.device
        self.sim = SimulationContext(sim_cfg)
        self.scene = InteractiveScene(TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True))
        self.sim.reset()
        self.robot = self.scene["robot"]
        names = list(self.robot.joint_names)
        self.jidx = torch.tensor([names.index(n) for n in T1_JOINT_NAMES], device=dev)
        self.identity_order = names == T1_JOINT_NAMES
        self.N = num_envs
        self.nj = len(T1_JOINT_NAMES)
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.default_q = mdp.default_q
        self.alpha = torch.tensor([action_scale(n) for n in T1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies(TORSO_COM_BODY)[0][0]
        self.soft_lo, self.soft_hi = mdp.soft_lo, mdp.soft_hi

        # ---- sim context xi = [friction, joint offsets(nj), torso com offset(3)] ----
        nj = self.nj
        self.xi = torch.zeros(self.N, mdp.xi_dim, device=dev)
        self._setup_friction()
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack((torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                               torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                               torch.empty(self.N, device=dev).uniform_(-0.05, 0.05)), -1)  # fmt: skip
            self.xi[:, 1 + nj : 4 + nj] = com
            default_com = self.robot.data.body_com_pos_b.torch[:, self.torso_id, :3].clone()
            self.robot.set_coms_index(coms=(default_com + com).unsqueeze(1).contiguous(), body_ids=[self.torso_id])
        self._apply_friction()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, nj, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        model = NewtonManager.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # Ground shapes (body == -1) get the lowest friction of the range: MuJoCo combines a contact pair's
        # friction by taking the max, so the robot shapes' value (xi[0]) is the effective friction.
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = 0.3 if not self.eval_mode else 1.0

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ joints in T1 order
    def _jget(self, x):
        return x.clone() if self.identity_order else x[:, self.jidx]

    def _jset(self, x):
        if self.identity_order:
            return x
        out = torch.empty_like(x)
        out[:, self.jidx] = x
        return out

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w.torch
        vel = d.root_link_vel_w.torch
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # Isaac Lab 3 uses xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (p, R, vel[:, :3].clone(), vel[:, 3:6].clone(), self._jget(d.joint_pos.torch), self._jget(d.joint_vel.torch))

    def write_state(self, env_ids, S):
        """Write (p, R, v, w, q, qd) into the simulator for ``env_ids`` (no stepping)."""
        p, R, v, w, q, qd = S
        quat = mat_to_quat(R)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=self._jset(q), env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self._jset(qd), env_ids=env_ids)

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int):
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n, dev = len(env_ids), self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        if self.reset_noise:  # BeyondMimic-T1 ranges (tracking_env_cfg.py pose_range / VELOCITY_RANGE)
            u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)  # noqa: E731
            p = p + torch.stack((u(-0.05, 0.05), u(-0.05, 0.05), u(-0.01, 0.01)), -1)
            R = so3_exp(torch.stack((u(-0.1, 0.1), u(-0.1, 0.1), u(-0.2, 0.2)), -1)) @ R
            v = v + torch.stack((u(-0.5, 0.5), u(-0.5, 0.5), u(-0.2, 0.2)), -1)
            w = w + torch.stack((u(-0.52, 0.52), u(-0.52, 0.52), u(-0.78, 0.78)), -1)
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
            q = torch.clamp(q, self.soft_lo, self.soft_hi)  # BeyondMimic clips to the soft limits
        else:
            q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self.write_state(env_ids, (p, R, v, w, q, qd))
        self.robot.set_joint_position_target_index(target=self._jset(q), env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1 : 1 + self.nj] = torch.empty(n, self.nj, device=dev).uniform_(-0.01, 0.01)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states (forward kinematics) so body poses are consistent without stepping."""
        self.sim.forward()
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    # GPU-KERNEL-CANDIDATE: collection, 73 ms/iteration (12%; 2.5 control steps), of which physics is ~8 ms per control
    #   step (4 x 5 ms MuJoCo-Warp CUDA-graph steps at 2048 envs) and the rest is Python/launch overhead: the explicit
    #   T-N-curve PD runs in Isaac Lab's Python actuator path between graph launches (use_newton_actuators=False), and
    #   state readout + FK + reward + termination are unfused torch ops (~8 ms per control step, measured with
    #   tools/test_env.py: 16.2 ms env.step vs 8.2 ms physics). A Warp kernel for the PD + torque-speed clip inside the
    #   captured physics graph, plus one fused readout/FK/reward kernel, could roughly halve collect time.
    def physics_step(self, target: torch.Tensor):
        self.robot.set_joint_position_target_index(target=self._jset(target))
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()  # explicit PD + T-N clipping evaluated every 5 ms
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)

    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f, xi, a_prev, ep_step = self.f.clone(), self.xi.clone(), self.a_prev.clone(), self.ep_step.clone()
        self.physics_step(self.default_q + xi[:, 1 : 1 + self.nj] + self.alpha * a)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        lay = self.mdp.layout
        out = {"s": lay.flatten(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": lay.flatten(S1), "r": r,
               "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms}  # fmt: skip
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            # finished envs keep stepping (no auto-reset in eval); keep f + 1 a valid frame of the clip
            self.f = torch.minimum(self.f, clip_end - 1)
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            update_bin_returns(self.bin_G, self.start_bin[ids], self.ep_disc_ret[ids] + boot)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            out["ep_term"] = term[ids]
            self.reset(ids)
        return out
