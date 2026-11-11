"""Booster T1 (23 DoF) for Isaac Lab 3 / Newton (MuJoCo-Warp), without importing booster_train.

Structure adapted from supermimic_vibed/supermimic/robot.py (MIT, Copyright (c) 2026 ETH Zurich,
Chong Zhang). All T1 numbers are copied from booster_train (Apache-2.0), cited per constant:

  [BT1]  booster_train/source/booster_train/booster_train/assets/robots/booster_t1.py
  [ACT]  booster_train/source/booster_train/booster_train/assets/robots/actuator.py
  [ENV]  booster_train/.../beyond_mimic/robots/t1/motion_tracking/env_cfg.py
  [MJCF] booster_assets/robots/T1/T1_23dof.xml (joint order, limits, armature; same numbers as [ACT])

Actuator model: explicit PD at the 5 ms physics rate with booster_train's piecewise-linear
torque-speed (T-N) clipping ([ACT] BoosterDelayedPDActuator._clip_effort). The command delay of the
stock config ([BT1]: min_delay 2, max_delay 8 physics steps) is set to ZERO here on purpose: a hidden
delay buffer makes (q, qd) non-Markov for the world model. Everything else (gains, effort limits,
velocity limits, knee-point velocities, armature, action scale) follows [BT1]/[ACT].
"""

from __future__ import annotations

import math
import os
import re

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
MJCF_PATH = os.path.join(DATA_DIR, "t1", "T1_23dof_nomesh.xml")  # built by tools/make_assets.py
USD_PATH = os.path.join(DATA_DIR, "t1", "t1_23dof.usda")
GROUND_USD = os.path.join(DATA_DIR, "t1", "ground.usda")
MOTION_CSV = os.path.join(os.environ.get("BOOSTER_ASSETS_DIR", os.path.expanduser("~/booster_assets")), "motions/T1/t1_dance.csv")
MOTION_FPS = 50.0  # booster_assets README lists its Booster motions at 50 fps; t1_dance is not listed (assumption)

# Joint order of the Booster CSVs (booster_assets/src/booster_assets/motions.py T1_JOINT_NAMES); it is also
# the depth-first order of the MJCF tree. Every tensor of joints in supermimic_t1 uses this order.
T1_JOINT_NAMES = [
    "aahead_yaw_joint", "aahead_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]  # fmt: skip
NUM_J = len(T1_JOINT_NAMES)

ROOT_BODY = "trunk"  # free-joint body; also BeyondMimic-T1's anchor body ([ENV] anchor_body_name)
TORSO_COM_BODY = "trunk"  # [ENV-base] beyond_mimic/robots/t1/motion_tracking/tracking_env_cfg.py base_com event uses body_names="trunk"
# Tracked bodies ([ENV] commands.motion.body_names)
TRACKED_BODIES = [
    "trunk", "aahead_pitch_link",
    "left_shoulder_roll_link", "left_elbow_pitch_link", "left_elbow_yaw_link",
    "right_shoulder_roll_link", "right_elbow_pitch_link", "right_elbow_yaw_link",
    "left_hip_roll_link", "left_hip_yaw_link", "left_knee_pitch_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_hip_yaw_link", "right_knee_pitch_link", "right_ankle_pitch_link", "right_ankle_roll_link",
]  # fmt: skip

# ---- motor models [ACT]: (effort_limit, velocity_limit, knee_point_velocity, armature) ----
_E4310 = (38.3, 17.59, 7.85, 0.0282528)  # [ACT] BoosterJointE4310 (T1 arms)
_E6408 = (68.0, 14.66, 1.88, 0.0478125)  # [ACT] BoosterJointE6408 (T1 waist, hip roll/yaw)
_E8112 = (98.8, 15.29, 7.53, 0.0523908)  # [ACT] BoosterJointE8112 (T1 hip pitch)
_T1E8116 = (130.5, 14.76, 6.71, 0.0636012)  # [ACT] BoosterJointT1E8116 (T1 knee)
_DM4310 = (7.0, 41.89, 26.18, 0.0018)  # [ACT] BoosterJointDM4310 (T1 head)
_E4315 = (43.0, 12.15, 8.38, 0.0339552)  # [ACT] BoosterJointE4315 (base of the ankle wrapper)
# [BT1] BoosterT1AnkleParaWrapperCfg(effort_ratio=(1.7, 0.4), velocity_ratio=(0.6, 1.0),
#        armature_ratio=(2, 0.6), knee_point_velocity_ratio=(1.0, 1.0)); serial_index 0 pitch, 1 roll
_ANKLE_PITCH = (1.7 * _E4315[0], 0.6 * _E4315[1], 1.0 * _E4315[2], 2.0 * _E4315[3])
_ANKLE_ROLL = (0.4 * _E4315[0], 1.0 * _E4315[1], 1.0 * _E4315[2], 0.6 * _E4315[3])

JOINT_MOTOR = {
    "aahead_.*": _DM4310,
    ".*_shoulder_pitch_joint": _E4310, ".*_shoulder_roll_joint": _E4310,
    ".*_elbow_pitch_joint": _E4310, ".*_elbow_yaw_joint": _E4310,
    "waist_yaw_joint": _E6408,
    ".*_hip_pitch_joint": _E8112, ".*_hip_roll_joint": _E6408, ".*_hip_yaw_joint": _E6408,
    ".*_knee_pitch_joint": _T1E8116,
    ".*_ankle_pitch_joint": _ANKLE_PITCH, ".*_ankle_roll_joint": _ANKLE_ROLL,
}  # fmt: skip
NATURAL_FREQ_HZ = 4.0  # [BT1] every actuator group: natural_freq=4.0
DAMPING_RATIO = 1.5  # [BT1] every actuator group: damping_ratio=1.5

# [BT1] per_joint_scale_coefficient; joints not listed (head) use 0.25. scale = coef * effort / kp
ACTION_SCALE_COEF = {
    "waist_yaw_joint": 0.16, ".*_hip_yaw_joint": 0.28, ".*_hip_roll_joint": 0.35, ".*_hip_pitch_joint": 0.35,
    ".*_knee_pitch_joint": 0.3, ".*_ankle_pitch_joint": 0.17, ".*_ankle_roll_joint": 0.28,
    ".*_shoulder_pitch_joint": 0.5, ".*_shoulder_roll_joint": 0.5, ".*_elbow_pitch_joint": 0.5,
    ".*_elbow_yaw_joint": 0.7,
}  # fmt: skip

# [BT1] init_state
DEFAULT_ROOT_HEIGHT = 0.70
DEFAULT_JOINT_POS = {
    ".*_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": -1.3, "right_shoulder_roll_joint": 1.3,
    "left_elbow_yaw_joint": -0.5, "right_elbow_yaw_joint": 0.5, ".*_hip_pitch_joint": -0.2,
    ".*_knee_pitch_joint": 0.4, ".*_ankle_pitch_joint": -0.2,
}  # fmt: skip


def _match(table: dict, name: str, default=None):
    for pat, v in table.items():
        if re.fullmatch(pat, name):
            return v
    if default is None:
        raise KeyError(name)
    return default


def motor(name: str):
    return _match(JOINT_MOTOR, name)


def kp_kd(name: str) -> tuple[float, float]:
    """[ACT] BoosterJointCfg.__post_init__: kp = J w^2, kd = 2 zeta J w."""
    arm = motor(name)[3]
    w = 2 * 3.1415926535 * NATURAL_FREQ_HZ
    return arm * w**2, 2 * DAMPING_RATIO * arm * w


def action_scale(name: str) -> float:
    return _match(ACTION_SCALE_COEF, name, 0.25) * motor(name)[0] / kp_kd(name)[0]


def default_joint_pos_list() -> list[float]:
    return [_match(DEFAULT_JOINT_POS, n, 0.0) for n in T1_JOINT_NAMES]


# ------------------------------------------------------------------------------------------ Isaac Lab
def make_robot_cfg():
    """ArticulationCfg for the generated USD. Imported lazily so FK/motion code needs no Isaac Lab."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.utils import configclass

    class T1TNCurvePDActuator(IdealPDActuator):
        """Explicit PD with booster_train's T-N curve ([ACT] BoosterDelayedPDActuator, delay removed):
        full torque for |qd| <= v_knee, linear to zero at v_max."""

        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            v_max = self.actuator_velocity_limit
            knee = torch.tensor([cfg.knee_point_velocity[n] for n in self._joint_names], device=self._device)
            self._v_knee = torch.minimum(knee.expand_as(v_max).clamp(min=0.0), v_max)
            self._denom = (v_max - self._v_knee).clamp(min=1e-6)
            self._joint_vel = torch.zeros_like(self.computed_effort)

        def compute(self, control_action, joint_pos, joint_vel):
            self._joint_vel[:] = joint_vel
            return super().compute(control_action, joint_pos, joint_vel)

        def _clip_effort(self, effort):
            v_max, tau_max = self.actuator_velocity_limit, self.actuator_effort_limit
            tau = (tau_max * (v_max - self._joint_vel.abs()) / self._denom).clamp(min=0.0)
            tau = torch.minimum(tau, tau_max)
            return torch.clip(effort, min=-tau, max=tau)

    @configclass
    class T1TNCurvePDActuatorCfg(IdealPDActuatorCfg):
        class_type: type = T1TNCurvePDActuator
        knee_point_velocity: dict = {}

    # one actuator group with exact per-joint values (23 groups cost ~6 ms per write_data_to_sim at 2048 envs)
    per = lambda f: {n: float(f(n)) for n in T1_JOINT_NAMES}  # noqa: E731
    acts = {
        "all": T1TNCurvePDActuatorCfg(
            joint_names_expr=list(T1_JOINT_NAMES),
            actuator_effort_limit=per(lambda n: motor(n)[0]),
            actuator_velocity_limit=per(lambda n: motor(n)[1]),
            joint_effort_limit=per(lambda n: motor(n)[0]),
            joint_velocity_limit=per(lambda n: motor(n)[1]),
            knee_point_velocity=per(lambda n: motor(n)[2]),
            stiffness=per(lambda n: kp_kd(n)[0]),
            damping=per(lambda n: kp_kd(n)[1]),
            armature=per(lambda n: motor(n)[3]),
        )
    }
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, DEFAULT_ROOT_HEIGHT), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=acts,
    )


if __name__ == "__main__":
    for n in T1_JOINT_NAMES:
        kp, kd = kp_kd(n)
        print(f"{n:28s} kp {kp:7.2f} kd {kd:6.3f} tau {motor(n)[0]:6.1f} vmax {motor(n)[1]:6.2f} scale {action_scale(n):.3f}")

