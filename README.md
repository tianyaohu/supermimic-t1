# SuperMimic on Booster T1

Unofficial reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* ([arXiv 2610.28480](https://arxiv.org/abs/2610.28480)) for the Booster T1 humanoid, on Isaac Lab 3.0 (kit-less, Newton with the MuJoCo-Warp solver).

The policy is trained by backpropagating a motion-tracking reward through short imagined rollouts of a learned world model that is fit to simulator transitions. There is no PPO component.

![T1 tracking the dance clip](media/t1_dance.gif)

Solid: simulated robot. Blue ghost: reference motion. First 16 s of the clip; the full 32.7 s video is [media/t1_dance_final.mp4](media/t1_dance_final.mp4).

## Status

One clip, one seed, simulation only. This shows the pipeline works on T1; it is not the paper's benchmark.

| Check | Result |
|---|---|
| Forward kinematics vs simulator, all 24 links | max error 9.8e-7 m, 7.0e-7 rad |
| World model only, 3000 iterations | held-out 8-step normalised error 0.409 to 0.035 |
| Training on `t1_dance`, 45 min, RTX 3080 Ti (12 GB) | 4856 iterations, 1.74 it/s, peak 5.99 GiB (torch) |
| Evaluation, deterministic policy, 33 starts (one per second of clip) | normalised return 0.976, 0 failures |
| Evaluation from frame 0 over the whole clip | normalised return 0.971 |
| Mean tracked-body position error | 1.6 cm |

Settings: 2048 environments, 2048 imagined rollouts of 32 steps, bf16 autocast, `torch.compile`.

## Known limits and deviations

- Training and evaluation use the same single motion, so generalisation is not tested.
- The tracking reward is lenient; 0.97 is not near-perfect imitation.
- Actuator command delay is zero. Booster's stock T1 config uses a random 2 to 8 step delay.
- Self-collisions are off.
- The clip's frame rate is assumed to be 50 fps; the file does not state it.
- The reference arm poses exceed the T1 joint limits in 2.9% of (frame, joint) entries.
- 2048 imagined rollouts instead of the paper's 4096, to fit in 12 GB.
- No model-free baseline has been run yet.
- The code has passed the checks above but has not had an independent line-by-line review.

## Setup

1. Isaac Lab 3.0.0 with `isaaclab_newton`, `newton`, `mujoco-warp` and PyTorch (tested: Isaac Lab 3.0.0 release, newton 1.6.0, mujoco-warp 3.12.0, warp-lang 1.17.0, torch 2.12).
2. Booster's robot description and motion data are not redistributed here. Clone [booster_assets](https://github.com/BoosterRobotics/booster_assets) and point `BOOSTER_ASSETS_DIR` at it (default `~/booster_assets`).

```sh
export BOOSTER_ASSETS_DIR=/path/to/booster_assets
export PYTHONPATH=$PWD
PY=/path/to/IsaacLab/.venv/bin/python

$PY supermimic_t1/tools/make_assets.py          # writes data/t1/: mesh-free MJCF, USD, ground
$PY supermimic_t1/tools/test_fk.py --num_envs 512 --trials 4 --out logs/gate1_fk.json
```

## Train and evaluate

```sh
# world model only (sanity check)
$PY supermimic_t1/train.py --policy_mode zero --iterations 3000 --num_envs 2048 --wm_batch 2048 \
    --log_every 50 --ckpt_every 1000 --log_dir logs/wm_only

# full training, 45 minutes
$PY supermimic_t1/train.py --num_envs 2048 --pi_batch 2048 --wm_batch 2048 --wall_minutes 45 \
    --log_every 50 --ckpt_every 500 --log_dir logs/dance

$PY supermimic_t1/evaluate.py --ckpt logs/dance/last.pt --out logs/dance/eval_last
python3 supermimic_t1/tools/summarize.py logs/dance/train_log.csv
```

`evaluate.py --record` saves a trajectory that `supermimic_t1/tools/render.py` turns into a video with MuJoCo (EGL).

## Credits and licences

- Method: Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath, *SuperMimic*, 2026.
- Parts of the code are adapted from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) by Chong Zhang (MIT). See `THIRD_PARTY_NOTICES`.
- Robot description and motion: Booster Robotics `booster_assets` (BSD-3-Clause); actuator constants from `booster_train` (Apache-2.0). Neither is redistributed.
- This repository: MIT.
