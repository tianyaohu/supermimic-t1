# Adapted from supermimic_vibed tools/render.py (MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.
"""Render a recorded evaluation rollout (evaluate.py --record) offscreen with MuJoCo (EGL).

Solid: the simulated T1 (Newton/MuJoCo-Warp rollout, replayed kinematically). Translucent blue: the reference
motion at the same frame. By default the ghost is drawn at its *global* position, so any drift of the robot
is visible; ``--anchored_ghost`` moves it to the robot's xy (as the body-tracking reward compares them).
Tracking camera, 50 fps, H.264 (yuv420p, faststart) for browsers.

Run: MUJOCO_GL=egl python supermimic_t1/tools/render.py traj.npz out.mp4 [--rollout 0] [--seconds 0 (= all)]
"""

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_MJCF = os.path.join(os.environ.get("BOOSTER_ASSETS_DIR", os.path.expanduser("~/booster_assets")), "robots/T1/T1_23dof.xml")


def two_robot_xml(w: int, h: int) -> str:
    xml = open(SRC_MJCF).read()
    meshdir = os.path.join(os.path.dirname(SRC_MJCF), "meshes") + "/"
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshdir}"', xml)
    xml = re.sub(r"<light [^>]*/>", "", xml)
    xml = re.sub(r'<geom name="ground"[^>]*/>', "", xml)
    xml = re.sub(r"<sensor>.*?</sensor>", "", xml, flags=re.S)
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    start = xml.index('<body name="trunk"')
    end = xml.index("</worldbody>")
    body = xml[start:end]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r"<site [^>]*/>", "", ghost)
    ghost = re.sub(r"<geom (?![^>]*mesh=)[^>]*/>", "", ghost)  # drop invisible collision primitives
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.30 0.60 1.0 0.30"', ghost)
    ghost = re.sub(r' (contype|conaffinity)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" ')
    scene = """
    <light pos="1 1 5" dir="-0.2 -0.2 -1" directional="true" diffuse="0.75 0.75 0.75" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml[:end] + ghost + scene + xml[end:]
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.78 0.80 0.84" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<asset>", f'<visual><global offwidth="{w}" offheight="{h}"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = whole rollout")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--anchored_ghost", action="store_true")
    ap.add_argument("--title", type=str, default="")
    a = ap.parse_args()

    d = np.load(a.traj)
    names = [str(n) for n in d["joint_names"]]
    model = mujoco.MjModel.from_xml_string(two_robot_xml(a.width, a.height))
    model.vis.headlight.ambient[:] = 0.4
    model.vis.headlight.diffuse[:] = 0.5
    data = mujoco.MjData(model)
    qa = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    qa_ref = np.array([model.jnt_qposadr[model.joint("ref_" + n).id] for n in names])
    fa = model.jnt_qposadr[model.joint("world_joint").id]
    fa_ref = model.jnt_qposadr[model.joint("ref_world_joint").id]
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.6, 140.0, -10.0
    font = ImageFont.load_default(size=22)
    c = a.rollout
    T = d["root_p"].shape[0]
    t_end = T if a.seconds <= 0 else min(T, int(a.seconds * 50))
    stride = max(1, int(round(50 / a.fps)))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}",
         "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
         a.out], stdin=subprocess.PIPE)  # fmt: skip
    start_f = int(d["start_frame"][c])
    fell_at = None
    look = None
    for t in range(0, t_end, stride):
        alive = bool(d["alive"][t, c])
        if not alive and fell_at is None:
            fell_at = t
        tt = t if alive else fell_at
        f = int(d["f"][tt, c])
        data.qpos[fa : fa + 3] = d["root_p"][tt, c]
        data.qpos[fa + 3 : fa + 7] = d["root_quat_wxyz"][tt, c]
        data.qpos[qa] = d["q"][tt, c]
        data.qpos[fa_ref : fa_ref + 3] = d["ref_root_p"][f]
        if a.anchored_ghost:
            data.qpos[fa_ref : fa_ref + 2] = d["root_p"][tt, c][:2]
        data.qpos[fa_ref + 3 : fa_ref + 7] = d["ref_root_quat_wxyz"][f]
        data.qpos[qa_ref] = d["ref_q"][f]
        mujoco.mj_kinematics(model, data)
        target = np.array([data.qpos[fa], data.qpos[fa + 1], 0.6])
        look = target if look is None else 0.9 * look + 0.1 * target  # smoothed tracking camera
        cam.lookat[:] = look
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        drift = np.linalg.norm(d["root_p"][tt, c][:2] - d["ref_root_p"][f][:2])
        label = f"{a.title}  t = {(f - start_f) * 0.02:5.2f} s   frame {f}   root xy offset from reference {100 * drift:4.1f} cm"
        if fell_at is not None:
            label += "   [terminated]"
        foot = "solid: simulated T1 policy (Isaac Lab 3 + Newton/MuJoCo-Warp)   blue ghost: reference" + (
            " (anchored to robot xy)" if a.anchored_ghost else " (global position)")  # fmt: skip
        for xy, txt in (((16, 12), label), ((16, a.height - 38), foot)):
            bb = draw.textbbox(xy, txt, font=font)
            draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
            draw.text(xy, txt, fill=(20, 20, 20), font=font)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {a.out}: {len(range(0, t_end, stride))} frames at {a.fps} fps, terminated at step {fell_at}")


if __name__ == "__main__":
    main()
