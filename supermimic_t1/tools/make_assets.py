"""Build the T1 simulation assets from booster_assets' T1_23dof.xml (read-only source).

Outputs (git-ignored, under <repo>/data/t1/):
  T1_23dof_nomesh.xml  the MJCF with the visual-only mesh geoms removed (FK and tests load this;
                       the meshes are contype=0, density=0, so dynamics are unchanged)
  t1_23dof.usda        a UsdPhysics articulation for Isaac Lab 3 / Newton, built from the compiled
                       MuJoCo model: flat link prims at their q=0 pose, revolute joints in a scope,
                       explicit mass/CoM/inertia, primitive collision shapes (boxes, cylinders, spheres)
  ground.usda          a plane collider at z=0 (avoids fetching Isaac Lab's ground USD from S3)

Why a hand-written converter: Isaac Lab 3.0's MjcfConverter needs the isaacsim asset importer,
which is not installed in the kit-less venv, and mujoco-usd-converter needs newer mujoco/usd-exchange
than the venv has. The T1 MJCF uses only primitive collision geoms, so a direct UsdPhysics writer is
small and exact.

The T1_23dof model has a *serial* ankle (ankle_pitch then ankle_roll revolute joints); the physical
parallel linkage is represented only through the actuator limits (booster_train's
BoosterT1AnkleParaWrapperCfg). So the simulated articulation, the FK and the motion CSV all use the
same 23 serial joints.

Run: python supermimic_t1/tools/make_assets.py
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(os.environ.get("BOOSTER_ASSETS_DIR", os.path.expanduser("~/booster_assets")), "robots/T1/T1_23dof.xml")
OUT = os.path.join(REPO, "data", "t1")


def strip_meshes(src: str, dst: str):
    tree = ET.parse(src)
    root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("type") == "mesh":
                parent.remove(child)
            elif child.tag == "mesh":
                parent.remove(child)
            elif child.tag == "texture" or child.tag == "material":
                parent.remove(child)
            elif child.tag == "geom" and child.get("material"):
                del child.attrib["material"]
    comp = root.find("compiler")
    if comp is not None and "meshdir" in comp.attrib:
        del comp.attrib["meshdir"]
    tree.write(dst)


def quat(q):  # mujoco wxyz -> Gf
    return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def build_usd(xml: str, dst: str):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    # root at the origin with identity rotation, all joints at 0
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    mujoco.mj_kinematics(m, d)
    stage = Usd.Stage.CreateNew(dst)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/T1")
    stage.SetDefaultPrim(root.GetPrim())
    joints_scope = UsdGeom.Scope.Define(stage, "/T1/joints")
    axis_tok = {0: "X", 1: "Y", 2: "Z"}
    for b in range(1, m.nbody):
        name = m.body(b).name
        if m.body_mass[b] <= 0:
            continue  # massless marker bodies (camera link) are not physical
        path = f"/T1/{name}"
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*d.xpos[b].tolist()))
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*mat_to_quat(d.xmat[b]).tolist()))
        prim = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(m.body_mass[b]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*m.body_ipos[b].tolist()))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*m.body_inertia[b].tolist()))
        mass.CreatePrincipalAxesAttr(quat(m.body_iquat[b]))
        if m.body_parentid[b] == 0:
            UsdPhysics.ArticulationRootAPI.Apply(prim)
        # collision geoms
        UsdGeom.Xform.Define(stage, f"{path}/collisions")
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            gp = f"{path}/collisions/geom{g}"
            t, s = m.geom_type[g], m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                geom = UsdGeom.Cube.Define(stage, gp)
                geom.CreateSizeAttr(2.0)
                scale = Gf.Vec3f(*s.tolist())
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom = UsdGeom.Cylinder.Define(stage, gp)
                geom.CreateRadiusAttr(float(s[0]))
                geom.CreateHeightAttr(float(2 * s[1]))
                geom.CreateAxisAttr("Z")
                scale = None
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom = UsdGeom.Sphere.Define(stage, gp)
                geom.CreateRadiusAttr(float(s[0]))
                scale = None
            else:
                raise RuntimeError(f"unsupported collision geom type {t} on {name}")
            geom.AddTranslateOp().Set(Gf.Vec3d(*m.geom_pos[g].tolist()))
            geom.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*m.geom_quat[g].tolist()))
            if scale is not None:
                geom.AddScaleOp().Set(scale)
            geom.CreatePurposeAttr("guide")
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        # joint to parent
        if m.body_parentid[b] == 0:
            continue
        assert m.body_jntnum[b] == 1, name
        j = m.body_jntadr[b]
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(m.jnt_pos[j], 0), name
        ax = m.jnt_axis[j]
        k = int(np.argmax(np.abs(ax)))
        assert np.isclose(abs(ax[k]), 1.0), f"non-cardinal axis on {name}"
        assert ax[k] > 0, f"negative axis on {name}"
        jp = UsdPhysics.RevoluteJoint.Define(stage, f"/T1/joints/{m.joint(j).name}")
        jp.CreateBody0Rel().SetTargets([Sdf.Path(f"/T1/{m.body(m.body_parentid[b]).name}")])
        jp.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        jp.CreateLocalPos0Attr(Gf.Vec3f(*m.body_pos[b].tolist()))
        jp.CreateLocalRot0Attr(quat(m.body_quat[b]))
        jp.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        jp.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        jp.CreateAxisAttr(axis_tok[k])
        lo, hi = np.degrees(m.jnt_range[j])
        jp.CreateLowerLimitAttr(float(lo))
        jp.CreateUpperLimitAttr(float(hi))
        drive = UsdPhysics.DriveAPI.Apply(jp.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    stage.GetRootLayer().Save()
    del joints_scope


def build_ground(dst: str):
    stage = Usd.Stage.CreateNew(dst)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Ground")
    stage.SetDefaultPrim(root.GetPrim())
    plane = UsdGeom.Plane.Define(stage, "/Ground/plane")
    plane.CreateAxisAttr("Z")
    plane.CreateWidthAttr(200.0)
    plane.CreateLengthAttr(200.0)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    xml = os.path.join(OUT, "T1_23dof_nomesh.xml")
    strip_meshes(SRC, xml)
    for f in ("t1_23dof.usda", "ground.usda"):
        if os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))
    build_usd(xml, os.path.join(OUT, "t1_23dof.usda"))
    build_ground(os.path.join(OUT, "ground.usda"))
    m = mujoco.MjModel.from_xml_path(xml)
    print(f"wrote {OUT}: nbody {m.nbody - 1}, njnt {m.njnt - 1} hinge, mass {m.body_mass.sum():.3f} kg")
