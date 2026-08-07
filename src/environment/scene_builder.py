"""
Scene builder for constructing dynamic MuJoCo environments for benchmark tasks.
Integrates AssetRegistry to render authentic 3D YCB & GSO textured meshes with hidden collision proxies.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mujoco

from src.environment.model_loading import load_model_from_string, load_assets_from_dir
from src.assets.asset_registry import AssetRegistry, AssetMetadata


# Standard mapping from simple string identifiers to AssetRegistry keys
ASSET_KEY_MAPPING = {
    "coffee_can": "ycb_coffee_can",
    "can": "ycb_coffee_can",
    "ycb_coffee_can": "ycb_coffee_can",
    "sugar_box": "ycb_sugar_box",
    "box_obj": "ycb_sugar_box",
    "ycb_sugar_box": "ycb_sugar_box",
    "tea_box": "ycb_tea_box",
    "ycb_tea_box": "ycb_tea_box",
    "mug": "ycb_mug",
    "ycb_mug": "ycb_mug",
    "cup": "ycb_cup",
    "ycb_cup": "ycb_cup",
    "bowl": "ycb_bowl",
    "ycb_bowl": "ycb_bowl",
    "plate": "ycb_plate",
    "ycb_plate": "ycb_plate",
    "fork": "ycb_fork",
    "ycb_fork": "ycb_fork",
    "spoon": "ycb_spoon",
    "ycb_spoon": "ycb_spoon",
    "knife": "ycb_knife",
    "ycb_knife": "ycb_knife",
    "marker": "ycb_marker",
    "ycb_marker": "ycb_marker",
    "gso_kettle": "gso_kettle",
    "gso_coffee_jar": "gso_coffee_jar",
    "gso_sugar_jar": "gso_sugar_jar",
    "gso_canister": "gso_canister_distractor",
    "gso_canister_distractor": "gso_canister_distractor",
    "gso_spatula": "gso_spatula_distractor",
    "gso_spatula_distractor": "gso_spatula_distractor",
}


class SceneBuilder:
    """Builder class for dynamically assembling kitchen workstation scenes with box, target region,
    and textured YCB/GSO 3D assets."""

    def __init__(
        self,
        base_xml_path: Union[str, Path] = "assets/kitchen_base.xml",
        assets_dir: Union[str, Path] = "assets",
        asset_config_path: Union[str, Path] = "configs/assets_ycb_gso.yaml",
        object_library_path: Union[str, Path] = "assets/object_library.xml",
    ):
        self.base_xml_path = Path(base_xml_path)
        self.assets_dir = Path(assets_dir)
        self.object_library_path = Path(object_library_path)
        self.registry = AssetRegistry(asset_config_path)

    def _inject_library_assets(self, root: ET.Element) -> None:
        """Inject `<asset>` elements from object_library.xml into root MJCF tree."""
        if not self.object_library_path.exists():
            return

        lib_tree = ET.parse(self.object_library_path)
        lib_asset = lib_tree.getroot().find("asset")
        if lib_asset is None:
            return

        root_asset = root.find("asset")
        if root_asset is None:
            root_asset = ET.SubElement(root, "asset")

        # Copy missing mesh/texture/material tags
        existing_names = {elem.get("name") for elem in root_asset if elem.get("name")}
        for elem in lib_asset:
            name = elem.get("name")
            if name and name not in existing_names:
                root_asset.append(elem)

    def build_scene_xml(
        self,
        objects_to_spawn: Optional[List[Dict[str, Union[str, List[float]]]]] = None,
        include_robot: bool = False,
        robot_base_pose: str = "home",
        weld_target_body: Optional[str] = None,
        box_pose: Optional[List[float]] = None,
        box_quat: Optional[List[float]] = None,
        target_region_pos: Optional[List[float]] = None,
        target_region_quat: Optional[List[float]] = None,
    ) -> str:
        """Construct scene XML string by reading base XML and injecting YCB/GSO assets and robot."""
        if not self.base_xml_path.exists():
            raise FileNotFoundError(f"Base XML file not found at {self.base_xml_path}")

        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Invalid MJCF XML: <worldbody> tag missing.")

        # Inject library assets (meshes/textures/materials)
        self._inject_library_assets(root)

        # Apply box_B1 overrides if provided
        if box_pose or box_quat:
            box_elem = worldbody.find(".//*[@name='box_B1']")
            if box_elem is not None:
                if box_pose:
                    box_elem.set("pos", f"{box_pose[0]} {box_pose[1]} {box_pose[2]}")
                if box_quat:
                    box_elem.set("quat", f"{box_quat[0]} {box_quat[1]} {box_quat[2]} {box_quat[3]}")

        # Apply target_region_geom overrides if provided
        if target_region_pos or target_region_quat:
            t_elem = worldbody.find(".//*[@name='target_region_geom']")
            if t_elem is not None:
                if target_region_pos:
                    t_elem.set("pos", f"{target_region_pos[0]} {target_region_pos[1]} {target_region_pos[2]}")
                if target_region_quat:
                    t_elem.set("quat", f"{target_region_quat[0]} {target_region_quat[1]} {target_region_quat[2]} {target_region_quat[3]}")

        if objects_to_spawn:
            for obj in objects_to_spawn:
                obj_name = obj["name"]
                raw_type = obj.get("type", "coffee_can")
                asset_key = obj.get("asset_id", ASSET_KEY_MAPPING.get(raw_type, raw_type))

                asset_meta = self.registry.get_asset(asset_key)

                pos = obj.get("pos", [0.0, 0.0, 0.9])
                quat = obj.get("quat", [1.0, 0.0, 0.0, 0.0])
                pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

                body = ET.SubElement(worldbody, "body", name=obj_name, pos=pos_str, quat=quat_str)
                ET.SubElement(body, "freejoint", name=f"{obj_name}_joint")

                if asset_meta is not None:
                    # Multi-geom structure: Visible mesh geom + Hidden collision proxy geom
                    mesh_name = f"mesh_{asset_key}"
                    mat_name = f"mat_{asset_key}"

                    # 1. Visible mesh geom (group=1 for rendering)
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{obj_name}_visual",
                        type="mesh",
                        mesh=mesh_name,
                        material=mat_name,
                        group="1",
                        mass="0.001",
                        contype="0",
                        conaffinity="0",
                    )

                    # 2. Hidden collision proxy geom (group=3, invisible)
                    col_type = asset_meta.collision_type
                    col_params = asset_meta.collision_params
                    fric_str = " ".join(str(f) for f in asset_meta.default_friction)

                    col_attribs = {
                        "name": f"{obj_name}_collision",
                        "group": "3",
                        "rgba": "0 0 0 0",
                        "mass": str(asset_meta.mass_kg),
                        "friction": fric_str,
                    }

                    if col_type == "box":
                        extents = col_params.get("half_extents", [0.04, 0.04, 0.06])
                        col_attribs.update({"type": "box", "size": f"{extents[0]} {extents[1]} {extents[2]}"})
                    elif col_type == "cylinder":
                        r = col_params.get("radius", 0.04)
                        h = col_params.get("half_height", 0.06)
                        col_attribs.update({"type": "cylinder", "size": f"{r} {h}"})
                    elif col_type == "capsule":
                        r = col_params.get("radius", 0.02)
                        h = col_params.get("half_length", 0.05)
                        col_attribs.update({"type": "capsule", "size": f"{r} {h}"})
                    else:  # Fallback compound / hollow cylinder approximation
                        r = col_params.get("radius", 0.04)
                        h = col_params.get("height", 0.10) / 2.0
                        col_attribs.update({"type": "cylinder", "size": f"{r} {h}"})

                    ET.SubElement(body, "geom", col_attribs)
                else:
                    # Fail clearly if registered asset is expected but missing
                    raise ValueError(f"Requested asset '{asset_key}' not found in AssetRegistry!")

        if include_robot:
            # Remove fallback virtual camera before injecting robot-attached ego camera
            for parent in root.iter():
                for elem in list(parent):
                    if elem.tag == "camera" and elem.get("name") == "robot0:ego_camera":
                        parent.remove(elem)
            from src.environment.robot_integration import inject_fetch_robot
            inject_fetch_robot(root, base_pose_name=robot_base_pose, spawn_welds=True, weld_target_body=weld_target_body)

        return ET.tostring(root, encoding="unicode")

    def create_environment(
        self,
        objects_to_spawn: Optional[List[Dict[str, Union[str, List[float]]]]] = None,
        settle_steps: int = 200,
        include_robot: bool = False,
        robot_base_pose: str = "home",
        weld_target_body: Optional[str] = None,
        box_pose: Optional[List[float]] = None,
        box_quat: Optional[List[float]] = None,
        target_region_pos: Optional[List[float]] = None,
        target_region_quat: Optional[List[float]] = None,
    ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Create and initialize MuJoCo MjModel and MjData with settling physics."""
        xml_string = self.build_scene_xml(
            objects_to_spawn=objects_to_spawn,
            include_robot=include_robot,
            robot_base_pose=robot_base_pose,
            weld_target_body=weld_target_body,
            box_pose=box_pose,
            box_quat=box_quat,
            target_region_pos=target_region_pos,
            target_region_quat=target_region_quat,
        )
        assets = load_assets_from_dir(self.assets_dir)
        model, data = load_model_from_string(xml_string, assets=assets)

        if include_robot:
            from src.environment.robot_integration import initialize_robot_qpos
            initialize_robot_qpos(model, data)

        if settle_steps > 0:
            hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint")
            hinge_qpos_adr = model.jnt_qposadr[hinge_id] if hinge_id != -1 else -1
            hinge_dof_adr = model.jnt_dofadr[hinge_id] if hinge_id != -1 else -1
            for _ in range(settle_steps):
                if hinge_qpos_adr != -1:
                    data.qpos[hinge_qpos_adr] = 0.0
                    data.qvel[hinge_dof_adr] = 0.0
                mujoco.mj_step(model, data)

        return model, data


def get_instance_visual_geom_names(model: mujoco.MjModel, instance_name: str) -> List[str]:
    """Retrieve all visible geom names associated with an object instance."""
    names = []
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
    if body_id == -1:
        return [f"{instance_name}_visual", f"{instance_name}_geom"]

    for i in range(model.ngeom):
        if model.geom_bodyid[i] == body_id:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            # Only include visual geoms (group != 3)
            if model.geom_group[i] != 3 and gname:
                names.append(gname)
    return names if names else [f"{instance_name}_visual"]


def get_instance_visual_geom_ids(model: mujoco.MjModel, instance_name: str) -> List[int]:
    """Retrieve all visible geom IDs associated with an object instance."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
    if body_id == -1:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{instance_name}_visual")
        if gid == -1:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{instance_name}_geom")
        return [gid] if gid != -1 else []

    ids = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == body_id and model.geom_group[i] != 3:
            ids.append(i)
    return ids


def get_instance_collision_geom_ids(model: mujoco.MjModel, instance_name: str) -> List[int]:
    """Retrieve all collision proxy geom IDs associated with an object instance."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, instance_name)
    if body_id == -1:
        return []

    ids = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == body_id and model.geom_group[i] == 3:
            ids.append(i)
    return ids
