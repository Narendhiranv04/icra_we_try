"""
Scene builder for constructing dynamic MuJoCo environments for benchmark tasks.
Supports full scene transform overrides for box and target region geometry.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mujoco

from src.environment.model_loading import load_model_from_string, load_assets_from_dir


class SceneBuilder:
    """Builder class for dynamically assembling kitchen workstation scenes with box, target region,
    and configurable manipulation/blocker objects."""

    def __init__(
        self,
        base_xml_path: Union[str, Path] = "assets/kitchen_base.xml",
        assets_dir: Union[str, Path] = "assets",
    ):
        self.base_xml_path = Path(base_xml_path)
        self.assets_dir = Path(assets_dir)
        self.object_configs: Dict[str, dict] = {}
        
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
        """Construct scene XML string by reading base XML and injecting object bodies and robot.
        
        Args:
            objects_to_spawn: List of object dictionaries.
            include_robot: Whether to inject Fetch manipulator.
            robot_base_pose: Base placement pose ("home" or "right_side").
            weld_target_body: Optional object body name to attach grasp weld equality constraint.
            box_pose: Optional position override for box_B1.
            box_quat: Optional quaternion override for box_B1.
            target_region_pos: Optional position override for target_region_geom.
            target_region_quat: Optional quaternion override for target_region_geom.
            
        Returns:
            Complete MJCF XML string.
        """
        if not self.base_xml_path.exists():
            raise FileNotFoundError(f"Base XML file not found at {self.base_xml_path}")
            
        tree = ET.parse(self.base_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Invalid MJCF XML: <worldbody> tag missing.")

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
                obj_type = obj.get("type", "coffee_can")
                pos = obj.get("pos", [0.0, 0.0, 0.9])
                quat = obj.get("quat", [1.0, 0.0, 0.0, 0.0])

                pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
                quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

                body = ET.SubElement(worldbody, "body", name=obj_name, pos=pos_str, quat=quat_str)
                ET.SubElement(body, "freejoint", name=f"{obj_name}_joint")
                
                # Render geometry based on object type
                if obj_type in ["coffee_can", "can"]:
                    ET.SubElement(
                        body, "geom",
                        name=f"{obj_name}_geom",
                        type="cylinder",
                        size="0.04 0.06",
                        material="mat_metal",
                        density="500",
                    )
                elif obj_type in ["sugar_box", "box_obj"]:
                    ET.SubElement(
                        body, "geom",
                        name=f"{obj_name}_geom",
                        type="box",
                        size="0.04 0.03 0.07",
                        material="mat_plastic_red",
                        density="400",
                    )
                elif obj_type == "mug":
                    ET.SubElement(
                        body, "geom",
                        name=f"{obj_name}_geom",
                        type="cylinder",
                        size="0.04 0.05",
                        material="mat_ceramic_blue",
                        density="450",
                    )
                elif obj_type == "cup":
                    ET.SubElement(
                        body, "geom",
                        name=f"{obj_name}_geom",
                        type="cylinder",
                        size="0.035 0.05",
                        material="mat_ceramic_white",
                        density="400",
                    )
                else: # Default fallback primitive
                    ET.SubElement(
                        body, "geom",
                        name=f"{obj_name}_geom",
                        type="box",
                        size="0.04 0.04 0.04",
                        material="mat_plastic_yellow",
                        density="300",
                    )

        if include_robot:
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

        # Step physics to settle objects on surfaces
        if settle_steps > 0:
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
                
        return model, data
