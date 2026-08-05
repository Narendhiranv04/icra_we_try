"""
Scene builder for constructing dynamic MuJoCo environments for benchmark tasks.
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
    ) -> str:
        """Construct scene XML string by reading base XML and injecting object bodies.
        
        Args:
            objects_to_spawn: List of dicts specifying:
                - 'name': object instance name (e.g. 'obj1')
                - 'type': catalog object type (e.g. 'coffee_can', 'sugar_box', 'mug')
                - 'pos': [x, y, z] initial 3D position
                - 'quat': [w, x, y, z] initial orientation (optional)
                
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

        return ET.tostring(root, encoding="unicode")

    def create_environment(
        self,
        objects_to_spawn: Optional[List[Dict[str, Union[str, List[float]]]]] = None,
        settle_steps: int = 200,
    ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Create and initialize MuJoCo MjModel and MjData with settling physics.
        
        Args:
            objects_to_spawn: List of object dictionaries to inject into scene.
            settle_steps: Number of mj_step simulation calls to allow physics to settle.
            
        Returns:
            Tuple of (MjModel, MjData).
        """
        xml_string = self.build_scene_xml(objects_to_spawn)
        assets = load_assets_from_dir(self.assets_dir)
        model, data = load_model_from_string(xml_string, assets=assets)
        
        # Step physics to settle objects on surfaces
        if settle_steps > 0:
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
                
        return model, data
