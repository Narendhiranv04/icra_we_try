import mujoco
from src.environment.scene_builder import SceneBuilder
import src.environment.scene_utils as su

builder = SceneBuilder()
model, data = builder.create_environment(box_pose=[0.0, -0.2, 0.58], target_region_pos=[0.0, 0.3, 0.0])

mujoco.mj_forward(model, data)
print(f"Target region geom world pos: {su.get_geom_world_pos(model, data, 'target_region_geom')}")
print(f"Target region body world pos: {su.get_body_world_pos(model, data, 'target_region_body')}")
print(f"Target region site world pos: {su.get_site_world_pos(model, data, 'target_region_site')}")
