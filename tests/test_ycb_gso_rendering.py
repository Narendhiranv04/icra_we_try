"""
Mandatory unit tests for YCB/GSO 3D realistic textured rendering and asset pipeline.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
import mujoco

from src.assets.asset_registry import AssetRegistry
from src.assets.downloader import ensure_asset_available
from src.environment.scene_builder import SceneBuilder, get_instance_visual_geom_ids, get_instance_collision_geom_ids
from src.environment.renderer import OffscreenRenderer
from src.generation.counterfactual_generator import CounterfactualPairGenerator


def test_10_ycb_reference_asset_loads_and_renders():
    asset_path = Path(ensure_asset_available("ycb_coffee_can"))
    assert asset_path.exists()
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "coffee_can", "type": "coffee_can", "pos": [0.50, 0.0, 0.75]}], include_robot=True)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    rgb = ren.render_rgb(d)
    assert rgb.shape == (480, 640, 3)
    ren.close()


def test_11_gso_reference_asset_loads_and_renders():
    asset_path = Path(ensure_asset_available("gso_coffee_jar"))
    assert asset_path.exists()
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "gso_jar", "type": "gso_coffee_jar", "pos": [0.50, 0.0, 0.75]}], include_robot=True)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    rgb = ren.render_rgb(d)
    assert rgb.shape == (480, 640, 3)
    ren.close()


def test_12_mesh_files_non_empty():
    obj_files = list(Path("assets/objects/meshes").glob("**/*.obj"))
    assert len(obj_files) > 0
    for obj_f in obj_files:
        assert obj_f.stat().st_size > 0


def test_13_texture_files_decode_correctly():
    png_files = list(Path("assets").glob("**/*.png"))
    assert len(png_files) > 0
    for png_f in png_files:
        img = cv2.imread(str(png_f))
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0


def test_14_collision_proxy_settles_stably():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "coffee_can", "type": "coffee_can", "pos": [0.50, 0.0, 0.90]}], settle_steps=100)
    # Ensure object did not fall through table or explode
    z_pos = d.qpos[model_jnt_qposadr(m, "coffee_can_joint") + 2]
    assert 0.70 <= z_pos <= 0.85


def model_jnt_qposadr(m, jname):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return m.jnt_qposadr[jid]


def test_15_collision_proxy_invisible_in_rgb_seg():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "coffee_can", "type": "coffee_can", "pos": [0.50, 0.0, 0.75]}], include_robot=True)
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    seg = ren.render_segmentation(d)[:, :, 0]
    col_ids = get_instance_collision_geom_ids(m, "coffee_can")
    for cid in col_ids:
        assert cid not in seg
    ren.close()


def test_16_scene_builder_uses_registry_assets():
    sb = SceneBuilder()
    reg = AssetRegistry()
    assert reg.get_asset("ycb_coffee_can") is not None
    m, d = sb.create_environment([{"name": "c1", "type": "coffee_can", "pos": [0.5, 0.0, 0.75]}])
    vis_ids = get_instance_visual_geom_ids(m, "c1")
    assert len(vis_ids) > 0


def test_17_missing_registered_asset_fails():
    sb = SceneBuilder()
    with pytest.raises(ValueError):
        sb.create_environment([{"name": "non_existent", "type": "completely_invalid_type_xyz"}])


def test_18_multigeom_candidate_mask_covers_full_object():
    sb = SceneBuilder()
    m, d = sb.create_environment([{"name": "coffee_can", "type": "coffee_can", "pos": [0.50, 0.0, 0.75]}], include_robot=True)
    vis_ids = get_instance_visual_geom_ids(m, "coffee_can")
    ren = OffscreenRenderer(m, camera_name="robot0:ego_camera")
    seg = ren.render_segmentation(d)[:, :, 0]
    mask = np.isin(seg, vis_ids)
    assert np.count_nonzero(mask) > 50
    ren.close()


def test_19_exact_asset_id_and_texture_regenerate():
    reg = AssetRegistry()
    asset1 = reg.get_asset("ycb_coffee_can")
    asset2 = reg.get_asset("ycb_coffee_can")
    assert asset1.name == asset2.name
    assert asset1.texture_path == asset2.texture_path


def test_20_task_pair_with_ycb_asset_passes(tmp_path):
    gen = CounterfactualPairGenerator(output_dir=tmp_path)
    pair_meta = gen.generate_task1_pair("pair_ycb_test", blocker_type="coffee_can")
    assert pair_meta["pair_id"] == "pair_ycb_test"
    assert Path(tmp_path / "pair_ycb_test" / "stop_rgb.png").is_file()


def test_21_task_pair_with_gso_asset_passes(tmp_path):
    gen = CounterfactualPairGenerator(output_dir=tmp_path)
    pair_meta = gen.generate_task2_pair("pair_gso_test", target_occupant_type="gso_coffee_jar")
    assert pair_meta["pair_id"] == "pair_gso_test"
    assert Path(tmp_path / "pair_gso_test" / "stop_rgb.png").is_file()
