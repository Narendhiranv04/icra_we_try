"""
Unit tests for intervention rendering, live post-state callback capture, generator hardening,
and canonical state invariants.
"""

from pathlib import Path
import numpy as np
import pytest
import PIL.Image as Image
import mujoco

from src.environment.scene_builder import SceneBuilder
from src.environment.renderer import OffscreenRenderer
from src.environment.observation_rig import (
    get_task_observation_rig,
    apply_observation_rig,
)
from src.environment.scene_utils import (
    sample_position_on_lid,
    sample_position_beside_box,
    sample_position_in_target,
    sample_position_outside_target,
)
from src.interventions.intervention_types import (
    Action,
    Intervention,
    InterventionOperator,
    InterventionOutcome,
    ObjectPose,
    PrivilegedCategory,
)
from src.interventions.intervention_generator import InterventionGenerator
from src.interventions.intervention_validator import (
    InterventionValidator,
    get_body_freejoint_pose,
    snapshot_simulator_state,
    restore_simulator_state,
)
from src.interventions.intervention_scene_generator import (
    _crop_object_from_rgb,
    _colorize_segmentation,
)
from src.validation.occupancy_checks import evaluate_relational_feasibility


@pytest.fixture
def base_settled_task1_scene():
    builder = SceneBuilder()
    model, data = builder.create_environment(
        objects_to_spawn=[
            {"name": "blocker1", "type": "coffee_can", "pos": [0.52, 0.18, 0.76], "quat": [1.0, 0.0, 0.0, 0.0]},
            {"name": "distractor1", "type": "sugar_box", "pos": [0.22, 0.05, 0.44], "quat": [1.0, 0.0, 0.0, 0.0]},
        ],
        settle_steps=0,
        include_robot=True,
        robot_base_pose="home",
    )
    rig = get_task_observation_rig("task_1")
    apply_observation_rig(model, data, rig)
    evaluate_relational_feasibility(model, data, "task_1", candidate_objects=["blocker1", "distractor1"], settle_steps=300, hold_observation_robot=True)
    return model, data, rig


@pytest.fixture
def base_settled_task2_scene():
    builder = SceneBuilder()
    rng = np.random.default_rng(200)
    ref_m, ref_d = builder.create_environment(settle_steps=0, include_robot=False)
    occupant_pos = sample_position_in_target(ref_m, ref_d, rng, x_frac=0.0, y_frac=0.0, height_above=0.07).tolist()
    distractor_pos = sample_position_outside_target(ref_m, ref_d, rng, offset_x=0.30, offset_y=0.0, height_above=0.07).tolist()
    pick_pos = sample_position_outside_target(ref_m, ref_d, rng, offset_x=-0.25, offset_y=-0.10, height_above=0.07).tolist()

    model, data = builder.create_environment(
        objects_to_spawn=[
            {"name": "coffee_can", "type": "coffee_can", "pos": pick_pos, "quat": [1.0, 0.0, 0.0, 0.0]},
            {"name": "occupant1", "type": "sugar_box", "pos": occupant_pos, "quat": [1.0, 0.0, 0.0, 0.0]},
            {"name": "distractor1", "type": "mug", "pos": distractor_pos, "quat": [1.0, 0.0, 0.0, 0.0]},
        ],
        settle_steps=0,
        include_robot=True,
        robot_base_pose="home",
    )
    rig = get_task_observation_rig("task_2")
    apply_observation_rig(model, data, rig)
    evaluate_relational_feasibility(model, data, "task_2", candidate_objects=["occupant1", "distractor1"], settle_steps=300, hold_observation_robot=True)
    return model, data, rig


def test_generator_loud_failure_on_missing_body(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)

    with pytest.raises(KeyError, match="not found in MuJoCo model"):
        gen.generate_candidates(
            scene_id="sc_test",
            task_id="task_1",
            model=model,
            data=data,
            culprit_names=["nonexistent_body_xyz"],
            distractor_names=["distractor1"],
            rng=rng,
            is_stop_scene=True,
        )


def test_generator_strict_culprit_counts_stop(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)

    # 0 culprits on STOP must raise ValueError
    with pytest.raises(ValueError, match="STOP scene requires exactly 1 culprit"):
        gen.generate_candidates(
            scene_id="sc_test",
            task_id="task_1",
            model=model,
            data=data,
            culprit_names=[],
            distractor_names=["distractor1"],
            rng=rng,
            is_stop_scene=True,
        )

    # 2 culprits on STOP must raise ValueError
    with pytest.raises(ValueError, match="STOP scene requires exactly 1 culprit"):
        gen.generate_candidates(
            scene_id="sc_test",
            task_id="task_1",
            model=model,
            data=data,
            culprit_names=["blocker1", "distractor1"],
            distractor_names=[],
            rng=rng,
            is_stop_scene=True,
        )


def test_generator_strict_culprit_counts_proceed(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)

    # Non-empty culprit on PROCEED must raise ValueError
    with pytest.raises(ValueError, match="PROCEED scene requires exactly 0 culprits"):
        gen.generate_candidates(
            scene_id="sc_test",
            task_id="task_1",
            model=model,
            data=data,
            culprit_names=["blocker1"],
            distractor_names=["distractor1"],
            rng=rng,
            is_stop_scene=False,
        )


def test_generator_relocate_minimum_displacement(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    min_dist = 0.04
    gen = InterventionGenerator(min_relocation_m=min_dist)
    rng = np.random.default_rng(42)

    candidates = gen.generate_candidates(
        scene_id="sc_dist_test",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=["blocker1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )

    init_poses = {
        "blocker1": np.array(get_body_freejoint_pose(model, data, "blocker1").position),
        "distractor1": np.array(get_body_freejoint_pose(model, data, "distractor1").position),
    }

    for c in candidates:
        if c.operator == InterventionOperator.RELOCATE:
            dest_p = np.array(c.destination_pose.position)
            curr_p = init_poses[c.object_name]
            disp = np.linalg.norm(dest_p - curr_p)
            assert disp >= min_dist, f"Candidate {c.intervention_id} displacement {disp:.4f} < {min_dist}"


def test_canonical_pre_state_ordering(base_settled_task1_scene):
    model, data, rig = base_settled_task1_scene

    # Robot joint qpos should be preserved in home posture
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_pan_joint")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:head_tilt_joint")

    pan_qpos = data.qpos[model.jnt_qposadr[pan_id]]
    tilt_qpos = data.qpos[model.jnt_qposadr[tilt_id]]

    assert abs(pan_qpos - rig.head_pan) < 1e-3
    assert abs(tilt_qpos - rig.head_tilt) < 1e-3


def test_evaluate_candidate_from_canonical_state_live_post_callback(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    validator = InterventionValidator(settle_steps=300)
    canonical_snap = snapshot_simulator_state(model, data)
    pre_res = evaluate_relational_feasibility(model, data, "task_1", candidate_objects=["blocker1"], settle_steps=300)

    # RELOCATE blocker1 to safe area
    rep_pose = ObjectPose(position=(0.22, 0.05, 0.44), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    interv = Intervention(
        intervention_id="int_live_cb_test",
        operator=InterventionOperator.RELOCATE,
        object_name="blocker1",
        destination_pose=rep_pose,
        intended_category=PrivilegedCategory.REPAIR,
    )

    callback_invoked = False
    callback_object_pos = None

    def _capture_cb(post_model, post_data, outcome):
        nonlocal callback_invoked, callback_object_pos
        callback_invoked = True
        # Read live position inside callback
        p = get_body_freejoint_pose(post_model, post_data, "blocker1").position
        callback_object_pos = p
        assert outcome.post_feasible is True

    outcome = validator.evaluate_candidate_from_canonical_state(
        model=model,
        data=data,
        canonical_snapshot=canonical_snap,
        pre_result=pre_res,
        intervention=interv,
        task_id="task_1",
        candidate_objects=["blocker1"],
        on_post_settled_callback=_capture_cb,
    )

    assert callback_invoked is True
    assert outcome.causal_effect == 1
    # Check that callback saw object in relocated area (z ~ 0.44 on table, not z ~ 0.76 on lid)
    assert callback_object_pos[2] < 0.60
    # Check that after evaluate_candidate_from_canonical_state returns, simulator state is restored to lid (z ~ 0.76)
    restored_p = get_body_freejoint_pose(model, data, "blocker1").position
    assert restored_p[2] > 0.70


def test_evaluate_candidate_from_canonical_state_restores_on_callback_exception(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    validator = InterventionValidator(settle_steps=300)
    canonical_snap = snapshot_simulator_state(model, data)
    pre_res = evaluate_relational_feasibility(model, data, "task_1", candidate_objects=["blocker1"], settle_steps=300)

    rep_pose = ObjectPose(position=(0.22, 0.05, 0.44), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    interv = Intervention(
        intervention_id="int_err_test",
        operator=InterventionOperator.RELOCATE,
        object_name="blocker1",
        destination_pose=rep_pose,
    )

    def _failing_cb(post_model, post_data, outcome):
        raise RuntimeError("Simulated render exception during post-capture")

    with pytest.raises(RuntimeError, match="Simulated render exception"):
        validator.evaluate_candidate_from_canonical_state(
            model=model,
            data=data,
            canonical_snapshot=canonical_snap,
            pre_result=pre_res,
            intervention=interv,
            task_id="task_1",
            candidate_objects=["blocker1"],
            on_post_settled_callback=_failing_cb,
        )

    # Canonical baseline MUST be restored despite exception
    restored_p = get_body_freejoint_pose(model, data, "blocker1").position
    assert restored_p[2] > 0.70


def test_task2_action_subject_excluded_from_obstruction_candidates(base_settled_task2_scene):
    model, data, _ = base_settled_task2_scene
    # In Task 2: coffee_can is at pick position, occupant1 is in target
    # Passing only ['occupant1', 'distractor1'] to evaluate_relational_feasibility
    res = evaluate_relational_feasibility(model, data, "task_2", candidate_objects=["occupant1", "distractor1"], hold_observation_robot=True)
    assert res.feasible is False
    assert res.active_culprits == ("occupant1",)
    assert "coffee_can" not in res.active_culprits


def test_pre_state_object_crops_extracted_from_pre_only(base_settled_task1_scene):
    model, data, rig = base_settled_task1_scene
    renderer = OffscreenRenderer(model, width=640, height=480, camera_name=rig.camera_name)
    pre_rgb = renderer.render_rgb(data)
    mask = renderer.render_culprit_mask(data, ["blocker1_visual", "blocker1_geom"])
    renderer.close()

    crop = _crop_object_from_rgb(pre_rgb, mask)
    assert crop is not None
    assert crop.shape[0] > 16
    assert crop.shape[1] > 16
    assert crop.shape[2] == 3


def test_none_control_null_candidate_fields_and_zero_delta(base_settled_task1_scene):
    model, data, _ = base_settled_task1_scene
    validator = InterventionValidator(hold_observation_robot=True)
    canonical_snap = snapshot_simulator_state(model, data)
    pre_res = evaluate_relational_feasibility(model, data, "task_1", candidate_objects=["blocker1"], settle_steps=300, hold_observation_robot=True)

    none_interv = Intervention(
        intervention_id="int_none_test",
        operator=InterventionOperator.NONE,
        intended_category=PrivilegedCategory.IDENTITY,
    )

    outcome = validator.evaluate_candidate_from_canonical_state(
        model=model,
        data=data,
        canonical_snapshot=canonical_snap,
        pre_result=pre_res,
        intervention=none_interv,
        task_id="task_1",
        candidate_objects=["blocker1"],
    )

    assert outcome.causal_effect == 0
    assert outcome.pre_feasible == outcome.post_feasible
    assert outcome.intended_effect_matches is True


def test_settle_hold_robot_is_opt_in():
    builder = SceneBuilder()
    model, data = builder.create_environment(
        objects_to_spawn=[{"name": "blocker1", "type": "coffee_can", "pos": [0.52, 0.18, 0.76], "quat": [1.0, 0.0, 0.0, 0.0]}],
        settle_steps=0,
        include_robot=True,
        robot_base_pose="home",
    )
    rig = get_task_observation_rig("task_1")
    apply_observation_rig(model, data, rig)

    # Robot indices
    from src.validation.occupancy_checks import _collect_robot_qpos_dof_indices, settle_until_stable
    qpos_indices, dof_indices = _collect_robot_qpos_dof_indices(model)
    init_qpos = data.qpos[qpos_indices].copy()

    # Opt-in False: robot joint values may drift
    settle_until_stable(model, data, body_names=["blocker1"], max_steps=100, hold_observation_robot=False)
    
    # Opt-in True: robot joint values MUST match target exactly
    apply_observation_rig(model, data, rig)
    target_qpos = data.qpos[qpos_indices].copy()
    settle_until_stable(model, data, body_names=["blocker1"], max_steps=100, hold_observation_robot=True)
    post_held_qpos = data.qpos[qpos_indices].copy()
    np.testing.assert_allclose(post_held_qpos, target_qpos, atol=1e-12)
    np.testing.assert_allclose(data.qvel[dof_indices], 0.0, atol=1e-12)


def test_none_control_production_pipeline_stability(base_settled_task1_scene):
    model, data, rig = base_settled_task1_scene
    validator = InterventionValidator(hold_observation_robot=True)
    canonical_snap = snapshot_simulator_state(model, data)
    pre_res = evaluate_relational_feasibility(
        model, data, "task_1", candidate_objects=["blocker1", "distractor1"],
        settle_steps=300, hold_observation_robot=True
    )

    renderer = OffscreenRenderer(model, width=640, height=480, camera_name=rig.camera_name)
    pre_rgb = renderer.render_rgb(data)
    renderer.close()

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, rig.camera_name)
    cam_pos_pre = data.cam_xpos[cam_id].copy()
    cam_mat_pre = data.cam_xmat[cam_id].reshape(3, 3).copy()

    post_rgb_capture = None
    cam_pos_post = None
    cam_mat_post = None

    def _cb(p_model, p_data, outcome):
        nonlocal post_rgb_capture, cam_pos_post, cam_mat_post
        r = OffscreenRenderer(p_model, width=640, height=480, camera_name=rig.camera_name)
        post_rgb_capture = r.render_rgb(p_data)
        r.close()
        cam_pos_post = p_data.cam_xpos[cam_id].copy()
        cam_mat_post = p_data.cam_xmat[cam_id].reshape(3, 3).copy()

    outcome = validator.evaluate_candidate_from_canonical_state(
        model=model,
        data=data,
        canonical_snapshot=canonical_snap,
        pre_result=pre_res,
        intervention=Intervention(intervention_id="int_none_prod", operator=InterventionOperator.NONE),
        task_id="task_1",
        candidate_objects=["blocker1", "distractor1"],
        on_post_settled_callback=_cb,
    )

    assert post_rgb_capture is not None
    rmse = float(np.sqrt(np.mean((pre_rgb.astype(np.float32) - post_rgb_capture.astype(np.float32)) ** 2)))
    cam_trans_drift = float(np.linalg.norm(cam_pos_post - cam_pos_pre))
    R_delta = cam_mat_pre.T @ cam_mat_post
    cos_theta = np.clip((float(np.trace(R_delta)) - 1.0) / 2.0, -1.0, 1.0)
    cam_rot_drift = float(np.arccos(cos_theta))

    assert outcome.causal_effect == 0
    assert rmse <= 3.0, f"NONE control RMSE {rmse:.4f} > 3.0"
    assert cam_trans_drift <= 0.001, f"Camera translation drift {cam_trans_drift:.6f}m > 0.001m"
    assert cam_rot_drift <= 0.005, f"Camera rotation drift {cam_rot_drift:.6f}rad > 0.005rad"


def test_task2_action_subject_unified_identifier():
    from src.interventions.intervention_records import InterventionSceneSpec, Action
    from src.interventions.intervention_scene_generator import InterventionSceneGenerator
    spec = InterventionSceneSpec(
        scene_id="sc_t2_test",
        task_id="task_2",
        intended_base_state="STOP",
        instruction="Place the coffee can in the target region.",
        action=Action(action_type="PLACE", target="target_region", arguments={"object": "coffee_can"}),
        action_subject_name="coffee_can",
        intended_culprit_type="sugar_box",
        intended_distractor_types=("mug",),
        seed=201,
    )
    assert spec.action_subject_name == "coffee_can"
    assert spec.action.arguments["object"] == "coffee_can"


def test_resolved_spec_distinguishes_spawn_and_canonical_pose(tmp_path):
    from src.interventions.intervention_records import InterventionSceneSpec, Action, SCHEMA_VERSION
    from src.interventions.intervention_scene_generator import InterventionSceneGenerator
    spec = InterventionSceneSpec(
        scene_id="sc_pose_test",
        task_id="task_1",
        intended_base_state="STOP",
        instruction="Open the box.",
        action=Action(action_type="OPEN", target="box_B1", arguments={}),
        action_subject_name=None,
        intended_culprit_type="coffee_can",
        intended_distractor_types=("sugar_box",),
        seed=101,
    )
    generator = InterventionSceneGenerator()
    records, resolved_spec = generator.generate_scene_dataset(spec=spec, dataset_root=tmp_path)
    assert resolved_spec.canonical_object_poses is not None
    assert "blocker1" in resolved_spec.canonical_object_poses
    assert "position" in resolved_spec.canonical_object_poses["blocker1"]
    assert "quaternion_wxyz" in resolved_spec.canonical_object_poses["blocker1"]
    assert SCHEMA_VERSION == "2.1.0"


def test_camera_rotation_distance_calculation():
    # Identity
    R = np.eye(3)
    cos_theta = np.clip((np.trace(R.T @ R) - 1.0) / 2.0, -1.0, 1.0)
    assert np.arccos(cos_theta) == 0.0

    # 90 deg rotation around Z
    R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    cos_theta = np.clip((np.trace(R.T @ R_z90) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    np.testing.assert_allclose(theta, np.pi / 2.0, atol=1e-6)
