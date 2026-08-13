"""
Unit and physical integration tests for the simulator-grounded causal intervention oracle.
"""

from dataclasses import replace
import math
import numpy as np
import pytest
import mujoco

from src.environment.scene_builder import SceneBuilder
from src.environment.scene_utils import (
    get_lid_center,
    get_target_center,
    sample_position_on_lid,
    sample_position_beside_box,
    sample_position_in_target,
    sample_position_outside_target,
)
from src.interventions.intervention_types import (
    Action,
    CandidateObject,
    Intervention,
    InterventionOperator,
    InterventionOutcome,
    ObjectPose,
    PrivilegedCategory,
)
from src.interventions.intervention_generator import (
    InterventionGenerator,
    generate_deterministic_intervention_id,
)
from src.interventions.intervention_validator import (
    InterventionValidator,
    apply_intervention,
    get_body_freejoint_pose,
    restore_simulator_state,
    snapshot_simulator_state,
)
from src.validation.occupancy_checks import (
    RelationalFeasibilityResult,
    check_action_feasibility,
    evaluate_relational_feasibility,
)


# ==============================================================================
# 1. DATATYPES & SERIALIZATION TESTS
# ==============================================================================

def test_object_pose_validation_and_canonicalization():
    """Verify ObjectPose validates finite shapes and canonicalizes quaternion to unit norm."""
    # Valid pose with unnormalized quaternion
    pose = ObjectPose(position=(0.1, 0.2, 0.3), quaternion_wxyz=(2.0, 0.0, 0.0, 0.0))
    assert pose.position == (0.1, 0.2, 0.3)
    assert np.isclose(np.linalg.norm(pose.quaternion_wxyz), 1.0)
    assert pose.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)

    # Serialization round-trip
    d = pose.to_dict()
    restored = ObjectPose.from_dict(d)
    assert restored.position == pose.position
    assert np.allclose(restored.quaternion_wxyz, pose.quaternion_wxyz)

    # Rejection of invalid inputs
    with pytest.raises(ValueError):
        ObjectPose(position=(0.1, 0.2), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ObjectPose(position=(0.1, 0.2, 0.3), quaternion_wxyz=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ObjectPose(position=(float("nan"), 0.2, 0.3), quaternion_wxyz=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ObjectPose(position=(0.1, 0.2, 0.3), quaternion_wxyz=(0.0, 0.0, 0.0, 0.0))


def test_action_serialization():
    """Verify Action dataclass serialization."""
    act = Action(action_type="OPEN", target="box_B1", arguments={"tool": "gripper"})
    d = act.to_dict()
    restored = Action.from_dict(d)
    assert restored == act
    assert restored.action_type == "OPEN"
    assert restored.target == "box_B1"


def test_intervention_invariants():
    """Enforce physical operator invariants for NONE and RELOCATE."""
    pose = ObjectPose(position=(0.1, 0.2, 0.3))
    
    # Valid NONE
    interv_none = Intervention(
        intervention_id="int_001",
        operator=InterventionOperator.NONE,
        intended_category=PrivilegedCategory.IDENTITY,
    )
    assert interv_none.operator == InterventionOperator.NONE

    # Invalid NONE with object or pose
    with pytest.raises(ValueError):
        Intervention(
            intervention_id="int_002",
            operator=InterventionOperator.NONE,
            object_name="coffee_can",
        )

    # Valid RELOCATE
    interv_reloc = Intervention(
        intervention_id="int_003",
        operator=InterventionOperator.RELOCATE,
        object_name="coffee_can",
        destination_pose=pose,
        intended_category=PrivilegedCategory.REPAIR,
    )
    assert interv_reloc.operator == InterventionOperator.RELOCATE

    # Invalid RELOCATE missing destination_pose
    with pytest.raises(ValueError):
        Intervention(
            intervention_id="int_004",
            operator=InterventionOperator.RELOCATE,
            object_name="coffee_can",
        )


def test_outcome_and_result_serialization():
    """Verify serialization round-trips for RelationalFeasibilityResult and InterventionOutcome."""
    res = RelationalFeasibilityResult(
        feasible=False,
        task_id="task_1",
        active_culprits=("blocker1",),
        measurements={"blocker1": {"overlap_ratio": 0.85}},
        settling_succeeded=True,
    )
    d_res = res.to_dict()
    restored_res = RelationalFeasibilityResult.from_dict(d_res)
    assert restored_res == res

    pose = ObjectPose(position=(0.1, 0.2, 0.3))
    interv = Intervention(
        intervention_id="int_test",
        operator=InterventionOperator.RELOCATE,
        object_name="blocker1",
        destination_pose=pose,
        intended_category=PrivilegedCategory.REPAIR,
    )
    outcome = InterventionOutcome(
        intervention=interv,
        pre_feasible=False,
        post_feasible=True,
        causal_effect=1,
        intended_effect_matches=True,
        active_culprits_before=("blocker1",),
        active_culprits_after=(),
        realized_object_pose=pose,
        validation_diagnostics={"test_flag": True},
    )
    d_out = outcome.to_dict()
    restored_out = InterventionOutcome.from_dict(d_out)
    assert restored_out == outcome


# ==============================================================================
# 2. GENERATOR TESTS (DETERMINISM, OPACITY, COMPOSITION)
# ==============================================================================

@pytest.fixture
def settled_task1_scene():
    """Fixture producing a physically settled Task 1 scene with 1 culprit and 1 distractor."""
    sb = SceneBuilder()
    rng = np.random.default_rng(100)
    ref_m, ref_d = sb.create_environment(settle_steps=0, include_robot=False)
    culprit_pos = sample_position_on_lid(ref_m, ref_d, rng, x_frac=0.0, y_frac=0.0, height_above=0.02).tolist()
    distractor_pos = sample_position_beside_box(ref_m, ref_d, rng, offset_x=-0.30, offset_y=-0.15, height_above_table=0.04).tolist()
    
    objects = [
        {"name": "blocker1", "type": "coffee_can", "pos": culprit_pos},
        {"name": "distractor1", "type": "sugar_box", "pos": distractor_pos},
    ]
    model, data = sb.create_environment(objects_to_spawn=objects, settle_steps=200, include_robot=False)
    return model, data


@pytest.fixture
def settled_task2_scene():
    """Fixture producing a physically settled Task 2 scene with 1 occupant and 1 distractor."""
    sb = SceneBuilder()
    rng = np.random.default_rng(200)
    ref_m, ref_d = sb.create_environment(settle_steps=0, include_robot=False)
    occupant_pos = sample_position_in_target(ref_m, ref_d, rng, x_frac=0.0, y_frac=0.0, height_above=0.07).tolist()
    distractor_pos = sample_position_outside_target(ref_m, ref_d, rng, offset_x=0.30, offset_y=0.0, height_above=0.07).tolist()
    
    objects = [
        {"name": "occupant1", "type": "sugar_box", "pos": occupant_pos},
        {"name": "distractor1", "type": "mug", "pos": distractor_pos},
    ]
    model, data = sb.create_environment(objects_to_spawn=objects, settle_steps=200, include_robot=False)
    return model, data


def test_generate_candidates_task1_stop_counts(settled_task1_scene):
    """STOP scene (1 culprit, 1 distractor) must generate exactly 4 candidates."""
    model, data = settled_task1_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)
    
    candidates = gen.generate_candidates(
        scene_id="sc_t1_01",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=["blocker1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )
    
    assert len(candidates) == 4
    categories = {c.intended_category for c in candidates}
    assert categories == {
        PrivilegedCategory.REPAIR,
        PrivilegedCategory.HARD_NEGATIVE,
        PrivilegedCategory.IRRELEVANT,
        PrivilegedCategory.IDENTITY,
    }
    # Check assigned indices
    assert [c.intervention_idx for c in candidates] == [0, 1, 2, 3]


def test_generate_candidates_task1_proceed_counts(settled_task1_scene):
    """PROCEED scene (1 distractor) must generate exactly 3 candidates."""
    model, data = settled_task1_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)
    
    candidates = gen.generate_candidates(
        scene_id="sc_t1_proc",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=[],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=False,
    )
    
    assert len(candidates) == 3
    categories = {c.intended_category for c in candidates}
    assert categories == {
        PrivilegedCategory.HARMFUL,
        PrivilegedCategory.IRRELEVANT,
        PrivilegedCategory.IDENTITY,
    }


def test_generate_candidates_task2_counts(settled_task2_scene):
    """Task 2 candidate generation counts and composition."""
    model, data = settled_task2_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)
    
    # STOP
    stop_cands = gen.generate_candidates(
        scene_id="sc_t2_01",
        task_id="task_2",
        model=model,
        data=data,
        culprit_names=["occupant1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )
    assert len(stop_cands) == 4
    
    # PROCEED
    proc_cands = gen.generate_candidates(
        scene_id="sc_t2_proc",
        task_id="task_2",
        model=model,
        data=data,
        culprit_names=[],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=False,
    )
    assert len(proc_cands) == 3


def test_candidate_ids_opaque_and_unique(settled_task1_scene):
    """Intervention IDs must be unique per scene and completely free of semantic category leakage."""
    model, data = settled_task1_scene
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)
    
    candidates = gen.generate_candidates(
        scene_id="sc_t1_01",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=["blocker1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )
    
    ids = [c.intervention_id for c in candidates]
    assert len(ids) == len(set(ids)), "Intervention IDs must be unique within scene"
    
    for c_id in ids:
        assert c_id.startswith("int_")
        # Ensure no privileged category words leaked into ID
        for leak_word in ["repair", "harmful", "hard_negative", "irrelevant", "culprit", "identity", "correct"]:
            assert leak_word not in c_id.lower()


def test_candidate_determinism_and_diversity(settled_task1_scene):
    """Same RNG seed produces identical geometry; different seeds produce varied geometry."""
    model, data = settled_task1_scene
    gen = InterventionGenerator()
    
    cands_seed1_a = gen.generate_candidates(
        scene_id="sc_t1", task_id="task_1", model=model, data=data,
        culprit_names=["blocker1"], distractor_names=["distractor1"],
        rng=np.random.default_rng(42), is_stop_scene=True
    )
    cands_seed1_b = gen.generate_candidates(
        scene_id="sc_t1", task_id="task_1", model=model, data=data,
        culprit_names=["blocker1"], distractor_names=["distractor1"],
        rng=np.random.default_rng(42), is_stop_scene=True
    )
    cands_seed2 = gen.generate_candidates(
        scene_id="sc_t1", task_id="task_1", model=model, data=data,
        culprit_names=["blocker1"], distractor_names=["distractor1"],
        rng=np.random.default_rng(999), is_stop_scene=True
    )
    
    # Same seed -> bitwise identical positions
    for c1, c2 in zip(cands_seed1_a, cands_seed1_b):
        assert c1.intervention_id == c2.intervention_id
        if c1.destination_pose:
            assert np.allclose(c1.destination_pose.position, c2.destination_pose.position)

    # Different seeds -> physical destinations differ for at least one candidate
    positions_seed1 = [c.destination_pose.position for c in cands_seed1_a if c.destination_pose]
    positions_seed2 = [c.destination_pose.position for c in cands_seed2 if c.destination_pose]
    assert not all(np.allclose(p1, p2) for p1, p2 in zip(positions_seed1, positions_seed2))


# ==============================================================================
# 3. APPLICATION & STATE ISOLATION TESTS
# ==============================================================================

def test_snapshot_restore_integration_state(settled_task1_scene):
    """Snapshotting and restoring integration state recovers identical continuous physics state."""
    model, data = settled_task1_scene
    
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    
    snapshot = snapshot_simulator_state(model, data)
    
    # Mutate state (move blocker1, give velocity, step sim)
    data.qpos[0] += 1.0
    data.qvel[0] = 5.0
    mujoco.mj_step(model, data)
    
    assert not np.allclose(data.qpos, initial_qpos)
    
    # Restore
    restore_simulator_state(model, data, snapshot)
    
    assert np.allclose(data.qpos, initial_qpos)
    assert np.allclose(data.qvel, initial_qvel)


def test_apply_intervention_isolation(settled_task1_scene):
    """RELOCATE modifies only target freejoint qpos/qvel before settling."""
    model, data = settled_task1_scene
    
    # Get initial poses of both objects
    pose_blocker_init = get_body_freejoint_pose(model, data, "blocker1")
    pose_dist_init = get_body_freejoint_pose(model, data, "distractor1")
    
    new_dest = ObjectPose(position=(pose_blocker_init.position[0] - 0.2, pose_blocker_init.position[1], pose_blocker_init.position[2]))
    interv = Intervention(
        intervention_id="int_reloc_test",
        operator=InterventionOperator.RELOCATE,
        object_name="blocker1",
        destination_pose=new_dest,
    )
    
    apply_intervention(model, data, interv)
    
    pose_blocker_post = get_body_freejoint_pose(model, data, "blocker1")
    pose_dist_post = get_body_freejoint_pose(model, data, "distractor1")
    
    # Target object moved to exact requested position
    assert np.allclose(pose_blocker_post.position, new_dest.position)
    # Untargeted distractor remained in exact initial position
    assert np.allclose(pose_dist_post.position, pose_dist_init.position)


def test_apply_intervention_none_is_noop(settled_task1_scene):
    """NONE operator leaves physics state completely untouched."""
    model, data = settled_task1_scene
    qpos_before = data.qpos.copy()
    
    interv = Intervention(intervention_id="int_none", operator=InterventionOperator.NONE)
    apply_intervention(model, data, interv)
    
    assert np.allclose(data.qpos, qpos_before)


def test_apply_intervention_errors(settled_task1_scene):
    """Rejects missing body or invalid configurations."""
    model, data = settled_task1_scene
    pose = ObjectPose(position=(0.0, 0.0, 0.0))
    
    # Missing body
    with pytest.raises(KeyError):
        apply_intervention(model, data, Intervention(
            intervention_id="int_err",
            operator=InterventionOperator.RELOCATE,
            object_name="nonexistent_object",
            destination_pose=pose,
        ))


# ==============================================================================
# 4. AUTHORITATIVE PHYSICAL ORACLE TESTS
# ==============================================================================

def test_task1_stop_physical_oracle(settled_task1_scene):
    """Task 1 STOP scene physical validation across all candidate types (+1, 0)."""
    model, data = settled_task1_scene
    validator = InterventionValidator(settle_steps=300)
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)
    
    candidates = gen.generate_candidates(
        scene_id="sc_t1_stop",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=["blocker1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )
    
    outcomes = validator.validate_scene_interventions(
        model, data, candidates, task_id="task_1", candidate_objects=["blocker1", "distractor1"]
    )
    
    outcomes_by_cat = {o.intervention.intended_category: o for o in outcomes}
    
    # REPAIR -> Delta = +1
    rep_out = outcomes_by_cat[PrivilegedCategory.REPAIR]
    assert rep_out.pre_feasible is False
    assert rep_out.post_feasible is True
    assert rep_out.causal_effect == 1
    assert rep_out.intended_effect_matches is True
    assert "blocker1" in rep_out.active_culprits_before
    assert len(rep_out.active_culprits_after) == 0

    # HARD_NEGATIVE -> Delta = 0 (still on lid)
    hn_out = outcomes_by_cat[PrivilegedCategory.HARD_NEGATIVE]
    assert hn_out.pre_feasible is False
    assert hn_out.post_feasible is False
    assert hn_out.causal_effect == 0
    assert hn_out.intended_effect_matches is True
    assert "blocker1" in hn_out.active_culprits_after

    # IRRELEVANT -> Delta = 0 (distractor moved, blocker still on lid)
    irr_out = outcomes_by_cat[PrivilegedCategory.IRRELEVANT]
    assert irr_out.pre_feasible is False
    assert irr_out.post_feasible is False
    assert irr_out.causal_effect == 0
    assert irr_out.intended_effect_matches is True

    # IDENTITY -> Delta = 0
    id_out = outcomes_by_cat[PrivilegedCategory.IDENTITY]
    assert id_out.pre_feasible is False
    assert id_out.post_feasible is False
    assert id_out.causal_effect == 0
    assert id_out.intended_effect_matches is True


def test_task1_proceed_physical_oracle():
    """Task 1 PROCEED scene physical validation (harmful Delta = -1, controls = 0)."""
    sb = SceneBuilder()
    rng = np.random.default_rng(101)
    ref_m, ref_d = sb.create_environment(settle_steps=0, include_robot=False)
    
    # PROCEED scene: distractor placed beside the box (lid is clear)
    dist_pos = sample_position_beside_box(ref_m, ref_d, rng, offset_x=-0.30, offset_y=-0.15, height_above_table=0.04).tolist()
    objects = [{"name": "distractor1", "type": "coffee_can", "pos": dist_pos}]
    model, data = sb.create_environment(objects_to_spawn=objects, settle_steps=200, include_robot=False)
    
    validator = InterventionValidator(settle_steps=300)
    gen = InterventionGenerator()
    
    candidates = gen.generate_candidates(
        scene_id="sc_t1_proc",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=[],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=False,
    )
    
    outcomes = validator.validate_scene_interventions(
        model, data, candidates, task_id="task_1", candidate_objects=["distractor1"]
    )
    outcomes_by_cat = {o.intervention.intended_category: o for o in outcomes}
    
    # HARMFUL -> Delta = -1 (lid becomes occupied)
    harm_out = outcomes_by_cat[PrivilegedCategory.HARMFUL]
    assert harm_out.pre_feasible is True
    assert harm_out.post_feasible is False
    assert harm_out.causal_effect == -1
    assert harm_out.intended_effect_matches is True
    assert "distractor1" in harm_out.active_culprits_after

    # IRRELEVANT -> Delta = 0 (distractor moved to another safe location)
    irr_out = outcomes_by_cat[PrivilegedCategory.IRRELEVANT]
    assert irr_out.pre_feasible is True
    assert irr_out.post_feasible is True
    assert irr_out.causal_effect == 0
    assert irr_out.intended_effect_matches is True

    # IDENTITY -> Delta = 0
    id_out = outcomes_by_cat[PrivilegedCategory.IDENTITY]
    assert id_out.pre_feasible is True
    assert id_out.post_feasible is True
    assert id_out.causal_effect == 0
    assert id_out.intended_effect_matches is True


def test_task2_stop_physical_oracle(settled_task2_scene):
    """Task 2 STOP scene physical validation across all candidate types (+1, 0)."""
    model, data = settled_task2_scene
    validator = InterventionValidator(settle_steps=300)
    gen = InterventionGenerator()
    rng = np.random.default_rng(201)
    
    candidates = gen.generate_candidates(
        scene_id="sc_t2_stop",
        task_id="task_2",
        model=model,
        data=data,
        culprit_names=["occupant1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )
    
    outcomes = validator.validate_scene_interventions(
        model, data, candidates, task_id="task_2", candidate_objects=["occupant1", "distractor1"]
    )
    outcomes_by_cat = {o.intervention.intended_category: o for o in outcomes}
    
    # REPAIR -> Delta = +1 (target cleared)
    rep_out = outcomes_by_cat[PrivilegedCategory.REPAIR]
    assert rep_out.pre_feasible is False
    assert rep_out.post_feasible is True
    assert rep_out.causal_effect == 1
    assert rep_out.intended_effect_matches is True

    # HARD_NEGATIVE -> Delta = 0 (still in target)
    hn_out = outcomes_by_cat[PrivilegedCategory.HARD_NEGATIVE]
    assert hn_out.pre_feasible is False
    assert hn_out.post_feasible is False
    assert hn_out.causal_effect == 0
    assert hn_out.intended_effect_matches is True

    # IRRELEVANT -> Delta = 0
    irr_out = outcomes_by_cat[PrivilegedCategory.IRRELEVANT]
    assert irr_out.pre_feasible is False
    assert irr_out.post_feasible is False
    assert irr_out.causal_effect == 0
    assert irr_out.intended_effect_matches is True

    # IDENTITY -> Delta = 0
    id_out = outcomes_by_cat[PrivilegedCategory.IDENTITY]
    assert id_out.pre_feasible is False
    assert id_out.post_feasible is False
    assert id_out.causal_effect == 0
    assert id_out.intended_effect_matches is True


def test_task2_proceed_physical_oracle():
    """Task 2 PROCEED scene physical validation (harmful Delta = -1, controls = 0)."""
    sb = SceneBuilder()
    rng = np.random.default_rng(202)
    ref_m, ref_d = sb.create_environment(settle_steps=0, include_robot=False)
    
    # PROCEED scene: distractor placed outside target (target is clear)
    dist_pos = sample_position_outside_target(ref_m, ref_d, rng, offset_x=0.30, offset_y=0.0, height_above=0.07).tolist()
    objects = [{"name": "distractor1", "type": "sugar_box", "pos": dist_pos}]
    model, data = sb.create_environment(objects_to_spawn=objects, settle_steps=200, include_robot=False)
    
    validator = InterventionValidator(settle_steps=300)
    gen = InterventionGenerator()
    
    candidates = gen.generate_candidates(
        scene_id="sc_t2_proc",
        task_id="task_2",
        model=model,
        data=data,
        culprit_names=[],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=False,
    )
    
    outcomes = validator.validate_scene_interventions(
        model, data, candidates, task_id="task_2", candidate_objects=["distractor1"]
    )
    outcomes_by_cat = {o.intervention.intended_category: o for o in outcomes}
    
    # HARMFUL -> Delta = -1 (target becomes occupied)
    harm_out = outcomes_by_cat[PrivilegedCategory.HARMFUL]
    assert harm_out.pre_feasible is True
    assert harm_out.post_feasible is False
    assert harm_out.causal_effect == -1
    assert harm_out.intended_effect_matches is True

    # IRRELEVANT -> Delta = 0
    irr_out = outcomes_by_cat[PrivilegedCategory.IRRELEVANT]
    assert irr_out.pre_feasible is True
    assert irr_out.post_feasible is True
    assert irr_out.causal_effect == 0
    assert irr_out.intended_effect_matches is True

    # IDENTITY -> Delta = 0
    id_out = outcomes_by_cat[PrivilegedCategory.IDENTITY]
    assert id_out.pre_feasible is True
    assert id_out.post_feasible is True
    assert id_out.causal_effect == 0
    assert id_out.intended_effect_matches is True


def test_task2_place_subject_excluded_from_occupants(settled_task2_scene):
    """Verify that the future PLACE manipulation subject is not mistakenly flagged as an obstruction."""
    sb = SceneBuilder()
    rng = np.random.default_rng(203)
    ref_m, ref_d = sb.create_environment(settle_steps=0, include_robot=False)
    
    # Scene with pick object (subject of future PLACE) outside target, and target clear
    pick_obj_pos = sample_position_outside_target(ref_m, ref_d, rng, offset_x=-0.25, offset_y=-0.10, height_above=0.07).tolist()
    objects = [{"name": "pick_can", "type": "coffee_can", "pos": pick_obj_pos}]
    model, data = sb.create_environment(objects_to_spawn=objects, settle_steps=200, include_robot=False)
    
    # Even if pick_can is movable in scene, evaluating with explicit target candidate objects (or checking target) confirms feasibility
    res = evaluate_relational_feasibility(model, data, task_id="task_2", candidate_objects=["pick_can"])
    assert res.feasible is True
    assert len(res.active_culprits) == 0


def test_arbitrary_instance_names():
    """Verify oracle correctly evaluates arbitrary instance names when explicit candidate list is provided."""
    sb = SceneBuilder()
    rng = np.random.default_rng(300)
    ref_m, ref_d = sb.create_environment(settle_steps=0, include_robot=False)
    
    # Custom non-standard instance names
    pos_on_lid = sample_position_on_lid(ref_m, ref_d, rng, x_frac=0.0, y_frac=0.0, height_above=0.02).tolist()
    objects = [{"name": "custom_can_99", "type": "coffee_can", "pos": pos_on_lid}]
    model, data = sb.create_environment(objects_to_spawn=objects, settle_steps=200, include_robot=False)
    
    # Check that passing explicit candidate name 'custom_can_99' flags it as active culprit
    res = evaluate_relational_feasibility(model, data, task_id="task_1", candidate_objects=["custom_can_99"])
    assert res.feasible is False
    assert "custom_can_99" in res.active_culprits


def test_candidate_order_independence(settled_task1_scene):
    """Evaluating candidate set in forward vs reverse order produces strictly identical outcomes."""
    model, data = settled_task1_scene
    validator = InterventionValidator(settle_steps=300)
    gen = InterventionGenerator()
    rng = np.random.default_rng(42)
    
    candidates = gen.generate_candidates(
        scene_id="sc_order",
        task_id="task_1",
        model=model,
        data=data,
        culprit_names=["blocker1"],
        distractor_names=["distractor1"],
        rng=rng,
        is_stop_scene=True,
    )
    
    # Forward evaluation
    outcomes_fwd = validator.validate_scene_interventions(
        model, data, candidates, task_id="task_1", candidate_objects=["blocker1", "distractor1"]
    )
    
    # Reverse evaluation
    candidates_rev = list(reversed(candidates))
    outcomes_rev = validator.validate_scene_interventions(
        model, data, candidates_rev, task_id="task_1", candidate_objects=["blocker1", "distractor1"]
    )
    
    # Index by intervention_id and compare
    fwd_map = {o.intervention.intervention_id: o for o in outcomes_fwd}
    rev_map = {o.intervention.intervention_id: o for o in outcomes_rev}
    
    for c_id in fwd_map:
        o_fwd = fwd_map[c_id]
        o_rev = rev_map[c_id]
        assert o_fwd.pre_feasible == o_rev.pre_feasible
        assert o_fwd.post_feasible == o_rev.post_feasible
        assert o_fwd.causal_effect == o_rev.causal_effect
        assert o_fwd.active_culprits_before == o_rev.active_culprits_before
        assert o_fwd.active_culprits_after == o_rev.active_culprits_after


def test_category_mismatch_audit(settled_task1_scene):
    """When an intended category expectation does not match physical reality, observed Delta is preserved and mismatch is flagged."""
    model, data = settled_task1_scene
    validator = InterventionValidator(settle_steps=300)
    
    # Construct an intervention labeled as REPAIR, but moving the culprit to a STILL OBSTRUCTING position
    pose_still_on_lid = ObjectPose(position=(0.52, 0.18, 0.85))  # squarely on lid
    mismatched_interv = Intervention(
        intervention_id="int_mismatch_test",
        operator=InterventionOperator.RELOCATE,
        object_name="blocker1",
        destination_pose=pose_still_on_lid,
        intended_category=PrivilegedCategory.REPAIR,  # Intended repair!
    )
    
    outcome = validator.validate_single_intervention(
        model, data, mismatched_interv, task_id="task_1", candidate_objects=["blocker1", "distractor1"]
    )
    
    # Pre: False, Post: False -> Observed Delta = 0
    assert outcome.pre_feasible is False
    assert outcome.post_feasible is False
    assert outcome.causal_effect == 0  # Physical truth is 0!
    assert outcome.intended_effect_matches is False  # Audit flags the mismatch!
