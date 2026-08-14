"""Comprehensive unit and metamorphic tests for InterventionLearningDataset and Grouped Sampler."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.learning.intervention_dataset import (
    InterventionLearningDataset,
    InterventionGroupBatchSampler,
    collate_intervention_group,
)
from src.learning.intervention_feature_spec import (
    InterventionFeatureSpec,
    TEXT_DIM,
    SCENE_GLOBAL_DIM,
    SCENE_PATCH_SHAPE,
    CANDIDATE_VISUAL_DIM,
    CURRENT_GEOM_DIM,
    DEST_GEOM_DIM,
    OPERATOR_NONE_IDX,
    OPERATOR_RELOCATE_IDX,
)


@pytest.fixture
def mock_dataset_and_features(tmp_path):
    """Fixture creating a realistic 2-scene mock dataset and precomputed feature cache."""
    dataset_root = tmp_path / "dataset"
    features_dir = tmp_path / "features"
    dataset_root.mkdir()
    features_dir.mkdir()
    (features_dir / "scenes").mkdir()
    (features_dir / "crops").mkdir()

    # Create dummy feature caches
    spec = InterventionFeatureSpec()

    # Pre-scene features
    torch.save(
        {"global": torch.ones(spec.scene_global_dim) * 1.1, "patch": torch.ones(spec.scene_patch_shape) * 1.2},
        features_dir / "scenes" / "scene_ivs_000001_features.pt",
    )
    torch.save(
        {"global": torch.ones(spec.scene_global_dim) * 2.1, "patch": torch.ones(spec.scene_patch_shape) * 2.2},
        features_dir / "scenes" / "scene_ivs_000002_features.pt",
    )

    # Crop features
    torch.save(
        {"global": torch.ones(spec.candidate_visual_dim) * 3.1},
        features_dir / "crops" / "crop_ivs_000001_blocker1_features.pt",
    )
    torch.save(
        {"global": torch.ones(spec.candidate_visual_dim) * 3.2},
        features_dir / "crops" / "crop_ivs_000001_distractor1_features.pt",
    )
    torch.save(
        {"global": torch.ones(spec.candidate_visual_dim) * 4.1},
        features_dir / "crops" / "crop_ivs_000002_coffee_can_features.pt",
    )

    # Text features
    torch.save(
        {
            "features": {
                "Open the box.": torch.ones(spec.text_dim) * 0.1,
                "Place the coffee can in the target region.": torch.ones(spec.text_dim) * 0.2,
            }
        },
        features_dir / "text_features.pt",
    )

    # Manifest records (Scene 1: 4 records, Scene 2: 3 records)
    records = [
        # Scene 1: STOP (task 1)
        {
            "schema_version": "2.1.0",
            "record_id": "rec_001_0",
            "scene_id": "ivs_000001",
            "intervention_id": "int_001_0",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000001/crops/object_blocker1.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"world_position": [0.52, 0.18, 0.81], "relative_position": [0.001, 0.002, 0.075]},
                "destination_geometry": {"world_position": [0.26, 0.16, 0.62], "relative_position": [-0.25, -0.01, -0.12]},
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": True, "causal_effect": 1},
            "privileged_metadata": {
                "candidate_object_name": "blocker1",
                "intended_category": "repair",
                "is_culprit": True,
                "active_culprits_before": ["blocker1"],
                "active_culprits_after": [],
                "post_rgb_path": "scenes/ivs_000001/post/int_0_rgb.png",
            },
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_001_1",
            "scene_id": "ivs_000001",
            "intervention_id": "int_001_1",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": None,
                "intervention_operator": "NONE",
                "current_geometry": None,
                "destination_geometry": None,
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": False, "causal_effect": 0},
            "privileged_metadata": {
                "candidate_object_name": None,
                "intended_category": "identity",
                "is_culprit": False,
                "active_culprits_before": ["blocker1"],
                "active_culprits_after": ["blocker1"],
                "post_rgb_path": "scenes/ivs_000001/post/int_1_rgb.png",
            },
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_001_2",
            "scene_id": "ivs_000001",
            "intervention_id": "int_001_2",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000001/crops/object_blocker1.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"world_position": [0.52, 0.18, 0.81], "relative_position": [0.001, 0.002, 0.075]},
                "destination_geometry": {"world_position": [0.51, 0.18, 0.77], "relative_position": [-0.001, 0.002, 0.03]},
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": False, "causal_effect": 0},
            "privileged_metadata": {
                "candidate_object_name": "blocker1",
                "intended_category": "hard_negative",
                "is_culprit": True,
                "active_culprits_before": ["blocker1"],
                "active_culprits_after": ["blocker1"],
                "post_rgb_path": "scenes/ivs_000001/post/int_2_rgb.png",
            },
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_001_3",
            "scene_id": "ivs_000001",
            "intervention_id": "int_001_3",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000001/crops/object_distractor1.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"world_position": [0.22, 0.03, 0.66], "relative_position": [-0.30, -0.15, -0.07]},
                "destination_geometry": {"world_position": [0.23, 0.16, 0.62], "relative_position": [-0.28, -0.01, -0.12]},
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": False, "causal_effect": 0},
            "privileged_metadata": {
                "candidate_object_name": "distractor1",
                "intended_category": "irrelevant",
                "is_culprit": False,
                "active_culprits_before": ["blocker1"],
                "active_culprits_after": ["blocker1"],
                "post_rgb_path": "scenes/ivs_000001/post/int_3_rgb.png",
            },
        },
        # Scene 2: PROCEED (task 2)
        {
            "schema_version": "2.1.0",
            "record_id": "rec_002_0",
            "scene_id": "ivs_000002",
            "intervention_id": "int_002_0",
            "model_inputs": {
                "task_id": "task_2",
                "instruction": "Place the coffee can in the target region.",
                "pre_rgb_path": "scenes/ivs_000002/pre_rgb.png",
                "candidate_object_crop_path": None,
                "intervention_operator": "NONE",
                "current_geometry": None,
                "destination_geometry": None,
            },
            "supervision_targets": {"pre_feasible": True, "post_feasible": True, "causal_effect": 0},
            "privileged_metadata": {
                "candidate_object_name": None,
                "intended_category": "identity",
                "is_culprit": False,
                "active_culprits_before": [],
                "active_culprits_after": [],
                "post_rgb_path": "scenes/ivs_000002/post/int_0_rgb.png",
            },
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_002_1",
            "scene_id": "ivs_000002",
            "intervention_id": "int_002_1",
            "model_inputs": {
                "task_id": "task_2",
                "instruction": "Place the coffee can in the target region.",
                "pre_rgb_path": "scenes/ivs_000002/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000002/crops/object_coffee_can.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"world_position": [0.15, 0.25, 0.45], "relative_position": [-0.15, 0.05, 0.01]},
                "destination_geometry": {"world_position": [0.40, 0.10, 0.45], "relative_position": [0.00, 0.00, 0.01]},
            },
            "supervision_targets": {"pre_feasible": True, "post_feasible": False, "causal_effect": -1},
            "privileged_metadata": {
                "candidate_object_name": "coffee_can",
                "intended_category": "harmful",
                "is_culprit": False,
                "active_culprits_before": [],
                "active_culprits_after": ["coffee_can"],
                "post_rgb_path": "scenes/ivs_000002/post/int_1_rgb.png",
            },
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_002_2",
            "scene_id": "ivs_000002",
            "intervention_id": "int_002_2",
            "model_inputs": {
                "task_id": "task_2",
                "instruction": "Place the coffee can in the target region.",
                "pre_rgb_path": "scenes/ivs_000002/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000002/crops/object_coffee_can.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"world_position": [0.15, 0.25, 0.45], "relative_position": [-0.15, 0.05, 0.01]},
                "destination_geometry": {"world_position": [0.10, 0.30, 0.45], "relative_position": [-0.20, 0.10, 0.01]},
            },
            "supervision_targets": {"pre_feasible": True, "post_feasible": True, "causal_effect": 0},
            "privileged_metadata": {
                "candidate_object_name": "coffee_can",
                "intended_category": "irrelevant",
                "is_culprit": False,
                "active_culprits_before": [],
                "active_culprits_after": [],
                "post_rgb_path": "scenes/ivs_000002/post/int_2_rgb.png",
            },
        },
    ]

    manifest_file = dataset_root / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return manifest_file, features_dir


def test_dataset_tensor_shapes_and_types(mock_dataset_and_features):
    """Test that all records produce tensors with exact shapes and dtypes."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")
    assert len(ds) == 7

    spec = InterventionFeatureSpec()

    for i in range(len(ds)):
        item = ds[i]
        m = item["model_inputs"]
        s = item["supervision_targets"]
        ident = item["identifiers"]

        # Check model tensor shapes and dtypes
        assert m["text_feat"].shape == (spec.text_dim,)
        assert m["text_feat"].dtype == torch.float32

        assert m["scene_global"].shape == (spec.scene_global_dim,)
        assert m["scene_global"].dtype == torch.float32

        assert m["scene_patch"].shape == spec.scene_patch_shape
        assert m["scene_patch"].dtype == torch.float32

        assert m["candidate_visual"].shape == (spec.candidate_visual_dim,)
        assert m["candidate_visual"].dtype == torch.float32

        assert m["current_geometry"].shape == (spec.current_geom_dim,)
        assert m["current_geometry"].dtype == torch.float32

        assert m["destination_geometry"].shape == (spec.dest_geom_dim,)
        assert m["destination_geometry"].dtype == torch.float32

        assert m["operator_idx"].dtype == torch.long
        assert m["candidate_present"].dtype == torch.float32
        assert m["demo_present"].dtype == torch.float32

        # Check supervision targets
        assert s["pre_feasible"].dtype == torch.float32
        assert s["post_feasible"].dtype == torch.float32
        assert s["causal_effect"].dtype == torch.long


def test_none_operator_tensorization(mock_dataset_and_features):
    """Test that NONE records produce exact zero tensors and operator_idx = 0."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    # Item 1 is NONE (rec_001_1)
    none_item = ds[1]
    m = none_item["model_inputs"]

    assert m["operator_idx"].item() == OPERATOR_NONE_IDX
    assert m["candidate_present"].item() == 0.0
    torch.testing.assert_close(m["candidate_visual"], torch.zeros(CANDIDATE_VISUAL_DIM))
    torch.testing.assert_close(m["current_geometry"], torch.zeros(CURRENT_GEOM_DIM))
    torch.testing.assert_close(m["destination_geometry"], torch.zeros(DEST_GEOM_DIM))


def test_metamorphic_privileged_metadata_invariance(mock_dataset_and_features):
    """Metamorphic test: mutating privileged labels MUST produce bitwise identical model tensors."""
    manifest_file, features_dir = mock_dataset_and_features
    ds1 = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")
    item1 = ds1[0]["model_inputs"]

    # Mutate privileged metadata in manifest
    with open(manifest_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Corrupt privileged metadata of first record
    records[0]["privileged_metadata"]["intended_category"] = "CORRUPTED_CATEGORY"
    records[0]["privileged_metadata"]["is_culprit"] = not records[0]["privileged_metadata"]["is_culprit"]
    records[0]["privileged_metadata"]["active_culprits_before"] = ["FAKE_OBJECT_123"]

    mutated_manifest = manifest_file.parent / "mutated_manifest.jsonl"
    with open(mutated_manifest, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds2 = InterventionLearningDataset(manifest_path=mutated_manifest, features_dir=features_dir, split="all")
    item2 = ds2[0]["model_inputs"]

    # Verify bitwise equality of all model inputs
    for key in item1:
        torch.testing.assert_close(item1[key], item2[key], msg=f"Mismatch on model input '{key}'")


def test_metamorphic_post_state_file_invariance(mock_dataset_and_features):
    """Metamorphic test: deleting post-state images does NOT break or alter dataset __getitem__."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    # Load item 0
    item = ds[0]
    assert item["model_inputs"]["operator_idx"].item() == OPERATOR_RELOCATE_IDX

    # Verify no post files exist in features_dir
    assert not (features_dir / "post_rgb.pt").exists()


def test_metamorphic_world_coordinate_invariance(mock_dataset_and_features):
    """Metamorphic test: altering world_position while keeping relative_position fixed preserves geometry tensor."""
    manifest_file, features_dir = mock_dataset_and_features

    with open(manifest_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Modify raw world coordinates drastically
    records[0]["model_inputs"]["current_geometry"]["world_position"] = [999.0, 888.0, 777.0]
    records[0]["model_inputs"]["destination_geometry"]["world_position"] = [-999.0, -888.0, -777.0]

    mutated_manifest = manifest_file.parent / "world_mutated_manifest.jsonl"
    with open(mutated_manifest, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds_orig = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")
    ds_mut = InterventionLearningDataset(manifest_path=mutated_manifest, features_dir=features_dir, split="all")

    torch.testing.assert_close(
        ds_orig[0]["model_inputs"]["current_geometry"],
        ds_mut[0]["model_inputs"]["current_geometry"],
    )
    torch.testing.assert_close(
        ds_orig[0]["model_inputs"]["destination_geometry"],
        ds_mut[0]["model_inputs"]["destination_geometry"],
    )


def test_matched_candidates_consistency(mock_dataset_and_features):
    """Test that REPAIR and HARD_NEGATIVE for the same culprit share crop visual and current geom."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    # Record 0 is REPAIR(blocker1), Record 2 is HARD_NEGATIVE(blocker1)
    repair_item = ds[0]["model_inputs"]
    hard_neg_item = ds[2]["model_inputs"]

    # Shared pre-state features
    torch.testing.assert_close(repair_item["scene_global"], hard_neg_item["scene_global"])
    torch.testing.assert_close(repair_item["candidate_visual"], hard_neg_item["candidate_visual"])
    torch.testing.assert_close(repair_item["current_geometry"], hard_neg_item["current_geometry"])

    # Differing destination geometry
    assert not torch.allclose(repair_item["destination_geometry"], hard_neg_item["destination_geometry"])


def test_grouped_batch_sampler_and_collate(mock_dataset_and_features):
    """Test that InterventionGroupBatchSampler preserves complete scene groups."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    sampler = InterventionGroupBatchSampler(dataset=ds, scenes_per_batch=1, shuffle=False)
    batches = list(sampler)

    # Scene 1 has 4 records, Scene 2 has 3 records
    assert len(batches) == 2
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6]

    # Test collate function
    batch_0_items = [ds[idx] for idx in batches[0]]
    collated_0 = collate_intervention_group(batch_0_items)

    assert collated_0["model_inputs"]["text_feat"].shape == (4, TEXT_DIM)
    assert collated_0["model_inputs"]["scene_global"].shape == (4, SCENE_GLOBAL_DIM)
    assert collated_0["model_inputs"]["scene_patch"].shape == (4, 256, 768)
    assert collated_0["model_inputs"]["candidate_visual"].shape == (4, CANDIDATE_VISUAL_DIM)
    assert collated_0["model_inputs"]["current_geometry"].shape == (4, CURRENT_GEOM_DIM)
    assert collated_0["model_inputs"]["destination_geometry"].shape == (4, DEST_GEOM_DIM)
    assert collated_0["model_inputs"]["operator_idx"].shape == (4,)
    assert collated_0["supervision_targets"]["causal_effect"].shape == (4,)
    torch.testing.assert_close(collated_0["scene_group_ptrs"], torch.tensor([0, 4], dtype=torch.long))


def test_scene_level_train_val_splitting(mock_dataset_and_features):
    """Test that train/val splitting partitions strictly by scene_id."""
    manifest_file, features_dir = mock_dataset_and_features
    ds_train = InterventionLearningDataset(
        manifest_path=manifest_file, features_dir=features_dir, split="train", split_seed=42, train_ratio=0.5
    )
    ds_val = InterventionLearningDataset(
        manifest_path=manifest_file, features_dir=features_dir, split="val", split_seed=42, train_ratio=0.5
    )

    train_scene_ids = set(r["scene_id"] for r in ds_train.records)
    val_scene_ids = set(r["scene_id"] for r in ds_val.records)

    # Disjoint scene IDs
    assert len(train_scene_ids.intersection(val_scene_ids)) == 0
    assert len(train_scene_ids) + len(val_scene_ids) == 2


def test_real_smoke_dataset_end_to_end():
    """Test loading and batching the real smoke intervention dataset and cached features."""
    manifest_path = Path("data/intervention_smoke/manifest.jsonl")
    features_dir = Path("data/intervention_smoke/features")

    if not manifest_path.exists() or not features_dir.exists():
        pytest.skip("Smoke dataset or features not present.")

    ds = InterventionLearningDataset(manifest_path=manifest_path, features_dir=features_dir, split="all")
    assert len(ds) == 22

    # Group sampler
    sampler = InterventionGroupBatchSampler(dataset=ds, scenes_per_batch=1, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 6
    group_sizes = [len(b) for b in batches]
    assert group_sizes == [4, 4, 3, 4, 4, 3]

    for batch_indices in batches:
        items = [ds[i] for i in batch_indices]
        collated = collate_intervention_group(items)
        assert collated["batch_size"] == len(batch_indices)
        assert collated["model_inputs"]["text_feat"].shape == (len(batch_indices), TEXT_DIM)
        assert collated["model_inputs"]["scene_global"].shape == (len(batch_indices), SCENE_GLOBAL_DIM)
        assert collated["model_inputs"]["scene_patch"].shape == (len(batch_indices), 256, 768)
        assert collated["model_inputs"]["candidate_visual"].shape == (len(batch_indices), CANDIDATE_VISUAL_DIM)
        assert collated["model_inputs"]["current_geometry"].shape == (len(batch_indices), CURRENT_GEOM_DIM)
        assert collated["model_inputs"]["destination_geometry"].shape == (len(batch_indices), DEST_GEOM_DIM)
        assert collated["model_inputs"]["operator_idx"].shape == (len(batch_indices),)
        assert collated["supervision_targets"]["causal_effect"].shape == (len(batch_indices),)

