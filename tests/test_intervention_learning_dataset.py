"""Comprehensive unit and metamorphic tests for InterventionLearningDataset and Grouped Sampler."""

import hashlib
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
    FEATURE_CACHE_SCHEMA_VERSION,
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
    """Fixture creating a realistic 2-scene mock dataset and Schema 1.1.0 feature cache."""
    dataset_root = tmp_path / "dataset"
    features_dir = tmp_path / "features"
    dataset_root.mkdir()
    features_dir.mkdir()
    (features_dir / "scenes").mkdir()
    (features_dir / "crops").mkdir()

    spec = InterventionFeatureSpec()

    # Pre-scene features
    torch.save(
        {"global": torch.ones(spec.scene_global_dim) * 1.1, "patch": torch.ones(spec.scene_patch_shape) * 1.2, "source_sha256": "sha_scene_1"},
        features_dir / "scenes" / "scene_ivs_000001_features.pt",
    )
    torch.save(
        {"global": torch.ones(spec.scene_global_dim) * 2.1, "patch": torch.ones(spec.scene_patch_shape) * 2.2, "source_sha256": "sha_scene_2"},
        features_dir / "scenes" / "scene_ivs_000002_features.pt",
    )

    # Crop features (neutral sanitized filename)
    torch.save(
        {"global": torch.ones(spec.candidate_visual_dim) * 3.1, "source_sha256": "sha_crop_1"},
        features_dir / "crops" / "crop_scenes_ivs_000001_crops_object_blocker1_png_features.pt",
    )
    torch.save(
        {"global": torch.ones(spec.candidate_visual_dim) * 3.2, "source_sha256": "sha_crop_2"},
        features_dir / "crops" / "crop_scenes_ivs_000001_crops_object_distractor1_png_features.pt",
    )
    torch.save(
        {"global": torch.ones(spec.candidate_visual_dim) * 4.1, "source_sha256": "sha_crop_3"},
        features_dir / "crops" / "crop_scenes_ivs_000002_crops_object_coffee_can_png_features.pt",
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
        },
    ]

    manifest_file = dataset_root / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    # Schema 1.1.0 Cache Index
    cache_index = {
        "schema_version": "2.1.0",
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "source": {
            "manifest_path": "manifest.jsonl",
            "manifest_sha256": manifest_sha,
        },
        "dimensions": {
            "text_dim": TEXT_DIM,
            "scene_global_dim": SCENE_GLOBAL_DIM,
            "scene_patch_shape": list(SCENE_PATCH_SHAPE),
            "candidate_visual_dim": CANDIDATE_VISUAL_DIM,
            "current_geom_dim": CURRENT_GEOM_DIM,
            "dest_geom_dim": DEST_GEOM_DIM,
            "operator_count": 2,
        },
        "scenes": {
            "scenes/ivs_000001/pre_rgb.png": {"feature_path": "scenes/scene_ivs_000001_features.pt", "source_sha256": "sha_scene_1"},
            "scenes/ivs_000002/pre_rgb.png": {"feature_path": "scenes/scene_ivs_000002_features.pt", "source_sha256": "sha_scene_2"},
        },
        "crops": {
            "scenes/ivs_000001/crops/object_blocker1.png": {"feature_path": "crops/crop_scenes_ivs_000001_crops_object_blocker1_png_features.pt", "source_sha256": "sha_crop_1"},
            "scenes/ivs_000001/crops/object_distractor1.png": {"feature_path": "crops/crop_scenes_ivs_000001_crops_object_distractor1_png_features.pt", "source_sha256": "sha_crop_2"},
            "scenes/ivs_000002/crops/object_coffee_can.png": {"feature_path": "crops/crop_scenes_ivs_000002_crops_object_coffee_can_png_features.pt", "source_sha256": "sha_crop_3"},
        },
        "texts": {
            "Open the box.": {"exact_text": "Open the box.", "text_sha256": "sha_text_1"},
            "Place the coffee can in the target region.": {"exact_text": "Place the coffee can in the target region.", "text_sha256": "sha_text_2"},
        },
        "text_features_path": "text_features.pt",
    }

    with open(features_dir / "feature_cache_index.json", "w", encoding="utf-8") as f:
        json.dump(cache_index, f, indent=2)

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

        # Exactly 7 model input fields
        assert set(m.keys()) == {
            "text_feat", "scene_global", "scene_patch", "candidate_visual",
            "current_geometry", "destination_geometry", "operator_idx"
        }

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

        assert s["pre_feasible"].dtype == torch.float32
        assert s["post_feasible"].dtype == torch.float32
        assert s["causal_effect"].dtype == torch.long


def test_none_operator_tensorization(mock_dataset_and_features):
    """Test that NONE records produce exact zero tensors and operator_idx = 0."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    none_item = ds[1]
    m = none_item["model_inputs"]

    assert m["operator_idx"].item() == OPERATOR_NONE_IDX
    torch.testing.assert_close(m["candidate_visual"], torch.zeros(CANDIDATE_VISUAL_DIM))
    torch.testing.assert_close(m["current_geometry"], torch.zeros(CURRENT_GEOM_DIM))
    torch.testing.assert_close(m["destination_geometry"], torch.zeros(DEST_GEOM_DIM))


def test_complete_removal_of_privileged_metadata(mock_dataset_and_features):
    """Hard leakage test: completely stripping privileged_metadata must produce bitwise identical model tensors."""
    manifest_file, features_dir = mock_dataset_and_features
    ds_orig = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    # Load original model tensors
    orig_tensors = [ds_orig[i]["model_inputs"] for i in range(len(ds_orig))]

    # Create manifest with ZERO privileged metadata anywhere
    with open(manifest_file, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    stripped_records = []
    for r in records:
        r_clean = {
            "schema_version": r["schema_version"],
            "record_id": r["record_id"],
            "scene_id": r["scene_id"],
            "intervention_id": r["intervention_id"],
            "model_inputs": r["model_inputs"],
            "supervision_targets": r["supervision_targets"],
        }
        stripped_records.append(r_clean)

    stripped_manifest = manifest_file.parent / "stripped_manifest.jsonl"
    with open(stripped_manifest, "w") as f:
        for r in stripped_records:
            f.write(json.dumps(r) + "\n")

    # Update cache index to match stripped manifest SHA
    stripped_sha = hashlib.sha256(stripped_manifest.read_bytes()).hexdigest()
    with open(features_dir / "feature_cache_index.json", "r") as f:
        idx_dict = json.load(f)
    idx_dict["source"]["manifest_sha256"] = stripped_sha
    with open(features_dir / "feature_cache_index.json", "w") as f:
        json.dump(idx_dict, f, indent=2)

    ds_stripped = InterventionLearningDataset(manifest_path=stripped_manifest, features_dir=features_dir, split="all")

    for i in range(len(ds_stripped)):
        stripped_item = ds_stripped[i]["model_inputs"]
        for key in orig_tensors[i]:
            torch.testing.assert_close(orig_tensors[i][key], stripped_item[key], msg=f"Mismatch on {key} for idx {i}")


def test_wrong_manifest_sha_fails_loudly(mock_dataset_and_features):
    """Test that InterventionLearningDataset rejects a feature cache from a different manifest."""
    manifest_file, features_dir = mock_dataset_and_features

    # Modify one line in manifest to change SHA
    with open(manifest_file, "r") as f:
        lines = f.readlines()
    lines[0] = lines[0].replace("rec_001_0", "rec_999_9")

    wrong_manifest = manifest_file.parent / "wrong_manifest.jsonl"
    with open(wrong_manifest, "w") as f:
        f.writelines(lines)

    with pytest.raises(ValueError, match="extracted from a different manifest"):
        InterventionLearningDataset(manifest_path=wrong_manifest, features_dir=features_dir, split="all")


def test_non_contiguous_scene_batch_rejected(mock_dataset_and_features):
    """Test that collate_intervention_group rejects non-contiguous scene groups."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    # Create non-contiguous batch: [Scene1, Scene2, Scene1]
    bad_batch = [ds[0], ds[4], ds[1]]  # ds[0] is ivs_000001, ds[4] is ivs_000002, ds[1] is ivs_000001

    with pytest.raises(ValueError, match="Non-contiguous scene group detected"):
        collate_intervention_group(bad_batch)


def test_quaternion_antipodal_equivalence():
    """Test that quaternion distance correctly treats q and -q as identical."""
    q1 = np.array([0.818, 0.519, -0.132, -0.208])
    q2 = -q1.copy()

    dist = min(float(np.linalg.norm(q1 - q2)), float(np.linalg.norm(q1 + q2)))
    assert dist == 0.0


def test_grouped_batch_sampler_and_collate(mock_dataset_and_features):
    """Test that InterventionGroupBatchSampler preserves complete scene groups."""
    manifest_file, features_dir = mock_dataset_and_features
    ds = InterventionLearningDataset(manifest_path=manifest_file, features_dir=features_dir, split="all")

    sampler = InterventionGroupBatchSampler(dataset=ds, scenes_per_batch=1, shuffle=False)
    batches = list(sampler)

    assert len(batches) == 2
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6]

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


def test_real_smoke_dataset_end_to_end():
    """Test loading and batching the real smoke intervention dataset and cached features."""
    manifest_path = Path("data/intervention_smoke/manifest.jsonl")
    features_dir = Path("data/intervention_smoke/features")

    if not manifest_path.exists() or not features_dir.exists():
        pytest.skip("Smoke dataset or features not present.")

    # Recompute features if index is older schema
    index_file = features_dir / "feature_cache_index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            idx = json.load(f)
        if idx.get("feature_cache_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            pytest.skip("Cache needs regeneration to Schema 1.1.0.")

    ds = InterventionLearningDataset(manifest_path=manifest_path, features_dir=features_dir, split="all")
    assert len(ds) == 22

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
