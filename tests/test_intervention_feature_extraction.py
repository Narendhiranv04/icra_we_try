"""Unit and integration tests for intervention feature extraction and caching."""

import json
from pathlib import Path
import tempfile
import torch
import pytest
from PIL import Image

from scripts.precompute_intervention_features import (
    extract_intervention_features,
    compute_file_sha256,
    compute_text_sha256,
)
from src.learning.intervention_feature_spec import (
    InterventionFeatureSpec,
    TEXT_DIM,
    SCENE_GLOBAL_DIM,
    SCENE_PATCH_SHAPE,
    CANDIDATE_VISUAL_DIM,
)


class StubVisionEncoder:
    def __init__(self, embed_dim=768, num_patches=256):
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.call_count = 0

    def __call__(self, images):
        self.call_count += len(images)
        b = len(images)
        cls_token = torch.ones(b, self.embed_dim) * 0.42
        patch_tokens = torch.ones(b, self.num_patches, self.embed_dim) * 0.24
        return cls_token, patch_tokens


class StubTextEncoder:
    def __init__(self, embed_dim=384):
        self.embed_dim = embed_dim
        self.model_name = "stub_text_encoder"
        self.call_count = 0

    def __call__(self, texts):
        self.call_count += len(texts)
        b = len(texts)
        return torch.ones(b, self.embed_dim) * 0.77


def test_feature_extraction_deduplication(tmp_path):
    """Test that scenes, crops, and text are deduplicated during feature extraction."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    scenes_dir = dataset_root / "scenes" / "ivs_000001"
    crops_dir = scenes_dir / "crops"
    crops_dir.mkdir(parents=True)

    pre_rgb_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    pre_rgb_path = scenes_dir / "pre_rgb.png"
    pre_rgb_img.save(pre_rgb_path)

    crop_img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    crop_path = crops_dir / "object_blocker1.png"
    crop_img.save(crop_path)

    manifest_records = [
        {
            "schema_version": "2.1.0",
            "record_id": "rec_000",
            "scene_id": "ivs_000001",
            "intervention_id": "int_000",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000001/crops/object_blocker1.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"relative_position": [0.0, 0.0, 0.05]},
                "destination_geometry": {"relative_position": [-0.25, 0.0, -0.12]},
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": True, "causal_effect": 1},
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_001",
            "scene_id": "ivs_000001",
            "intervention_id": "int_001",
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
        },
        {
            "schema_version": "2.1.0",
            "record_id": "rec_002",
            "scene_id": "ivs_000001",
            "intervention_id": "int_002",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000001/crops/object_blocker1.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"relative_position": [0.0, 0.0, 0.05]},
                "destination_geometry": {"relative_position": [0.0, 0.0, 0.02]},
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": False, "causal_effect": 0},
        },
    ]

    manifest_file = dataset_root / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")

    out_features_dir = tmp_path / "features"

    vision_stub = StubVisionEncoder()
    text_stub = StubTextEncoder()

    cache_index = extract_intervention_features(
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        out_dir=out_features_dir,
        vision_encoder=vision_stub,
        text_encoder=text_stub,
    )

    assert cache_index["scene_count"] == 1
    assert cache_index["crop_count"] == 1
    assert cache_index["instruction_count"] == 1
    assert cache_index["record_count"] == 3

    assert vision_stub.call_count == 2
    assert text_stub.call_count == 1

    # Second pass: unchanged files must reuse cache without encoder calls
    vision_stub.call_count = 0
    text_stub.call_count = 0
    extract_intervention_features(
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        out_dir=out_features_dir,
        vision_encoder=vision_stub,
        text_encoder=text_stub,
        force=False,
    )
    assert vision_stub.call_count == 0
    assert text_stub.call_count == 0


def test_cache_invalidation_on_source_image_change(tmp_path):
    """Test that modifying a source image invalidates and recomputes the feature cache."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    scenes_dir = dataset_root / "scenes" / "ivs_000001"
    crops_dir = scenes_dir / "crops"
    crops_dir.mkdir(parents=True)

    pre_rgb_path = scenes_dir / "pre_rgb.png"
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(pre_rgb_path)

    crop_path = crops_dir / "object_blocker1.png"
    Image.new("RGB", (100, 100), color=(40, 50, 60)).save(crop_path)

    manifest_file = dataset_root / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "schema_version": "2.1.0",
            "record_id": "rec_000",
            "scene_id": "ivs_000001",
            "intervention_id": "int_000",
            "model_inputs": {
                "task_id": "task_1",
                "instruction": "Open the box.",
                "pre_rgb_path": "scenes/ivs_000001/pre_rgb.png",
                "candidate_object_crop_path": "scenes/ivs_000001/crops/object_blocker1.png",
                "intervention_operator": "RELOCATE",
                "current_geometry": {"relative_position": [0.0, 0.0, 0.05]},
                "destination_geometry": {"relative_position": [-0.25, 0.0, -0.12]},
            },
            "supervision_targets": {"pre_feasible": False, "post_feasible": True, "causal_effect": 1},
        }) + "\n")

    out_features_dir = tmp_path / "features"
    vision_stub = StubVisionEncoder()
    text_stub = StubTextEncoder()

    # Pass 1
    extract_intervention_features(
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        out_dir=out_features_dir,
        vision_encoder=vision_stub,
        text_encoder=text_stub,
    )
    assert vision_stub.call_count == 2

    # Mutate source crop image
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(crop_path)

    # Pass 2: crop must be recomputed
    vision_stub.call_count = 0
    extract_intervention_features(
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        out_dir=out_features_dir,
        vision_encoder=vision_stub,
        text_encoder=text_stub,
        force=False,
    )
    # Only crop recomputed, pre scene reused
    assert vision_stub.call_count == 1


def test_cache_invalidation_on_encoder_signature_change(tmp_path):
    """Test that changing encoder signature invalidates cache even if source image is unchanged."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    scenes_dir = dataset_root / "scenes" / "ivs_000001"
    scenes_dir.mkdir(parents=True)

    pre_rgb_path = scenes_dir / "pre_rgb.png"
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(pre_rgb_path)

    manifest_file = dataset_root / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "schema_version": "2.1.0",
            "record_id": "rec_000",
            "scene_id": "ivs_000001",
            "intervention_id": "int_000",
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
        }) + "\n")

    out_features_dir = tmp_path / "features"
    vision_stub1 = StubVisionEncoder()
    vision_stub1.model_name = "dinov2_model_v1"
    text_stub = StubTextEncoder()

    # Pass 1 with model v1
    extract_intervention_features(
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        out_dir=out_features_dir,
        vision_encoder=vision_stub1,
        text_encoder=text_stub,
    )
    assert vision_stub1.call_count == 1

    # Pass 2 with model v2 (same output dimensions, different encoder signature)
    vision_stub2 = StubVisionEncoder()
    vision_stub2.model_name = "dinov2_model_v2"
    extract_intervention_features(
        manifest_path=manifest_file,
        dataset_root=dataset_root,
        out_dir=out_features_dir,
        vision_encoder=vision_stub2,
        text_encoder=text_stub,
        force=False,
    )
    assert vision_stub2.call_count == 1  # Must recompute due to signature mismatch


def test_hashed_filename_collision_safety():
    """Test that path pairs that would collide under old sanitization get distinct hashed cache filenames."""
    from scripts.precompute_intervention_features import hash_cache_filename

    path_a = "scenes/ivs_000001/crops/object_a_b.png"
    path_b = "scenes/ivs_000001/crops/object/a_b.png"

    file_a = hash_cache_filename(path_a, "crop")
    file_b = hash_cache_filename(path_b, "crop")

    assert file_a != file_b, f"Colliding cache filenames: {file_a} == {file_b}"

