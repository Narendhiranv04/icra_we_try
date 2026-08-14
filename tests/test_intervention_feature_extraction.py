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
    # Create mock dataset directory
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    scenes_dir = dataset_root / "scenes" / "ivs_000001"
    crops_dir = scenes_dir / "crops"
    crops_dir.mkdir(parents=True)

    # Create dummy images
    pre_rgb_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    pre_rgb_path = scenes_dir / "pre_rgb.png"
    pre_rgb_img.save(pre_rgb_path)

    crop_img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    crop_path = crops_dir / "object_blocker1.png"
    crop_img.save(crop_path)

    # Manifest with 3 records for 1 scene: 1 REPAIR, 1 NONE, 1 HARD_NEGATIVE
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
            "privileged_metadata": {"candidate_object_name": "blocker1", "intended_category": "repair"},
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
            "privileged_metadata": {"candidate_object_name": None, "intended_category": "identity"},
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
            "privileged_metadata": {"candidate_object_name": "blocker1", "intended_category": "hard_negative"},
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

    # Verify counts: exactly 1 scene, 1 crop, 1 instruction across 3 records
    assert cache_index["scene_count"] == 1
    assert cache_index["crop_count"] == 1
    assert cache_index["instruction_count"] == 1
    assert cache_index["record_count"] == 3

    # Verify cache files exist
    assert (out_features_dir / "scenes" / "scene_ivs_000001_features.pt").exists()
    assert (out_features_dir / "crops" / "crop_ivs_000001_blocker1_features.pt").exists()
    assert (out_features_dir / "text_features.pt").exists()
    assert (out_features_dir / "feature_cache_index.json").exists()

    # Verify encoder was invoked only once per unique visual item
    assert vision_stub.call_count == 2  # 1 for pre scene + 1 for crop
    assert text_stub.call_count == 1    # 1 for unique instruction
