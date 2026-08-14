"""
End-to-end tests for intervention dataset generation, nested manifest schema validation,
lossless raw segmentation, and independent physical recomputation.
"""

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml

from src.interventions.intervention_records import (
    MODEL_INPUTS_ALLOWED_KEYS,
    SCHEMA_VERSION,
)
from scripts.generate_intervention_dataset import generate_dataset_from_config
from scripts.validate_intervention_dataset import validate_intervention_dataset


@pytest.fixture(scope="module")
def shared_smoke_dataset(tmp_path_factory):
    """Generate smoke dataset once and share across validation tests."""
    out_dir = tmp_path_factory.mktemp("smoke_dataset")
    config_p = Path("configs/intervention/smoke.yaml")
    with open(config_p, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["output_root"] = str(out_dir)
    tmp_config_p = out_dir / "smoke.yaml"
    with open(tmp_config_p, "w") as f:
        yaml.dump(cfg, f)

    meta = generate_dataset_from_config(tmp_config_p)
    return out_dir, meta


def test_smoke_scene_spec_composition():
    config_p = Path("configs/intervention/smoke.yaml")
    with open(config_p, "r") as f:
        smoke_config_dict = yaml.safe_load(f)
    scenes = smoke_config_dict["scenes"]
    assert len(scenes) == 6

    # Verify task distribution
    task1_scenes = [s for s in scenes if s["task_id"] == "task_1"]
    task2_scenes = [s for s in scenes if s["task_id"] == "task_2"]
    assert len(task1_scenes) == 3
    assert len(task2_scenes) == 3

    # Verify state distribution
    stop_scenes = [s for s in scenes if s["intended_base_state"] == "STOP"]
    proceed_scenes = [s for s in scenes if s["intended_base_state"] == "PROCEED"]
    assert len(stop_scenes) == 4
    assert len(proceed_scenes) == 2

    # Verify Task 2 coffee_can action subject invariant
    for s in task2_scenes:
        assert s["action"]["arguments"]["object"] == "coffee_can"
        assert s["action_subject_name"] == "pick_can"


def test_smoke_dataset_generation_end_to_end(shared_smoke_dataset):
    out_dir, metadata = shared_smoke_dataset
    assert metadata["record_count"] == 22
    assert metadata["scene_count"] == 6
    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "dataset_metadata.json").exists()


def test_manifest_schema_partitioning_and_allowlist(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset
    manifest_p = out_dir / "manifest.jsonl"

    with open(manifest_p, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            assert rec["schema_version"] == SCHEMA_VERSION
            assert "model_inputs" in rec
            assert "supervision_targets" in rec
            assert "privileged_metadata" in rec

            # Check model_inputs keys are strictly inside allowlist
            m_keys = set(rec["model_inputs"].keys())
            assert m_keys.issubset(MODEL_INPUTS_ALLOWED_KEYS)

            # Check paths are dataset-root-relative
            assert not rec["model_inputs"]["pre_rgb_path"].startswith("/")
            assert not rec["model_inputs"]["pre_rgb_path"].startswith("data/")


def test_all_referenced_files_exist_and_readable(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset

    with open(out_dir / "manifest.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            pre_rgb = out_dir / rec["model_inputs"]["pre_rgb_path"]
            assert pre_rgb.exists()

            if rec["model_inputs"]["candidate_object_crop_path"]:
                crop = out_dir / rec["model_inputs"]["candidate_object_crop_path"]
                assert crop.exists()

            post_rgb = out_dir / rec["privileged_metadata"]["post_rgb_path"]
            assert post_rgb.exists()

            pre_seg = out_dir / rec["privileged_metadata"]["pre_segmentation_path"]
            assert pre_seg.exists()

            post_seg = out_dir / rec["privileged_metadata"]["post_segmentation_path"]
            assert post_seg.exists()


def test_segmentation_raw_int32_dtype_and_shape(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset

    with open(out_dir / "manifest.jsonl", "r") as f:
        first_line = json.loads(f.readline().strip())
        seg_file = out_dir / first_line["privileged_metadata"]["pre_segmentation_path"]
        arr = np.load(seg_file)
        assert arr.dtype == np.int32
        assert arr.shape == (480, 640, 2)


def test_effect_coverage_in_smoke_manifest(shared_smoke_dataset):
    _, meta = shared_smoke_dataset
    effects = meta["effect_counts"]
    assert effects["1"] == 4
    assert effects["0"] == 16
    assert effects["-1"] == 2

    cats = meta["category_counts"]
    assert cats["repair"] == 4
    assert cats["hard_negative"] == 4
    assert cats["harmful"] == 2
    assert cats["irrelevant"] == 6
    assert cats["identity"] == 6


def test_independent_physical_recomputation(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset
    report = validate_intervention_dataset(
        manifest_path=out_dir / "manifest.jsonl",
        report_output_path=out_dir / "report.json",
        generator_commit="test_commit_sha",
    )

    assert report["validation_status"] == "PASSED"
    assert report["recomputed_delta_agreement"] == "22/22 (100%)"


def test_relational_culprit_identity_rules(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset

    with open(out_dir / "manifest.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            cat = rec["privileged_metadata"]["intended_category"]
            c_before = rec["privileged_metadata"]["active_culprits_before"]
            c_after = rec["privileged_metadata"]["active_culprits_after"]
            eff = rec["supervision_targets"]["causal_effect"]

            if cat == "repair":
                assert eff == 1
                assert len(c_after) == 0
            elif cat == "hard_negative":
                assert eff == 0
                assert len(c_after) > 0
            elif cat == "harmful":
                assert eff == -1
                assert len(c_before) == 0
                assert len(c_after) == 1
            elif cat == "identity":
                assert eff == 0
                assert c_before == c_after


def test_collateral_displacement_threshold(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset
    report = validate_intervention_dataset(
        manifest_path=out_dir / "manifest.jsonl",
    )

    assert report["max_collateral_displacement_m"] <= 0.05


def test_deterministic_dataset_regeneration(tmp_path):
    config_p = Path("configs/intervention/smoke.yaml")
    with open(config_p, "r") as f:
        cfg = yaml.safe_load(f)

    # First run
    out1 = tmp_path / "run1"
    cfg["output_root"] = str(out1)
    p1 = tmp_path / "c1.yaml"
    with open(p1, "w") as f:
        yaml.dump(cfg, f)
    meta1 = generate_dataset_from_config(p1)

    # Second run
    out2 = tmp_path / "run2"
    cfg["output_root"] = str(out2)
    p2 = tmp_path / "c2.yaml"
    with open(p2, "w") as f:
        yaml.dump(cfg, f)
    meta2 = generate_dataset_from_config(p2)

    assert meta1["manifest_sha256"] == meta2["manifest_sha256"]


def test_opaque_ids_no_semantic_leakage(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset

    with open(out_dir / "manifest.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            interv_id = rec["intervention_id"]
            # Intervention ID format must be int_<sha16>
            assert interv_id.startswith("int_")
            # Must not contain category words
            for term in ["repair", "harmful", "negative", "irrelevant", "identity"]:
                assert term not in interv_id.lower()


def test_validation_cli_subprocess_success(shared_smoke_dataset):
    out_dir, _ = shared_smoke_dataset

    res = subprocess.run(
        [
            sys.executable,
            "scripts/validate_intervention_dataset.py",
            "--manifest",
            str(out_dir / "manifest.jsonl"),
            "--report",
            str(out_dir / "report.json"),
            "--commit",
            "cli_test_commit",
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0
    assert "ALL 7 VALIDATION GATES PASSED" in res.stdout
