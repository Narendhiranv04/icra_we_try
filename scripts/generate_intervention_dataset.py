#!/usr/bin/env python3
"""
CLI script to generate an intervention dataset from a YAML configuration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interventions.intervention_records import (
    InterventionSceneSpec,
    InterventionDatasetRecord,
    Action,
    SCHEMA_VERSION,
)
from src.interventions.intervention_scene_generator import InterventionSceneGenerator
from src.interventions.intervention_validator import InterventionValidator
from src.interventions.intervention_generator import InterventionGenerator


def resolve_demonstration_reference(task_id: str, repo_root: Path) -> str:
    """Resolve an existing, verified demonstration file path or return null."""
    if "1" in str(task_id):
        demo_candidate = repo_root / "data/pilot_demos/demo_task1_pilot_001.mp4"
    else:
        demo_candidate = repo_root / "data/pilot_demos/demo_task2_pilot_001.mp4"

    if demo_candidate.exists():
        return str(demo_candidate.relative_to(repo_root))
    return None


def generate_dataset_from_config(config_path: Path, generator_commit: Optional[str] = None) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    with open(config_path, "rb") as f:
        config_sha = hashlib.sha256(f.read()).hexdigest()

    output_root = Path(cfg.get("output_root", "data/intervention_smoke"))
    output_root.mkdir(parents=True, exist_ok=True)

    resolution = tuple(cfg.get("resolution", [640, 480]))
    min_reloc = float(cfg.get("min_relocation_m", 0.03))
    max_collateral = float(cfg.get("max_collateral_translation_m", 0.05))
    min_inst_px = int(cfg.get("min_instance_pixels", 50))
    min_target_px = int(cfg.get("min_relation_target_pixels", 50))
    max_torso_frac = float(cfg.get("max_torso_fraction", 0.25))

    max_base_retries = int(cfg.get("max_base_retries", 10))
    max_cand_retries = int(cfg.get("max_candidate_set_retries", 5))

    validator = InterventionValidator(settle_steps=300, max_collateral_translation_m=max_collateral, hold_observation_robot=True)
    generator = InterventionGenerator(min_relocation_m=min_reloc)

    scene_gen = InterventionSceneGenerator(
        validator=validator,
        generator=generator,
        resolution=resolution,
        min_relocation_m=min_reloc,
        max_collateral_translation_m=max_collateral,
        min_instance_pixels=min_inst_px,
        min_relation_target_pixels=min_target_px,
        max_torso_fraction=max_torso_frac,
    )

    all_records: List[InterventionDatasetRecord] = []
    resolved_specs: Dict[str, Any] = {}

    print(f"======================================================================")
    print(f"Generating Intervention Dataset: {cfg.get('dataset_name', 'dataset')}")
    print(f"Output Root: {output_root.resolve()}")
    print(f"======================================================================")

    for s_dict in cfg.get("scenes", []):
        spec = InterventionSceneSpec(
            scene_id=s_dict["scene_id"],
            task_id=s_dict["task_id"],
            intended_base_state=s_dict["intended_base_state"],
            instruction=s_dict["instruction"],
            action=Action(
                action_type=s_dict["action"]["action_type"],
                target=s_dict["action"]["target"],
                arguments=s_dict["action"].get("arguments", {}),
            ),
            action_subject_name=s_dict.get("action_subject_name"),
            intended_culprit_type=s_dict.get("intended_culprit_type"),
            intended_distractor_types=tuple(s_dict.get("intended_distractor_types", ())),
            seed=int(s_dict["seed"]),
            split=s_dict.get("split", "id"),
            background_profile=s_dict.get("background_profile", "bg_neutral_wood"),
        )

        demo_ref = s_dict.get("demonstration_reference", cfg.get("demonstration_reference", None))
        print(f"--> Generating base scene {spec.scene_id} ({spec.task_id} {spec.intended_base_state})...")
        records, resolved_spec = scene_gen.generate_scene_dataset(
            spec=spec,
            dataset_root=output_root,
            max_base_retries=max_base_retries,
            max_candidate_set_retries=max_cand_retries,
            demonstration_reference=demo_ref,
        )
        print(f"    Accepted {len(records)} candidate intervention records.")
        all_records.extend(records)
        resolved_specs[spec.scene_id] = resolved_spec.to_dict()

    # Write manifest.jsonl
    manifest_path = output_root / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    with open(manifest_path, "rb") as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()

    # Compute statistics
    effect_counts = {1: 0, 0: 0, -1: 0}
    category_counts = {}
    for rec in all_records:
        eff = rec.supervision_targets["causal_effect"]
        effect_counts[eff] = effect_counts.get(eff, 0) + 1
        cat = rec.privileged_metadata.get("intended_category")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    commit_str = generator_commit or cfg.get("generator_commit")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": cfg.get("dataset_name", "intervention_smoke"),
        "manifest_path": "manifest.jsonl",
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "generator_commit": commit_str,
        "resolution": list(resolution),
        "scene_count": len(cfg.get("scenes", [])),
        "record_count": len(all_records),
        "effect_counts": {str(k): v for k, v in effect_counts.items()},
        "category_counts": category_counts,
        "min_relocation_m": min_reloc,
        "max_collateral_translation_m": max_collateral,
        "min_instance_pixels": min_inst_px,
        "min_relation_target_pixels": min_target_px,
        "max_torso_fraction": max_torso_frac,
        "tolerances": {
            "rgb_pre_reconstruction_rmse_tolerance": float(cfg.get("rgb_pre_reconstruction_rmse_tolerance", 1.0)),
            "rgb_post_reconstruction_rmse_tolerance": float(cfg.get("rgb_post_reconstruction_rmse_tolerance", 1.0)),
            "none_rgb_rmse_tolerance": float(cfg.get("none_rgb_rmse_tolerance", 1.0)),
            "camera_translation_tolerance_m": float(cfg.get("camera_translation_tolerance_m", 0.001)),
            "camera_rotation_tolerance_rad": float(cfg.get("camera_rotation_tolerance_rad", 0.005)),
            "max_collateral_translation_m": max_collateral,
            "geometry_position_tolerance_m": float(cfg.get("geometry_position_tolerance_m", 0.001)),
            "geometry_quaternion_tolerance": float(cfg.get("geometry_quaternion_tolerance", 0.01)),
        },
        "resolved_scenes": resolved_specs,
    }

    metadata_path = output_root / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n======================================================================")
    print(f"Generation Complete!")
    print(f"Total Scenes: {len(cfg.get('scenes', []))}")
    print(f"Total Records: {len(all_records)}")
    print(f"Effect Distribution: +1: {effect_counts.get(1, 0)}, 0: {effect_counts.get(0, 0)}, -1: {effect_counts.get(-1, 0)}")
    print(f"Category Distribution: {category_counts}")
    print(f"Manifest SHA256: {manifest_sha}")
    print(f"Metadata written to: {metadata_path}")
    print(f"======================================================================")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate Intervention Dataset")
    parser.add_argument("--config", type=str, default="configs/intervention/smoke.yaml", help="Path to config YAML")
    parser.add_argument("--commit", type=str, default=None, help="Git commit SHA of the generator implementation")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    generate_dataset_from_config(config_path, generator_commit=args.commit)


if __name__ == "__main__":
    main()
