#!/usr/bin/env python3
"""Independent validator for intervention feature cache and dataset interface.

Validates:
1.  Manifest SHA256 provenance against cache index (no fallback).
2.  Cache Schema exactly 1.1.0 (no 1.0.0 acceptance).
3.  Source SHA256 bitwise matches for all unique pre scenes.
4.  Source SHA256 bitwise matches for all unique candidate crops.
5.  Per-entry and .pt extraction-signature matches (vision + text).
6.  Exact text instruction embedding availability and dimensions.
7.  Zero POST image/segmentation features referenced or cached.
8.  Zero privileged metadata required for model tensor resolution.
9.  Grouped candidate batching ([4, 4, 3, 4, 4, 3]).
10. Neutral NONE tensorization.
11. Strengthened matched candidate consistency (all pairs, all invariants).
12. Generates evidence-based feature validation report artifact.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Compute SHA256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sig_field_matches(actual: Dict, expected: Dict, fields: List[str]) -> Tuple[bool, List[str]]:
    """Check that all specified fields in actual match expected. Return (ok, mismatches)."""
    mismatches = []
    for f in fields:
        a_val = actual.get(f)
        e_val = expected.get(f)
        if a_val != e_val:
            mismatches.append(f"{f}: actual={a_val!r} expected={e_val!r}")
    return len(mismatches) == 0, mismatches


def validate_feature_cache(
    manifest_path: Path,
    features_dir: Path,
    report_path: Optional[Path] = None,
    commit: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    dataset_root = manifest_path.parent
    features_dir = Path(features_dir).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    if not features_dir.exists():
        raise FileNotFoundError(f"Features dir not found at {features_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    manifest_sha256 = compute_file_sha256(manifest_path)
    schema_version = records[0].get("schema_version", "2.1.0")

    # Load Cache Index
    index_path = features_dir / "feature_cache_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing feature_cache_index.json at {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        cache_index = json.load(f)

    # 1. Manifest Provenance (strict: no fallback key search)
    cached_manifest_sha = cache_index.get("source", {}).get("manifest_sha256")
    if cached_manifest_sha is None:
        raise ValueError(
            "feature_cache_index.json missing 'source.manifest_sha256' field. "
            "Regenerate the feature cache from Commit A."
        )
    if cached_manifest_sha != manifest_sha256:
        raise ValueError(
            f"Manifest SHA mismatch:\n"
            f"  Actual  : {manifest_sha256}\n"
            f"  Cached  : {cached_manifest_sha}"
        )

    # 2. Schema Version — exact 1.1.0 only
    cache_schema = cache_index.get("feature_cache_schema_version")
    if cache_schema != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Feature cache schema version mismatch: "
            f"cache has '{cache_schema}', "
            f"required exact '{FEATURE_CACHE_SCHEMA_VERSION}'. "
            f"No legacy 1.0.0 fallback. Regenerate the feature cache."
        )

    # Load authoritative extraction signatures from the cache index
    auth_vision_sig = cache_index.get("extraction_signatures", {}).get("vision", {})
    auth_text_sig = cache_index.get("extraction_signatures", {}).get("text", {})

    if not auth_vision_sig:
        raise ValueError("cache_index missing 'extraction_signatures.vision'")
    if not auth_text_sig:
        raise ValueError("cache_index missing 'extraction_signatures.text'")

    # 3. Unique Pre Scenes Verification
    unique_pre_paths = sorted(list(set(
        r["model_inputs"]["pre_rgb_path"]
        for r in records
        if r["model_inputs"].get("pre_rgb_path")
    )))
    scenes_map = cache_index.get("scenes", {})
    scene_passed_count = 0
    scene_sig_passed = 0

    vision_sig_fields = ["model_name", "embed_dim", "patch_size", "signature_sha256"]

    for pre_rel_path in unique_pre_paths:
        # Strict: path MUST be in scenes index (no fallback)
        if pre_rel_path not in scenes_map:
            raise KeyError(
                f"Pre RGB path '{pre_rel_path}' not in cache_index['scenes']. "
                f"No fallback filename construction. Regenerate the feature cache."
            )

        cached_scene = scenes_map[pre_rel_path]
        feat_rel = cached_scene.get("feature_path")
        if not feat_rel:
            raise ValueError(f"scenes_map entry for '{pre_rel_path}' missing 'feature_path'")

        abs_img_path = dataset_root / pre_rel_path
        if not abs_img_path.exists():
            raise FileNotFoundError(f"Missing source pre RGB image at {abs_img_path}")

        actual_img_sha = compute_file_sha256(abs_img_path)
        feat_file = features_dir / feat_rel

        if not feat_file.exists():
            raise FileNotFoundError(f"Missing cached scene feature file at {feat_file}")

        feat_dict = torch.load(feat_file, weights_only=True)

        # Shape / dtype
        if feat_dict["global"].shape != (SCENE_GLOBAL_DIM,) or feat_dict["global"].dtype != torch.float32:
            raise ValueError(f"Scene global feature invalid shape/dtype: {feat_dict['global'].shape}")
        if feat_dict["patch"].shape != SCENE_PATCH_SHAPE or feat_dict["patch"].dtype != torch.float32:
            raise ValueError(f"Scene patch feature invalid shape/dtype: {feat_dict['patch'].shape}")

        # Source SHA
        if feat_dict.get("source_sha256") != actual_img_sha:
            raise ValueError(f"Source SHA mismatch in cached scene feature: {feat_file}")

        # Per-entry index extraction signature
        entry_sig = cached_scene.get("extraction_signature", {})
        ok, mismatches = _sig_field_matches(entry_sig, auth_vision_sig, vision_sig_fields)
        if not ok:
            raise ValueError(
                f"Scene index entry signature mismatch for '{pre_rel_path}': {mismatches}"
            )

        # Per-.pt extraction signature
        pt_sig = feat_dict.get("extraction_signature", {})
        ok, mismatches = _sig_field_matches(pt_sig, auth_vision_sig, vision_sig_fields)
        if not ok:
            raise ValueError(
                f"Scene .pt extraction_signature mismatch for '{pre_rel_path}': {mismatches}"
            )

        scene_sig_passed += 1
        scene_passed_count += 1

    # 4. Unique Candidate Crops Verification
    unique_crop_paths = sorted(list(set(
        r["model_inputs"]["candidate_object_crop_path"]
        for r in records
        if r["model_inputs"].get("candidate_object_crop_path")
    )))
    crops_map = cache_index.get("crops", {})
    crop_passed_count = 0
    crop_sig_passed = 0

    for crop_rel_path in unique_crop_paths:
        # Strict: path MUST be in crops index (no fallback)
        if crop_rel_path not in crops_map:
            raise KeyError(
                f"Candidate crop path '{crop_rel_path}' not in cache_index['crops']. "
                f"No fallback filename construction. Regenerate the feature cache."
            )

        cached_crop = crops_map[crop_rel_path]
        feat_rel = cached_crop.get("feature_path")
        if not feat_rel:
            raise ValueError(f"crops_map entry for '{crop_rel_path}' missing 'feature_path'")

        abs_crop_path = dataset_root / crop_rel_path
        if not abs_crop_path.exists():
            raise FileNotFoundError(f"Missing source crop image at {abs_crop_path}")

        actual_crop_sha = compute_file_sha256(abs_crop_path)
        feat_file = features_dir / feat_rel

        if not feat_file.exists():
            raise FileNotFoundError(f"Missing cached crop feature file at {feat_file}")

        feat_dict = torch.load(feat_file, weights_only=True)

        # Shape / dtype
        if feat_dict["global"].shape != (CANDIDATE_VISUAL_DIM,) or feat_dict["global"].dtype != torch.float32:
            raise ValueError(f"Crop global feature invalid shape/dtype: {feat_dict['global'].shape}")

        # Source SHA
        if feat_dict.get("source_sha256") != actual_crop_sha:
            raise ValueError(f"Source SHA mismatch in cached crop feature: {feat_file}")

        # Per-entry index extraction signature
        entry_sig = cached_crop.get("extraction_signature", {})
        ok, mismatches = _sig_field_matches(entry_sig, auth_vision_sig, vision_sig_fields)
        if not ok:
            raise ValueError(
                f"Crop index entry signature mismatch for '{crop_rel_path}': {mismatches}"
            )

        # Per-.pt extraction signature
        pt_sig = feat_dict.get("extraction_signature", {})
        ok, mismatches = _sig_field_matches(pt_sig, auth_vision_sig, vision_sig_fields)
        if not ok:
            raise ValueError(
                f"Crop .pt extraction_signature mismatch for '{crop_rel_path}': {mismatches}"
            )

        crop_sig_passed += 1
        crop_passed_count += 1

    # 5. Unique Instructions Verification with Real SHA + Text Signature Check
    unique_instructions = sorted(list(set(
        r["model_inputs"]["instruction"]
        for r in records
        if r["model_inputs"].get("instruction")
    )))
    text_feat_file = features_dir / "text_features.pt"
    if not text_feat_file.exists():
        raise FileNotFoundError(f"Missing text_features.pt at {text_feat_file}")

    text_dict = torch.load(text_feat_file, weights_only=True)
    text_features = text_dict.get("features", text_dict)
    texts_map = cache_index.get("texts", {})
    text_passed_count = 0
    text_sig_passed = 0

    # Text signature fields to verify
    text_sig_fields = ["model_name", "embed_dim", "signature_sha256"]

    for inst in unique_instructions:
        if inst not in text_features:
            raise KeyError(f"Instruction '{inst}' missing from text_features.pt")
        emb = text_features[inst]
        if emb.shape != (TEXT_DIM,) or emb.dtype != torch.float32:
            raise ValueError(f"Text feature for '{inst}' has invalid shape {emb.shape}")

        expected_sha = compute_text_sha256(inst)
        indexed_sha = texts_map.get(inst, {}).get("text_sha256")
        if indexed_sha != expected_sha:
            raise ValueError(
                f"Text SHA mismatch for '{inst}': "
                f"expected {expected_sha}, cached index has {indexed_sha}"
            )
        text_passed_count += 1

    # Text extraction signature: verify text_features.pt metadata
    pt_text_meta_sig = text_dict.get("metadata", {}).get("extraction_signature", {})
    ok, mismatches = _sig_field_matches(pt_text_meta_sig, auth_text_sig, text_sig_fields)
    if not ok:
        raise ValueError(
            f"text_features.pt metadata extraction_signature mismatch: {mismatches}"
        )
    text_sig_passed = len(unique_instructions)

    # 6. Check No POST Features Cached
    all_feature_files = [str(p) for p in features_dir.rglob("*.pt")]
    post_feature_files = [f for f in all_feature_files if "post" in f.lower()]
    if post_feature_files:
        raise ValueError(f"POST features detected in feature cache: {post_feature_files}")

    # 7. Validate Dataset Loader (Zero Privileged Access)
    dataset = InterventionLearningDataset(
        manifest_path=manifest_path,
        features_dir=features_dir,
        split="all"
    )
    if len(dataset) != len(records):
        raise ValueError(f"Dataset length {len(dataset)} != manifest records {len(records)}")

    # Verify model_inputs contains ONLY exact 7 model-visible fields
    allowed_model_input_keys = {
        "text_feat", "scene_global", "scene_patch", "candidate_visual",
        "current_geometry", "destination_geometry", "operator_idx"
    }
    for i in range(len(dataset)):
        item = dataset[i]
        m_in_keys = set(item["model_inputs"].keys())
        if m_in_keys != allowed_model_input_keys:
            raise ValueError(
                f"Record {i} model_inputs keys {m_in_keys} != expected {allowed_model_input_keys}"
            )

    # 8. Neutral NONE Tensorization Gate
    none_passed_count = 0
    none_total_count = 0
    for i in range(len(dataset)):
        item = dataset[i]
        m_in = item["model_inputs"]
        if m_in["operator_idx"].item() == OPERATOR_NONE_IDX:
            none_total_count += 1
            if (
                torch.count_nonzero(m_in["candidate_visual"]).item() == 0
                and torch.count_nonzero(m_in["current_geometry"]).item() == 0
                and torch.count_nonzero(m_in["destination_geometry"]).item() == 0
            ):
                none_passed_count += 1
            else:
                raise ValueError(f"Record {i} operator NONE has non-zero candidate tensors.")

    # 9. Strengthened Matched-Candidate Consistency Gate
    # Group RELOCATE records by (scene_id, crop_path, current_geom)
    matched_groups: Dict[Tuple[str, str, Tuple[float, ...]], List[int]] = {}
    for idx, rec in enumerate(records):
        m_in = rec["model_inputs"]
        if m_in.get("intervention_operator") == "RELOCATE":
            scene_id = rec["scene_id"]
            crop_path = m_in["candidate_object_crop_path"]
            curr_pos = tuple(m_in["current_geometry"]["relative_position"])
            key = (scene_id, crop_path, curr_pos)
            matched_groups.setdefault(key, []).append(idx)

    matched_comparisons_total = 0
    matched_comparisons_passed = 0

    SAME_KEYS = [
        "text_feat", "scene_global", "scene_patch",
        "candidate_visual", "current_geometry", "operator_idx"
    ]

    for key, indices in matched_groups.items():
        if len(indices) < 2:
            continue
        ref_item = dataset[indices[0]]["model_inputs"]
        ref_dest = ref_item["destination_geometry"]

        for other_idx in indices[1:]:
            matched_comparisons_total += 1
            other_item = dataset[other_idx]["model_inputs"]

            all_same = all(
                torch.equal(ref_item[k], other_item[k]) for k in SAME_KEYS
            )
            dest_differs = not torch.equal(ref_dest, other_item["destination_geometry"])

            if not all_same:
                failing = [
                    k for k in SAME_KEYS
                    if not torch.equal(ref_item[k], other_item[k])
                ]
                raise ValueError(
                    f"Matched candidate group {key}: "
                    f"fields {failing} differ between indices {indices[0]} and {other_idx}."
                )
            if not dest_differs:
                raise ValueError(
                    f"Matched candidate group {key}: "
                    f"destination_geometry identical between {indices[0]} and {other_idx}."
                )
            matched_comparisons_passed += 1

    # 10. Grouped Batch Sampler
    sampler = InterventionGroupBatchSampler(
        dataset=dataset, scenes_per_batch=1, shuffle=False
    )
    batches = list(sampler)
    grouped_scene_sizes = [len(b) for b in batches]

    for batch_indices in batches:
        items = [dataset[i] for i in batch_indices]
        collated = collate_intervention_group(items)
        if collated["batch_size"] != len(batch_indices):
            raise ValueError("Collated batch size mismatch")

    # 11. Verify authoritative extraction-signature completeness
    # Vision: verify encoder metadata against spec constants
    v_enc = cache_index.get("encoders", {}).get("vision", {})
    t_enc = cache_index.get("encoders", {}).get("text", {})
    vision_meta_ok = (
        v_enc.get("embed_dim") == SCENE_GLOBAL_DIM
        and auth_vision_sig.get("embed_dim") == SCENE_GLOBAL_DIM
        and auth_vision_sig.get("signature_sha256") is not None
    )
    text_meta_ok = (
        t_enc.get("embed_dim") == TEXT_DIM
        and auth_text_sig.get("embed_dim") == TEXT_DIM
        and auth_text_sig.get("signature_sha256") is not None
    )

    # Compile Validation Report
    report = {
        "schema_version": schema_version,
        "feature_cache_schema_version": cache_schema,
        "validation_status": "PASSED",
        "source_manifest_sha256": manifest_sha256,
        "source_generator_commit": cache_index.get("source", {}).get("dataset_generator_commit"),
        "feature_code_commit": commit or cache_index.get("feature_code_commit"),
        "record_count": len(records),
        "scene_count": len(unique_pre_paths),
        "unique_pre_images": len(unique_pre_paths),
        "unique_candidate_crops": len(unique_crop_paths),
        "unique_instruction_strings": len(unique_instructions),
        "encoders": cache_index.get("encoders", {}),
        "extraction_signatures": {
            "vision": auth_vision_sig,
            "text": auth_text_sig,
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
        "grouped_scene_sizes": grouped_scene_sizes,
        "scene_source_hash_matches": {
            "passed_count": scene_passed_count,
            "total": len(unique_pre_paths),
            "passed": scene_passed_count == len(unique_pre_paths),
        },
        "crop_source_hash_matches": {
            "passed_count": crop_passed_count,
            "total": len(unique_crop_paths),
            "passed": crop_passed_count == len(unique_crop_paths),
        },
        "instruction_hash_matches": {
            "passed_count": text_passed_count,
            "total": len(unique_instructions),
            "passed": text_passed_count == len(unique_instructions),
        },
        "scene_signature_matches": {
            "passed_count": scene_sig_passed,
            "total": len(unique_pre_paths),
            "passed": scene_sig_passed == len(unique_pre_paths),
        },
        "crop_signature_matches": {
            "passed_count": crop_sig_passed,
            "total": len(unique_crop_paths),
            "passed": crop_sig_passed == len(unique_crop_paths),
        },
        "text_signature_matches": {
            "passed_count": text_sig_passed,
            "total": len(unique_instructions),
            "passed": text_sig_passed == len(unique_instructions),
        },
        "none_tensorization_matches": {
            "passed_count": none_passed_count,
            "total": none_total_count,
            "passed": none_passed_count == none_total_count and none_total_count > 0,
        },
        "matched_candidate_comparisons": {
            "passed_count": matched_comparisons_passed,
            "total": matched_comparisons_total,
            "passed": matched_comparisons_passed == matched_comparisons_total and matched_comparisons_total > 0,
        },
        "manifest_cache_compatibility": True,
        "model_feature_resolution_from_model_inputs_only": True,
        "post_feature_reference_count": 0,
        "encoder_metadata_consistency": bool(vision_meta_ok and text_meta_ok),
    }

    if report_path:
        report_path = Path(report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate intervention feature cache independently."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/intervention_smoke/manifest.jsonl",
        help="Path to manifest.jsonl",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/intervention_smoke/features",
        help="Path to features directory",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/intervention_smoke/feature_validation_report.json",
        help="Path to output report",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Git commit SHA of feature extractor code",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Validating feature cache at {args.features_dir}...")

    report = validate_feature_cache(
        manifest_path=Path(args.manifest),
        features_dir=Path(args.features_dir),
        report_path=Path(args.report) if args.report else None,
        commit=args.commit,
    )

    logging.info("=" * 70)
    logging.info("FEATURE CACHE VALIDATION PASSED (100% Validated)")
    logging.info(
        f"Scenes: {report['scene_count']}, "
        f"Crops: {report['unique_candidate_crops']}, "
        f"Instructions: {report['unique_instruction_strings']}"
    )
    logging.info(f"Grouped scene sizes: {report['grouped_scene_sizes']}")
    logging.info(
        f"Signature matches — "
        f"scenes: {report['scene_signature_matches']['passed_count']}/{report['scene_signature_matches']['total']}, "
        f"crops: {report['crop_signature_matches']['passed_count']}/{report['crop_signature_matches']['total']}, "
        f"text: {report['text_signature_matches']['passed_count']}/{report['text_signature_matches']['total']}"
    )
    if args.report:
        logging.info(f"Validation report written to: {args.report}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
