#!/usr/bin/env python3
"""Precompute and cache frozen vision and text features for intervention datasets.

Features extracted:
1. Unique Pre-Scene Visual Features (DINOv2 global [768] + 256x768 patch tokens)
2. Unique Candidate Object Crop Visual Features (DINOv2 global [768])
3. Unique Instruction Text Features (all-MiniLM-L6-v2 [384])
4. Feature Cache Index with source-hash provenance tracking.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import torch

from src.learning.intervention_feature_spec import (
    InterventionFeatureSpec,
    TEXT_DIM,
    SCENE_GLOBAL_DIM,
    SCENE_PATCH_SHAPE,
    CANDIDATE_VISUAL_DIM,
    CURRENT_GEOM_DIM,
    DEST_GEOM_DIM,
)
from src.learning.text_encoder import TextEncoder
from src.learning.vision_encoder import VisionEncoder


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


def extract_intervention_features(
    manifest_path: Path,
    dataset_root: Path,
    out_dir: Path,
    device: Optional[str] = None,
    force: bool = False,
    vision_encoder: Optional[Any] = None,
    text_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract and cache frozen vision and text features for an intervention manifest."""
    manifest_path = Path(manifest_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"Manifest at {manifest_path} is empty.")

    manifest_sha256 = compute_file_sha256(manifest_path)
    schema_version = records[0].get("schema_version", "2.1.0")

    # Group unique pre scenes, candidate crops, and text instructions
    unique_scenes: Dict[str, str] = {}  # scene_id -> rel_pre_rgb_path
    unique_crops: Dict[Tuple[str, str], str] = {}  # (scene_id, obj_name) -> rel_crop_path
    unique_instructions: Dict[str, str] = {}  # text -> text_sha256

    for rec in records:
        m_in = rec["model_inputs"]
        p_meta = rec.get("privileged_metadata", {})
        scene_id = rec["scene_id"]

        # Scene pre RGB
        pre_rgb_path = m_in.get("pre_rgb_path")
        if pre_rgb_path and scene_id not in unique_scenes:
            unique_scenes[scene_id] = pre_rgb_path

        # Candidate crop
        crop_path = m_in.get("candidate_object_crop_path")
        if crop_path:
            obj_name = p_meta.get("candidate_object_name")
            if not obj_name:
                # Extract from crop filename if privileged_metadata omitted
                obj_name = Path(crop_path).stem.replace("object_", "")
            key = (scene_id, obj_name)
            if key not in unique_crops:
                unique_crops[key] = crop_path

        # Instruction
        inst = m_in.get("instruction")
        if inst and inst not in unique_instructions:
            unique_instructions[inst] = compute_text_sha256(inst)

    # Initialize encoders
    if vision_encoder is None:
        vision_encoder = VisionEncoder(model_name="dinov2_vitb14", device=device)
    if text_encoder is None:
        text_encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    # 1. Extract Unique Pre Scene Visual Features
    scene_feature_paths: Dict[str, str] = {}
    for scene_id, rel_path in unique_scenes.items():
        scene_feat_file = out_dir / "scenes" / f"scene_{scene_id}_features.pt"
        abs_img_path = dataset_root / rel_path

        if not abs_img_path.exists():
            raise FileNotFoundError(f"Missing pre RGB image at {abs_img_path}")

        img_sha = compute_file_sha256(abs_img_path)

        if not scene_feat_file.exists() or force:
            img = Image.open(abs_img_path).convert("RGB")
            cls_token, patch_tokens = vision_encoder([img])
            torch.save(
                {
                    "scene_id": scene_id,
                    "source_path": rel_path,
                    "source_sha256": img_sha,
                    "global": cls_token[0].cpu().to(torch.float32),
                    "patch": patch_tokens[0].cpu().to(torch.float32),
                },
                scene_feat_file,
            )
        scene_feature_paths[scene_id] = str(scene_feat_file.relative_to(out_dir))

    # 2. Extract Unique Candidate Crop Visual Features
    crop_feature_paths: Dict[str, str] = {}
    for (scene_id, obj_name), rel_path in unique_crops.items():
        crop_feat_file = out_dir / "crops" / f"crop_{scene_id}_{obj_name}_features.pt"
        abs_crop_path = dataset_root / rel_path

        if not abs_crop_path.exists():
            raise FileNotFoundError(f"Missing candidate crop image at {abs_crop_path}")

        crop_sha = compute_file_sha256(abs_crop_path)

        if not crop_feat_file.exists() or force:
            crop_img = Image.open(abs_crop_path).convert("RGB")
            cls_token, _ = vision_encoder([crop_img])
            torch.save(
                {
                    "scene_id": scene_id,
                    "object_name": obj_name,
                    "source_path": rel_path,
                    "source_sha256": crop_sha,
                    "global": cls_token[0].cpu().to(torch.float32),
                },
                crop_feat_file,
            )
        crop_feature_paths[f"{scene_id}_{obj_name}"] = str(crop_feat_file.relative_to(out_dir))

    # 3. Extract Unique Instruction Text Features
    text_feat_file = out_dir / "text_features.pt"
    if not text_feat_file.exists() or force:
        instruction_list = list(unique_instructions.keys())
        text_embeddings = text_encoder(instruction_list)  # (N, 384)
        text_features_dict = {
            inst: text_embeddings[i].cpu().to(torch.float32) for i, inst in enumerate(instruction_list)
        }
        torch.save(
            {
                "features": text_features_dict,
                "metadata": {
                    "encoder": getattr(text_encoder, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                    "embed_dim": TEXT_DIM,
                    "instructions": [
                        {"text": inst, "sha256": unique_instructions[inst]}
                        for inst in instruction_list
                    ],
                },
            },
            text_feat_file,
        )

    # 4. Write Feature Cache Index
    cache_index = {
        "schema_version": schema_version,
        "feature_cache_schema_version": "1.0.0",
        "manifest_sha256": manifest_sha256,
        "scene_count": len(unique_scenes),
        "crop_count": len(unique_crops),
        "instruction_count": len(unique_instructions),
        "record_count": len(records),
        "dimensions": {
            "text_dim": TEXT_DIM,
            "scene_global_dim": SCENE_GLOBAL_DIM,
            "scene_patch_shape": list(SCENE_PATCH_SHAPE),
            "candidate_visual_dim": CANDIDATE_VISUAL_DIM,
            "current_geom_dim": CURRENT_GEOM_DIM,
            "dest_geom_dim": DEST_GEOM_DIM,
            "operator_dim": 1,
        },
        "scene_features": scene_feature_paths,
        "crop_features": crop_feature_paths,
        "text_features_path": str(text_feat_file.relative_to(out_dir)),
    }

    index_file = out_dir / "feature_cache_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(cache_index, f, indent=2)

    return cache_index


def main():
    parser = argparse.ArgumentParser(description="Precompute frozen features for intervention dataset.")
    parser.add_argument("--manifest", type=str, default="data/intervention_smoke/manifest.jsonl", help="Path to manifest.jsonl")
    parser.add_argument("--dataset-root", type=str, default="data/intervention_smoke", help="Root directory of the dataset")
    parser.add_argument("--out-dir", type=str, default="data/intervention_smoke/features", help="Output directory for features")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cuda/cpu)")
    parser.add_argument("--force", action="store_true", help="Force recomputation of features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Extracting intervention features from {args.manifest}...")

    cache_index = extract_intervention_features(
        manifest_path=Path(args.manifest),
        dataset_root=Path(args.dataset_root),
        out_dir=Path(args.out_dir),
        device=args.device,
        force=args.force,
    )

    logging.info("Feature extraction complete!")
    logging.info(f"Unique Scenes: {cache_index['scene_count']}, Unique Crops: {cache_index['crop_count']}, Instructions: {cache_index['instruction_count']}")
    logging.info(f"Feature Cache Index written to: {Path(args.out_dir) / 'feature_cache_index.json'}")


if __name__ == "__main__":
    main()
