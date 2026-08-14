#!/usr/bin/env python3
"""Precompute and cache frozen vision and text features for intervention datasets.

Features extracted:
1. Unique Pre-Scene Visual Features (DINOv2 global [768] + 256x768 patch tokens)
2. Unique Candidate Object Crop Visual Features (DINOv2 global [768])
3. Unique Instruction Text Features (all-MiniLM-L6-v2 [384])
4. Feature Cache Index (Schema 1.1.0) with source-hash provenance tracking.
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
    FEATURE_CACHE_SCHEMA_VERSION,
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


def hash_cache_filename(rel_path: str, prefix: str) -> str:
    """Compute a neutral, injective SHA256-based cache filename from relative source path."""
    digest = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}_features.pt"


def get_vision_extraction_signature(vision_encoder: Any) -> Dict[str, Any]:
    """Return explicit visual feature extraction signature dictionary."""
    model_name = getattr(vision_encoder, "model_name", "dinov2_vitb14")
    embed_dim = SCENE_GLOBAL_DIM
    patch_size = getattr(vision_encoder, "patch_size", 14)
    num_patches = SCENE_PATCH_SHAPE[0]
    transform_str = "bicubic_224_centercrop_imagenet_norm"
    sig_str = f"{model_name}:{embed_dim}:{patch_size}:{num_patches}:{transform_str}"
    sig_sha = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
    return {
        "model_name": model_name,
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "num_patches": num_patches,
        "transform": transform_str,
        "signature_sha256": sig_sha,
    }


def get_text_extraction_signature(text_encoder: Any) -> Dict[str, Any]:
    """Return explicit text feature extraction signature dictionary."""
    model_name = getattr(text_encoder, "model_name", "sentence-transformers/all-MiniLM-L6-v2")
    embed_dim = TEXT_DIM
    pooling_str = "mean_attention_mask"
    sig_str = f"{model_name}:{embed_dim}:{pooling_str}"
    sig_sha = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
    return {
        "model_name": model_name,
        "embed_dim": embed_dim,
        "pooling": pooling_str,
        "signature_sha256": sig_sha,
    }


def extract_intervention_features(
    manifest_path: Path,
    dataset_root: Path,
    out_dir: Path,
    device: Optional[str] = None,
    force: bool = False,
    commit: Optional[str] = None,
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

    # Read generator commit from dataset_metadata.json if present
    metadata_path = dataset_root / "dataset_metadata.json"
    generator_commit = None
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            generator_commit = meta.get("generator_commit")

    # Group unique pre scenes, candidate crops, and text instructions strictly from model_inputs
    unique_scenes: Dict[str, Tuple[str, str]] = {}  # pre_rgb_path -> (scene_id, cache_filename)
    unique_crops: Dict[str, str] = {}  # candidate_crop_path -> crop_filename
    unique_instructions: Dict[str, str] = {}  # text -> text_sha256

    for rec in records:
        m_in = rec["model_inputs"]
        scene_id = rec["scene_id"]

        # Scene pre RGB (model-visible)
        pre_rgb_path = m_in.get("pre_rgb_path")
        if pre_rgb_path and pre_rgb_path not in unique_scenes:
            unique_scenes[pre_rgb_path] = (scene_id, hash_cache_filename(pre_rgb_path, "scene"))

        # Candidate crop (model-visible)
        crop_path = m_in.get("candidate_object_crop_path")
        if crop_path and crop_path not in unique_crops:
            unique_crops[crop_path] = hash_cache_filename(crop_path, "crop")

        # Instruction (model-visible)
        inst = m_in.get("instruction")
        if inst and inst not in unique_instructions:
            unique_instructions[inst] = compute_text_sha256(inst)

    # Initialize encoders on demand
    if vision_encoder is None:
        vision_encoder = VisionEncoder(model_name="dinov2_vitb14", device=device)
    if text_encoder is None:
        text_encoder = TextEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    vision_sig = get_vision_extraction_signature(vision_encoder)
    text_sig = get_text_extraction_signature(text_encoder)

    # 1. Extract Unique Pre Scene Visual Features
    scene_index: Dict[str, Any] = {}
    for rel_path, (scene_id, cache_filename) in unique_scenes.items():
        scene_feat_file = out_dir / "scenes" / cache_filename
        abs_img_path = dataset_root / rel_path

        if not abs_img_path.exists():
            raise FileNotFoundError(f"Missing pre RGB image at {abs_img_path}")

        img_sha = compute_file_sha256(abs_img_path)

        needs_recompute = force or not scene_feat_file.exists()
        if not needs_recompute:
            try:
                cached = torch.load(scene_feat_file, weights_only=True)
                cached_sig = cached.get("extraction_signature", {}).get("signature_sha256")
                if (
                    cached.get("source_sha256") != img_sha
                    or cached_sig != vision_sig["signature_sha256"]
                    or cached["global"].shape != (SCENE_GLOBAL_DIM,)
                    or cached["patch"].shape != SCENE_PATCH_SHAPE
                ):
                    needs_recompute = True
            except Exception:
                needs_recompute = True

        if needs_recompute:
            img = Image.open(abs_img_path).convert("RGB")
            cls_token, patch_tokens = vision_encoder([img])
            torch.save(
                {
                    "scene_id": scene_id,
                    "source_path": rel_path,
                    "source_sha256": img_sha,
                    "extraction_signature": vision_sig,
                    "global": cls_token[0].cpu().to(torch.float32),
                    "patch": patch_tokens[0].cpu().to(torch.float32),
                },
                scene_feat_file,
            )

        scene_index[rel_path] = {
            "scene_id": scene_id,
            "source_path": rel_path,
            "source_sha256": img_sha,
            "extraction_signature": vision_sig,
            "feature_path": str(scene_feat_file.relative_to(out_dir)),
            "global_shape": [SCENE_GLOBAL_DIM],
            "patch_shape": list(SCENE_PATCH_SHAPE),
        }

    # 2. Extract Unique Candidate Crop Visual Features
    crop_index: Dict[str, Any] = {}
    for rel_crop_path, cache_filename in unique_crops.items():
        crop_feat_file = out_dir / "crops" / cache_filename
        abs_crop_path = dataset_root / rel_crop_path

        if not abs_crop_path.exists():
            raise FileNotFoundError(f"Missing candidate crop image at {abs_crop_path}")

        crop_sha = compute_file_sha256(abs_crop_path)

        needs_recompute = force or not crop_feat_file.exists()
        if not needs_recompute:
            try:
                cached = torch.load(crop_feat_file, weights_only=True)
                cached_sig = cached.get("extraction_signature", {}).get("signature_sha256")
                if (
                    cached.get("source_sha256") != crop_sha
                    or cached_sig != vision_sig["signature_sha256"]
                    or cached["global"].shape != (CANDIDATE_VISUAL_DIM,)
                ):
                    needs_recompute = True
            except Exception:
                needs_recompute = True

        if needs_recompute:
            crop_img = Image.open(abs_crop_path).convert("RGB")
            cls_token, _ = vision_encoder([crop_img])
            torch.save(
                {
                    "source_path": rel_crop_path,
                    "source_sha256": crop_sha,
                    "extraction_signature": vision_sig,
                    "global": cls_token[0].cpu().to(torch.float32),
                },
                crop_feat_file,
            )

        crop_index[rel_crop_path] = {
            "source_path": rel_crop_path,
            "source_sha256": crop_sha,
            "extraction_signature": vision_sig,
            "feature_path": str(crop_feat_file.relative_to(out_dir)),
            "global_shape": [CANDIDATE_VISUAL_DIM],
        }

    # 3. Extract Unique Instruction Text Features
    text_feat_file = out_dir / "text_features.pt"
    instruction_list = list(unique_instructions.keys())

    needs_recompute_text = force or not text_feat_file.exists()
    if not needs_recompute_text:
        try:
            cached_text = torch.load(text_feat_file, weights_only=True)
            cached_sig = cached_text.get("metadata", {}).get("extraction_signature", {}).get("signature_sha256")
            if cached_sig != text_sig["signature_sha256"]:
                needs_recompute_text = True
            else:
                feats = cached_text.get("features", {})
                for inst in instruction_list:
                    if inst not in feats or feats[inst].shape != (TEXT_DIM,):
                        needs_recompute_text = True
                        break
        except Exception:
            needs_recompute_text = True

    if needs_recompute_text:
        text_embeddings = text_encoder(instruction_list)
        text_features_dict = {
            inst: text_embeddings[i].cpu().to(torch.float32) for i, inst in enumerate(instruction_list)
        }
        torch.save(
            {
                "features": text_features_dict,
                "metadata": {
                    "encoder": text_sig["model_name"],
                    "embed_dim": TEXT_DIM,
                    "extraction_signature": text_sig,
                    "instructions": [
                        {"text": inst, "sha256": unique_instructions[inst]}
                        for inst in instruction_list
                    ],
                },
            },
            text_feat_file,
        )

    text_index = {
        inst: {
            "exact_text": inst,
            "text_sha256": unique_instructions[inst],
            "feature_dim": TEXT_DIM,
            "extraction_signature": text_sig,
        }
        for inst in instruction_list
    }

    # 4. Write Authoritative Feature Cache Index (Schema 1.1.0)
    cache_index = {
        "schema_version": schema_version,
        "feature_cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "source": {
            "manifest_path": str(manifest_path.name),
            "manifest_sha256": manifest_sha256,
            "dataset_generator_commit": generator_commit,
        },
        "feature_code_commit": commit,
        "extraction_signatures": {
            "vision": vision_sig,
            "text": text_sig,
        },
        "encoders": {
            "vision": {
                "model_name": vision_sig["model_name"],
                "embed_dim": SCENE_GLOBAL_DIM,
                "patch_size": vision_sig["patch_size"],
                "num_patches": SCENE_PATCH_SHAPE[0],
                "transform": vision_sig["transform"],
            },
            "text": {
                "model_name": text_sig["model_name"],
                "embed_dim": TEXT_DIM,
                "pooling": text_sig["pooling"],
            },
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
        "scene_count": len(unique_scenes),
        "crop_count": len(unique_crops),
        "instruction_count": len(unique_instructions),
        "record_count": len(records),
        "scenes": scene_index,
        "crops": crop_index,
        "texts": text_index,
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
    parser.add_argument("--commit", type=str, default=None, help="Git commit SHA of feature extractor code")
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
        commit=args.commit,
    )

    logging.info("Feature extraction complete!")
    logging.info(f"Unique Scenes: {cache_index['scene_count']}, Unique Crops: {cache_index['crop_count']}, Instructions: {cache_index['instruction_count']}")
    logging.info(f"Feature Cache Index written to: {Path(args.out_dir) / 'feature_cache_index.json'}")


if __name__ == "__main__":
    main()
