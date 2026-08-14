"""Leakage-safe Intervention PyTorch Dataset and Grouped Candidate Sampler.

This module establishes the interface between the validated physical intervention dataset
and neural learning models (V2/V3), ensuring strict information partitioning:
- Model inputs are strictly pre-intervention observations with ZERO privileged metadata access.
- Geometry is task-relative (meters).
- NONE interventions are neutrally tensorized with zero-padding.
- Feature cache provenance is verified against source manifest SHA256 at initialization.
- Samplers preserve complete candidate sets per scene for ranking loss with contiguity guarantees.
"""

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, Sampler

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


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class InterventionLearningDataset(Dataset):
    """PyTorch Dataset loading intervention records with strict leakage-safe partitioning."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        features_dir: Union[str, Path],
        split: str = "all",
        split_seed: int = 42,
        train_ratio: float = 0.8,
        return_privileged_for_eval: bool = False,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.features_dir = Path(features_dir).resolve()
        self.split = split
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.return_privileged_for_eval = return_privileged_for_eval

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        if not all_records:
            raise ValueError(f"Manifest at {self.manifest_path} contains no records.")

        # Load and Validate Feature Cache Index
        index_file = self.features_dir / "feature_cache_index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Missing feature cache index at {index_file}")

        with open(index_file, "r", encoding="utf-8") as f:
            self.cache_index = json.load(f)

        # Validate feature cache schema version
        cache_schema = self.cache_index.get("feature_cache_schema_version")
        if cache_schema != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Feature cache at {self.features_dir} has schema version '{cache_schema}', "
                f"expected exact '{FEATURE_CACHE_SCHEMA_VERSION}'."
            )

        # Provenance verification: Manifest SHA256 must match cached source
        actual_manifest_sha = _compute_file_sha256(self.manifest_path)
        cached_manifest_sha = self.cache_index.get("source", {}).get("manifest_sha256") or self.cache_index.get("manifest_sha256")

        if cached_manifest_sha != actual_manifest_sha:
            raise ValueError(
                f"Feature cache at {self.features_dir} was extracted from a different manifest.\n"
                f"Expected SHA256: {actual_manifest_sha}\n"
                f"Cached SHA256:   {cached_manifest_sha}"
            )

        # Partition by Scene ID to prevent candidate leakage across train/val splits
        unique_scene_ids = sorted(list(set(r["scene_id"] for r in all_records)))

        if split == "all":
            self.records = all_records
        elif split in ("train", "val"):
            rng = random.Random(split_seed)
            shuffled_scenes = list(unique_scene_ids)
            rng.shuffle(shuffled_scenes)
            split_idx = int(len(shuffled_scenes) * train_ratio)
            train_scenes = set(shuffled_scenes[:split_idx])
            val_scenes = set(shuffled_scenes[split_idx:])

            if split == "train":
                self.records = [r for r in all_records if r["scene_id"] in train_scenes]
            else:
                self.records = [r for r in all_records if r["scene_id"] in val_scenes]
        else:
            # Custom split tag in record if present
            self.records = [r for r in all_records if r.get("split") == split]

        # Load Text Features Cache
        text_feat_file = self.features_dir / "text_features.pt"
        if not text_feat_file.exists():
            raise FileNotFoundError(f"Missing text features at {text_feat_file}")

        text_cache = torch.load(text_feat_file, weights_only=True)
        if isinstance(text_cache, dict) and "features" in text_cache:
            self.text_features = text_cache["features"]
        else:
            self.text_features = text_cache

        # Feature specification
        self.spec = InterventionFeatureSpec()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        m_in = rec["model_inputs"]
        s_tar = rec["supervision_targets"]
        scene_id = rec["scene_id"]

        # 1. Text Feature (Model-Visible: model_inputs.instruction)
        instruction = m_in["instruction"]
        if instruction not in self.text_features:
            raise KeyError(f"Instruction '{instruction}' not found in cached text features.")
        text_feat = self.text_features[instruction].to(torch.float32)

        # 2. Pre-Scene Visual Features (Model-Visible: model_inputs.pre_rgb_path)
        pre_rgb_path = m_in.get("pre_rgb_path")
        if not pre_rgb_path:
            raise ValueError(f"Record {rec['record_id']}: Missing model_inputs.pre_rgb_path")

        scenes_map = self.cache_index.get("scenes", {})
        if pre_rgb_path not in scenes_map:
            raise KeyError(f"Pre RGB path '{pre_rgb_path}' missing from feature cache index at {self.features_dir}.")

        scene_feat_rel_path = scenes_map[pre_rgb_path]["feature_path"]
        scene_feat_file = self.features_dir / scene_feat_rel_path

        if not scene_feat_file.exists():
            raise FileNotFoundError(f"Missing scene features at {scene_feat_file}")

        scene_dict = torch.load(scene_feat_file, weights_only=True)
        scene_global = scene_dict["global"].to(torch.float32)
        scene_patch = scene_dict["patch"].to(torch.float32)

        # 3. Intervention Candidate Tensors (Model-Visible: operator, crop_path, current_geom, dest_geom)
        operator_str = m_in.get("intervention_operator", "NONE")

        if operator_str == "NONE":
            candidate_visual = torch.zeros(self.spec.candidate_visual_dim, dtype=torch.float32)
            current_geometry = torch.zeros(self.spec.current_geom_dim, dtype=torch.float32)
            destination_geometry = torch.zeros(self.spec.dest_geom_dim, dtype=torch.float32)
            operator_idx = torch.tensor(OPERATOR_NONE_IDX, dtype=torch.long)
        elif operator_str == "RELOCATE":
            crop_path = m_in.get("candidate_object_crop_path")
            if not crop_path:
                raise ValueError(f"Record {rec['record_id']}: RELOCATE operator missing candidate_object_crop_path")

            crops_map = self.cache_index.get("crops", {})
            if crop_path not in crops_map:
                raise KeyError(f"Candidate crop path '{crop_path}' missing from feature cache index at {self.features_dir}.")

            crop_feat_rel_path = crops_map[crop_path]["feature_path"]
            crop_feat_file = self.features_dir / crop_feat_rel_path

            if not crop_feat_file.exists():
                raise FileNotFoundError(f"Missing crop features at {crop_feat_file}")

            crop_dict = torch.load(crop_feat_file, weights_only=True)
            candidate_visual = crop_dict["global"].to(torch.float32)

            curr_geom_dict = m_in.get("current_geometry")
            if not curr_geom_dict or "relative_position" not in curr_geom_dict:
                raise ValueError(f"Record {rec['record_id']}: Missing current_geometry relative_position")
            current_geometry = torch.tensor(curr_geom_dict["relative_position"], dtype=torch.float32)

            dest_geom_dict = m_in.get("destination_geometry")
            if not dest_geom_dict or "relative_position" not in dest_geom_dict:
                raise ValueError(f"Record {rec['record_id']}: Missing destination_geometry relative_position")
            destination_geometry = torch.tensor(dest_geom_dict["relative_position"], dtype=torch.float32)

            operator_idx = torch.tensor(OPERATOR_RELOCATE_IDX, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported intervention operator: {operator_str}")

        # 4. Supervision Targets
        pre_feasible = torch.tensor(float(s_tar["pre_feasible"]), dtype=torch.float32)
        post_feasible = torch.tensor(float(s_tar["post_feasible"]), dtype=torch.float32)
        causal_effect = torch.tensor(int(s_tar["causal_effect"]), dtype=torch.long)

        item = {
            "model_inputs": {
                "text_feat": text_feat,
                "scene_global": scene_global,
                "scene_patch": scene_patch,
                "candidate_visual": candidate_visual,
                "current_geometry": current_geometry,
                "destination_geometry": destination_geometry,
                "operator_idx": operator_idx,
            },
            "supervision_targets": {
                "pre_feasible": pre_feasible,
                "post_feasible": post_feasible,
                "causal_effect": causal_effect,
            },
            "identifiers": {
                "scene_id": scene_id,
                "record_id": rec["record_id"],
                "intervention_id": rec["intervention_id"],
                "task_id": m_in["task_id"],
            },
        }

        if self.return_privileged_for_eval and "privileged_metadata" in rec:
            p = rec["privileged_metadata"]
            item["privileged_eval"] = {
                "intended_category": p.get("intended_category"),
                "is_culprit": p.get("is_culprit"),
                "active_culprits_before": p.get("active_culprits_before", []),
                "active_culprits_after": p.get("active_culprits_after", []),
            }

        return item


class InterventionGroupBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups candidate interventions by base scene.

    Guarantees that all candidate interventions belonging to a scene
    are yielded together in the same batch.
    """

    def __init__(
        self,
        dataset: InterventionLearningDataset,
        scenes_per_batch: int = 1,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.scenes_per_batch = scenes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group dataset indices by scene_id preserving sequential order
        scene_to_indices: Dict[str, List[int]] = {}
        for idx, rec in enumerate(self.dataset.records):
            s_id = rec["scene_id"]
            if s_id not in scene_to_indices:
                scene_to_indices[s_id] = []
            scene_to_indices[s_id].append(idx)

        self.scene_groups = list(scene_to_indices.values())

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic epoch-aware shuffling.

        For shuffle=True, the RNG is seeded with (self.seed + epoch),
        producing a different group order each epoch while remaining
        fully deterministic for any given (seed, epoch) pair.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.scene_groups) + self.scenes_per_batch - 1) // self.scenes_per_batch

    def __iter__(self):
        groups = list(self.scene_groups)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(groups)

        for i in range(0, len(groups), self.scenes_per_batch):
            batch_groups = groups[i : i + self.scenes_per_batch]
            batch_indices = [idx for group in batch_groups for idx in group]
            yield batch_indices


def collate_intervention_group(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for grouped candidate intervention batches.

    Guarantees contiguity of candidate groups within each batch.
    """
    m_in_list = [item["model_inputs"] for item in batch]
    s_tar_list = [item["supervision_targets"] for item in batch]
    ident_list = [item["identifiers"] for item in batch]

    scene_ids = [ident["scene_id"] for ident in ident_list]

    # Contiguity assertion: verify that each scene_id appears in exactly one contiguous block
    seen_scenes = set()
    last_scene = None
    for s_id in scene_ids:
        if s_id != last_scene:
            if s_id in seen_scenes:
                raise ValueError(
                    f"Non-contiguous scene group detected in batch: scene '{s_id}' appears in multiple separated blocks."
                )
            seen_scenes.add(s_id)
            last_scene = s_id

    collated_model_inputs = {
        "text_feat": torch.stack([m["text_feat"] for m in m_in_list]),
        "scene_global": torch.stack([m["scene_global"] for m in m_in_list]),
        "scene_patch": torch.stack([m["scene_patch"] for m in m_in_list]),
        "candidate_visual": torch.stack([m["candidate_visual"] for m in m_in_list]),
        "current_geometry": torch.stack([m["current_geometry"] for m in m_in_list]),
        "destination_geometry": torch.stack([m["destination_geometry"] for m in m_in_list]),
        "operator_idx": torch.stack([m["operator_idx"] for m in m_in_list]),
    }

    collated_supervision = {
        "pre_feasible": torch.stack([s["pre_feasible"] for s in s_tar_list]),
        "post_feasible": torch.stack([s["post_feasible"] for s in s_tar_list]),
        "causal_effect": torch.stack([s["causal_effect"] for s in s_tar_list]),
    }

    # Compute scene group pointers
    scene_group_ptrs = [0]
    current_scene = scene_ids[0]
    for i, s_id in enumerate(scene_ids):
        if s_id != current_scene:
            scene_group_ptrs.append(i)
            current_scene = s_id
    scene_group_ptrs.append(len(scene_ids))

    collated = {
        "model_inputs": collated_model_inputs,
        "supervision_targets": collated_supervision,
        "identifiers": {
            "scene_ids": scene_ids,
            "record_ids": [ident["record_id"] for ident in ident_list],
            "intervention_ids": [ident["intervention_id"] for ident in ident_list],
            "task_ids": [ident["task_id"] for ident in ident_list],
        },
        "scene_group_ptrs": torch.tensor(scene_group_ptrs, dtype=torch.long),
        "batch_size": len(batch),
    }

    if "privileged_eval" in batch[0]:
        collated["privileged_eval"] = [item.get("privileged_eval", {}) for item in batch]

    return collated
