"""
Dataset validator script to verify dataset integrity, counterfactual symmetry, and predicate consistency.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
import cv2
import numpy as np
import PIL.Image as Image


class DatasetValidator:
    """Validator class for verifying benchmark dataset compliance."""

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}")

    def load_manifest(self) -> List[Dict[str, dict]]:
        """Load manifest JSONL file records.
        
        Returns:
            List of metadata records.
        """
        records = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """Run complete validation suite.
        
        Returns:
            Tuple of (is_valid: bool, log_messages: List[str]).
        """
        messages = []
        is_valid = True
        records = self.load_manifest()

        if not records:
            messages.append("ERROR: Manifest file is empty!")
            return False, messages

        messages.append(f"Loaded {len(records)} counterfactual pair records from manifest.")

        for idx, rec in enumerate(records):
            pair_id = rec.get("pair_id", f"record_{idx}")

            # 1. File existence verification
            for mode in ["stop", "proceed"]:
                sub = rec[mode]
                rgb_p = Path(sub["rgb_path"])
                mask_p = Path(sub["culprit_mask_path"])
                region_p = Path(sub["region_mask_path"])

                if not rgb_p.exists():
                    messages.append(f"[{pair_id}] {mode} RGB image missing: {rgb_p}")
                    is_valid = False
                if not mask_p.exists():
                    messages.append(f"[{pair_id}] {mode} culprit mask missing: {mask_p}")
                    is_valid = False
                if not region_p.exists():
                    messages.append(f"[{pair_id}] {mode} region mask missing: {region_p}")
                    is_valid = False

                # 2. Check mask values
                if mask_p.exists():
                    mask_img = np.array(Image.open(mask_p))
                    if mode == "stop" and np.max(mask_img) == 0:
                        messages.append(f"[{pair_id}] STOP culprit mask is empty (0 non-zero pixels)!")
                        is_valid = False

            # 3. Label consistency checks
            if not rec["stop"]["is_occupied"]:
                messages.append(f"[{pair_id}] STOP state predicate check failed: expected occupied=True")
                is_valid = False

            if rec["proceed"]["is_occupied"]:
                messages.append(f"[{pair_id}] PROCEED state predicate check failed: expected occupied=False")
                is_valid = False

        if is_valid:
            messages.append("SUCCESS: All dataset validation checks passed cleanly!")
        else:
            messages.append("FAILURE: Dataset validation failed with errors.")

        return is_valid, messages


def validate_manifest(manifest_path: str) -> bool:
    """CLI entry point helper to run validation and print report.
    
    Args:
        manifest_path: Path to manifest JSONL file.
        
    Returns:
        True if valid, False otherwise.
    """
    validator = DatasetValidator(manifest_path)
    is_valid, logs = validator.validate_dataset()
    for log in logs:
        print(log)
    return is_valid


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/manifests/smoke_manifest.jsonl"
    success = validate_manifest(path)
    sys.exit(0 if success else 1)
