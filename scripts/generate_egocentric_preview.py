#!/usr/bin/env python3
"""Generate compact preview dataset with canonical egocentric camera rigs."""
import json
from pathlib import Path
from src.generation.counterfactual_generator import CounterfactualPairGenerator

def main():
    output_dir = Path("output/egocentric_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = CounterfactualPairGenerator(output_dir=output_dir, resolution=(640, 480))

    print("Generating Task 1 matched pairs...")
    gen.generate_task1_pair("t1_pair_101", seed=101)
    gen.generate_task1_pair("t1_pair_102", seed=102)

    print("Generating Task 2 matched pairs...")
    gen.generate_task2_pair("t2_pair_201", seed=201)
    gen.generate_task2_pair("t2_pair_202", seed=202)

    print("Generating Task 1 positive controls...")
    gen.generate_task1_control("t1_ctrl_301", control_subtype="one_object_beside", seed=301)
    gen.generate_task1_control("t1_ctrl_302", control_subtype="two_objects_beside", seed=302)

    print("Generating Task 2 positive controls...")
    gen.generate_task2_control("t2_ctrl_401", control_subtype="one_object_beside_target", seed=401)
    gen.generate_task2_control("t2_ctrl_402", control_subtype="multiple_distractors_outside", seed=402)

    print("Preview dataset generation complete!")

    # Verify generated directories
    expected_dirs = [
        "t1_pair_101", "t1_pair_102",
        "t2_pair_201", "t2_pair_202",
        "t1_ctrl_301", "t1_ctrl_302",
        "t2_ctrl_401", "t2_ctrl_402",
    ]

    for d in expected_dirs:
        dir_path = output_dir / d
        meta_path = dir_path / "metadata.json"
        assert meta_path.is_file(), f"Missing metadata in {d}"
        with open(meta_path) as f:
            meta = json.load(f)
        camera_name = meta.get("camera_name") or meta.get("stop", {}).get("camera", {}).get("name")
        print(f"Verified {d}: camera_name = {camera_name}")

if __name__ == "__main__":
    main()
