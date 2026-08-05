"""
Pilot generation script to generate full pilot benchmark dataset:
- 3 genuine robot demonstration videos per task family (6 total)
- 60 matched PROCEED/STOP counterfactual query pairs per task family (120 pairs = 240 query images)
- Dataset validation & preview generation
"""

import os
import sys
from pathlib import Path
import json
import time

sys.path.insert(0, os.path.abspath("."))

from src.generation.demonstration_generator import DemonstrationGenerator
from src.generation.query_generator import QueryGenerator
from src.validation.dataset_validator import DatasetValidator
from src.preview.html_preview import HTMLPreviewGenerator
from src.preview.contact_sheet import ContactSheetGenerator


def run_pilot():
    print("=======================================================")
    print(" Starting MuJoCo Relational Precondition Pilot Generation ")
    print("=======================================================")
    start_time = time.time()

    # 1. Generate Demonstration Videos (3 per task family)
    print("\n[Phase 1/4] Generating Pilot Demonstration Videos (3 per task family)...")
    demo_gen = DemonstrationGenerator(output_dir="data/pilot_demos")
    demo_paths = []

    for i in range(1, 4):
        p1 = demo_gen.generate_task_1_demo(f"demo_task1_pilot_{i:03d}")
        demo_paths.append(p1)

    for i, obj in enumerate(["coffee_can", "sugar_box", "mug"], start=1):
        p2 = demo_gen.generate_task_2_demo(f"demo_task2_pilot_{i:03d}", obj_name=obj)
        demo_paths.append(p2)

    print(f"Generated {len(demo_paths)} pilot demonstration videos.")

    # 2. Generate Matched Query Pairs (60 per task family = 120 pairs = 240 images)
    print("\n[Phase 2/4] Generating Matched Pilot Query Pairs (120 total pairs)...")
    query_gen = QueryGenerator(config_path="configs/pilot.yaml")
    records = query_gen.run_generation()
    print(f"Generated {len(records)} counterfactual query pair records.")

    # 3. Validate Pilot Dataset
    print("\n[Phase 3/4] Running Dataset Validator on Pilot Manifest...")
    manifest_path = "data/manifests/pilot_manifest.jsonl"
    validator = DatasetValidator(manifest_path)
    is_valid, logs = validator.validate_dataset()
    for log in logs:
        print(log)

    if not is_valid:
        raise RuntimeError("Pilot dataset validation failed!")

    # 4. Generate Previews
    print("\n[Phase 4/4] Generating Pilot HTML & Contact Sheet Previews...")
    html_gen = HTMLPreviewGenerator(output_dir="data/previews")
    html_path = html_gen.generate_html_report(manifest_path, output_filename="pilot_preview.html")

    cs_gen = ContactSheetGenerator(output_dir="data/previews")
    cs_path = cs_gen.generate_contact_sheet(manifest_path, output_filename="pilot_contact_sheet.png")

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f" Pilot Generation Completed Successfully in {elapsed:.1f}s! ")
    print(f"   Manifest: {manifest_path}")
    print(f"   HTML Report: {html_path}")
    print(f"   Contact Sheet: {cs_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    run_pilot()
