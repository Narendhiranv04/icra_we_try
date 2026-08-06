#!/usr/bin/env python3
"""
Release verification script for Relational Precondition Benchmark.

Verifies:
1. Candidate tested code commit exists.
2. No source code changed after tested code commit.
3. All required smoke reports exist and passed.
4. All required pilot reports exist and passed.
5. Exact pilot counts are validated.
6. All 8 control subtypes are represented.
7. Pilot reproducibility tested all required records.
Exports data/reports/release_verification.json.
"""

import sys
import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

REQUIRED_SMOKE_REPORTS = [
    "pytest_results.xml",
    "test_summary.json",
    "dataset_validation.json",
    "split_validation.json",
    "reproducibility_report.json",
    "demonstration_validation.json",
    "demonstration_distinctness.json",
    "control_distribution.json",
]

REQUIRED_PILOT_REPORTS = [
    "pilot_validation.json",
    "pilot_split_validation.json",
    "pilot_reproducibility.json",
    "pilot_distribution.json",
    "pilot_demo_distinctness.json",
    "pilot_demonstration_validation.json",
    "pilot_control_distribution.json",
]

REQUIRED_SMOKE_ARTIFACT_REPORT = "smoke_report.json"

# Exact pilot counts
PILOT_MATCHED_PAIRS = 120
PILOT_POSITIVE_CONTROLS = 32
PILOT_TOTAL_RECORDS = 152
PILOT_DEMONSTRATIONS = 6
PILOT_TASK1_DEMOS = 3
PILOT_TASK2_DEMOS = 3
PILOT_REPRODUCIBILITY_RECORD_COUNT = 152

REQUIRED_CONTROL_SUBTYPES = {
    "task_1": ["empty_lid", "one_object_beside", "two_objects_beside", "near_lid_outside_footprint"],
    "task_2": ["empty_target", "one_object_beside_target", "one_object_near_target_outside", "multiple_distractors_outside"],
}


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
    return res.stdout.strip()


def _check_report_status(r_path: Path, r_name: str, issues: List[str]) -> str:
    """Check a JSON report file for presence and PASSED status. Returns status string."""
    if not r_path.exists():
        issues.append(f"Required report missing: {r_name}")
        return "MISSING"
    try:
        with open(r_path, "r", encoding="utf-8") as f:
            r_data = json.load(f)
        st = r_data.get("status", "MISSING_STATUS")
        if st != "PASSED":
            issues.append(f"Required report '{r_name}' has status '{st}' (expected PASSED)")
        return st
    except Exception as err:
        issues.append(f"Failed to parse report '{r_name}': {err}")
        return "CORRUPT"


def verify_release_state(
    tested_commit: str = None,
    reports_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> bool:
    """
    Verify the full release state including smoke and pilot evidence.

    Args:
        tested_commit: The git commit SHA of the tested code.
        reports_dir: Path to reports directory (default: data/reports relative to repo_root).
        artifacts_dir: Path to smoke artifacts directory (default: artifacts/smoke relative to repo_root).
        repo_root: Root of the repository (default: current directory).

    Returns:
        True if all checks pass, False otherwise.
    """
    if repo_root is None:
        repo_root = Path(".")
    if reports_dir is None:
        reports_dir = repo_root / "data" / "reports"
    if artifacts_dir is None:
        artifacts_dir = repo_root / "artifacts" / "smoke"

    reports_dir = Path(reports_dir)
    artifacts_dir = Path(artifacts_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    issues: List[str] = []
    # Resolve git repository root
    git_root = repo_root
    while git_root and git_root != git_root.parent:
        if (git_root / ".git").exists():
            break
        git_root = git_root.parent
    if not git_root or not (git_root / ".git").exists():
        git_root = Path(".")
    cwd_str = str(git_root)

    # 1. Resolve tested_code_commit
    if not tested_commit:
        tested_commit = os.environ.get("TESTED_CODE_COMMIT")
    if not tested_commit:
        try:
            tested_commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd_str)
        except Exception:
            tested_commit = "unknown"

    try:
        current_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd_str)
    except Exception:
        current_head = "unknown"

    # 2. Check if tested commit exists in git
    try:
        run_cmd(["git", "rev-parse", "--verify", tested_commit], cwd=cwd_str)
    except Exception:
        issues.append(f"Tested commit '{tested_commit}' is invalid or does not exist in git log")

    # 3. Check diff between tested_commit and current_head for source code modifications
    code_changed_after = False
    try:
        diff_output = run_cmd(["git", "diff", "--name-only", tested_commit, "HEAD"], cwd=cwd_str)
        changed_files = [f.strip() for f in diff_output.splitlines() if f.strip()]
        for f in changed_files:
            if f.startswith(("src/", "scripts/", "tests/", "configs/")):
                code_changed_after = True
                issues.append(f"Source code file '{f}' was modified after tested_code_commit '{tested_commit}'")
    except Exception as err:
        issues.append(f"Failed to compute git diff from tested commit: {err}")

    # 4. Check working tree status
    try:
        tree_status = run_cmd(["git", "status", "--porcelain"], cwd=cwd_str)
        working_tree_clean = (len(tree_status) == 0)
    except Exception:
        tree_status = ""
        working_tree_clean = False

    # 5. Check all required SMOKE reports
    smoke_report_statuses: Dict[str, str] = {}
    for r_name in REQUIRED_SMOKE_REPORTS:
        r_path = reports_dir / r_name
        if r_name == "pytest_results.xml":
            # XML validity is checked via test_summary.json; just verify existence
            if r_path.exists():
                smoke_report_statuses[r_name] = "PRESENT"
            else:
                issues.append(f"Required smoke report missing: {r_name}")
                smoke_report_statuses[r_name] = "MISSING"
        else:
            st = _check_report_status(r_path, r_name, issues)
            smoke_report_statuses[r_name] = st

    # 6. Check smoke_report.json in artifacts dir
    smoke_artifact_p = artifacts_dir / REQUIRED_SMOKE_ARTIFACT_REPORT
    if smoke_artifact_p.exists():
        try:
            with open(smoke_artifact_p, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            s_commit = s_data.get("tested_code_commit")
            if s_commit != tested_commit:
                issues.append(
                    f"smoke_report.json tested_code_commit '{s_commit}' != '{tested_commit}'"
                )
            s_status = s_data.get("status", "MISSING_STATUS")
            smoke_report_statuses[REQUIRED_SMOKE_ARTIFACT_REPORT] = s_status
            if s_status != "PASSED":
                issues.append(f"smoke_report.json has status '{s_status}'")

            # Validate smoke test count from smoke_report.json
            smoke_counts = {
                "matched_pairs": s_data.get("counterfactual_pairs_generated"),
                "positive_controls": s_data.get("positive_controls_generated"),
                "unit_tests_total": s_data.get("unit_tests_total"),
                "unit_tests_passed": s_data.get("unit_tests_passed"),
            }

            # Load configs/smoke.yaml and validate smoke counts
            try:
                import yaml
                cfg_path = repo_root / "configs" / "smoke.yaml"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        smoke_cfg = yaml.safe_load(f)
                    q_gen = smoke_cfg.get("query_generation", {})
                    num_tasks = len(smoke_cfg.get("tasks", []))
                    exp_pairs = q_gen.get("num_pairs_per_task", 0) * num_tasks
                    exp_controls = q_gen.get("num_controls_per_task", 0) * num_tasks

                    act_pairs = smoke_counts.get("matched_pairs", 0)
                    act_controls = smoke_counts.get("positive_controls", 0)
                    if act_pairs != exp_pairs:
                        issues.append(f"Smoke matched pairs count mismatch: expected {exp_pairs} (from config), got {act_pairs}")
                    if act_controls != exp_controls:
                        issues.append(f"Smoke positive controls count mismatch: expected {exp_controls} (from config), got {act_controls}")
            except Exception as err:
                issues.append(f"Failed to validate smoke counts against configs/smoke.yaml: {err}")
        except Exception as err:
            issues.append(f"Failed to parse smoke_report.json: {err}")
            smoke_report_statuses[REQUIRED_SMOKE_ARTIFACT_REPORT] = "CORRUPT"
            smoke_counts = {}
    else:
        issues.append(f"Missing {artifacts_dir / REQUIRED_SMOKE_ARTIFACT_REPORT}")
        smoke_report_statuses[REQUIRED_SMOKE_ARTIFACT_REPORT] = "MISSING"
        smoke_counts = {}

    # 7. Check all required PILOT reports
    pilot_report_statuses: Dict[str, str] = {}
    for r_name in REQUIRED_PILOT_REPORTS:
        r_path = reports_dir / r_name
        st = _check_report_status(r_path, r_name, issues)
        pilot_report_statuses[r_name] = st

    # 8. Validate exact pilot counts from pilot_distribution.json
    pilot_counts: Dict[str, Any] = {}
    pilot_dist_p = reports_dir / "pilot_distribution.json"
    if pilot_dist_p.exists():
        try:
            with open(pilot_dist_p, "r", encoding="utf-8") as f:
                dist_data = json.load(f)
            pilot_counts = {
                "total_records": dist_data.get("total_samples"),
                "matched_pairs": dist_data.get("matched_pairs"),
                "positive_controls": dist_data.get("positive_controls"),
            }
            actual_total = dist_data.get("total_samples", 0)
            actual_pairs = dist_data.get("matched_pairs", 0)
            actual_controls = dist_data.get("positive_controls", 0)
            if actual_total != PILOT_TOTAL_RECORDS:
                issues.append(f"Pilot total records: expected {PILOT_TOTAL_RECORDS}, got {actual_total}")
            if actual_pairs != PILOT_MATCHED_PAIRS:
                issues.append(f"Pilot matched pairs: expected {PILOT_MATCHED_PAIRS}, got {actual_pairs}")
            if actual_controls != PILOT_POSITIVE_CONTROLS:
                issues.append(f"Pilot positive controls: expected {PILOT_POSITIVE_CONTROLS}, got {actual_controls}")
        except Exception as err:
            issues.append(f"Failed to parse pilot_distribution.json: {err}")

    # 9. Validate pilot demonstration counts from pilot_demonstration_validation.json
    pilot_demo_p = reports_dir / "pilot_demonstration_validation.json"
    if pilot_demo_p.exists():
        try:
            with open(pilot_demo_p, "r", encoding="utf-8") as f:
                demo_data = json.load(f)
            demo_count = demo_data.get("demonstration_count", 0)
            if demo_count != PILOT_DEMONSTRATIONS:
                issues.append(f"Pilot demonstrations: expected {PILOT_DEMONSTRATIONS}, got {demo_count}")
            pilot_counts["demonstrations"] = demo_count

            # Count per task family
            demos = demo_data.get("demonstrations", {})
            t1_count = sum(1 for d in demos.values() if d.get("task_family") == "open_box")
            t2_count = sum(1 for d in demos.values() if d.get("task_family") == "place_object")
            pilot_counts["task1_demos"] = t1_count
            pilot_counts["task2_demos"] = t2_count
            if t1_count != PILOT_TASK1_DEMOS:
                issues.append(f"Pilot Task-1 demos: expected {PILOT_TASK1_DEMOS}, got {t1_count}")
            if t2_count != PILOT_TASK2_DEMOS:
                issues.append(f"Pilot Task-2 demos: expected {PILOT_TASK2_DEMOS}, got {t2_count}")
        except Exception as err:
            issues.append(f"Failed to parse pilot_demonstration_validation.json: {err}")

    # 10. Validate all 8 control subtypes from pilot_control_distribution.json
    control_subtype_counts: Dict[str, Any] = {}
    pilot_ctrl_dist_p = reports_dir / "pilot_control_distribution.json"
    if pilot_ctrl_dist_p.exists():
        try:
            with open(pilot_ctrl_dist_p, "r", encoding="utf-8") as f:
                ctrl_data = json.load(f)
            subtype_dist = ctrl_data.get("subtype_distribution", {})
            control_subtype_counts = subtype_dist

            for task_id, required_subtypes in REQUIRED_CONTROL_SUBTYPES.items():
                task_subtypes = subtype_dist.get(task_id, {})
                for subtype in required_subtypes:
                    count = task_subtypes.get(subtype, 0)
                    if count == 0:
                        issues.append(f"Missing required control subtype '{subtype}' for {task_id}")
        except Exception as err:
            issues.append(f"Failed to parse pilot_control_distribution.json: {err}")
    elif pilot_ctrl_dist_p.exists() is False and (reports_dir / "pilot_control_distribution.json").exists() is False:
        issues.append("Missing pilot_control_distribution.json - cannot verify control subtypes")

    # 11. Validate pilot reproducibility tested_record_count
    repro_tested_count: Optional[int] = None
    pilot_repro_p = reports_dir / "pilot_reproducibility.json"
    if pilot_repro_p.exists():
        try:
            with open(pilot_repro_p, "r", encoding="utf-8") as f:
                repro_data = json.load(f)
            samples = repro_data.get("samples", [])
            repro_tested_count = len(samples)
            if repro_tested_count != PILOT_REPRODUCIBILITY_RECORD_COUNT:
                issues.append(
                    f"Pilot reproducibility tested {repro_tested_count} records, expected {PILOT_REPRODUCIBILITY_RECORD_COUNT}"
                )
        except Exception as err:
            issues.append(f"Failed to parse pilot_reproducibility.json: {err}")

    overall_passed = (len(issues) == 0)

    rel_report = {
        "status": "PASSED" if overall_passed else "FAILED",
        "tested_code_commit": tested_commit,
        "current_head_commit": current_head,
        "code_changes_since_tested_commit": code_changed_after,
        "working_tree_clean": working_tree_clean,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke_report_statuses": smoke_report_statuses,
        "pilot_report_statuses": pilot_report_statuses,
        "smoke_counts": smoke_counts,
        "pilot_counts": pilot_counts,
        "control_subtype_counts": control_subtype_counts,
        "reproducibility_tested_record_count": repro_tested_count,
        "issue_count": len(issues),
        "issues": issues,
    }

    out_path = reports_dir / "release_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rel_report, f, indent=2)

    return overall_passed


if __name__ == "__main__":
    t_commit = sys.argv[1] if len(sys.argv) > 1 else None
    passed = verify_release_state(t_commit)
    print(f"Release verification state: {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)
