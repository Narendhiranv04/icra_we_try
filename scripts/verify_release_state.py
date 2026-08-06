#!/usr/bin/env python3
"""
Release verification script for Relational Precondition Benchmark.

Verifies:
1. Candidate tested code commit exists.
2. Reports reference tested code commit.
3. No source code changed after tested code commit.
4. All required reports exist and passed.
5. Working tree state is recorded.
Exports data/reports/release_verification.json.
"""

import sys
import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List


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


def run_cmd(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def verify_release_state(tested_commit: str = None) -> bool:
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    issues: List[str] = []

    # 1. Resolve tested_code_commit
    if not tested_commit:
        tested_commit = os.environ.get("TESTED_CODE_COMMIT")
    if not tested_commit:
        try:
            tested_commit = run_cmd(["git", "rev-parse", "HEAD"])
        except Exception:
            tested_commit = "unknown"

    current_head = run_cmd(["git", "rev-parse", "HEAD"])

    # 2. Check if tested commit exists in git
    try:
        run_cmd(["git", "rev-parse", "--verify", tested_commit])
    except Exception:
        issues.append(f"Tested commit '{tested_commit}' is invalid or does not exist in git log")

    # 3. Check diff between tested_commit and current_head for source code modifications
    code_changed_after = False
    try:
        diff_output = run_cmd(["git", "diff", "--name-only", tested_commit, "HEAD"])
        changed_files = [f.strip() for f in diff_output.splitlines() if f.strip()]
        for f in changed_files:
            if f.startswith(("src/", "scripts/", "tests/", "configs/")):
                code_changed_after = True
                issues.append(f"Source code file '{f}' was modified after tested_code_commit '{tested_commit}'")
    except Exception as err:
        issues.append(f"Failed to compute git diff from tested commit: {err}")

    # 4. Check working tree status
    tree_status = run_cmd(["git", "status", "--porcelain"])
    working_tree_clean = (len(tree_status) == 0)

    # 5. Check all required reports exist and passed
    report_statuses: Dict[str, str] = {}
    for r_name in REQUIRED_SMOKE_REPORTS:
        r_path = reports_dir / r_name
        if not r_path.exists():
            issues.append(f"Required report missing: {r_name}")
            report_statuses[r_name] = "MISSING"
        elif r_name.endswith(".json"):
            try:
                with open(r_path, "r", encoding="utf-8") as f:
                    r_data = json.load(f)
                    st = r_data.get("status", "MISSING_STATUS")
                    report_statuses[r_name] = st
                    if st != "PASSED":
                        issues.append(f"Required report '{r_name}' failed with status '{st}'")
            except Exception as err:
                issues.append(f"Failed to parse report '{r_name}': {err}")
                report_statuses[r_name] = "CORRUPT"
        elif r_name == "pytest_results.xml":
            report_statuses[r_name] = "PASSED"  # XML checked via test_summary.json

    # 6. Check smoke_report.json reference
    smoke_rep_p = Path("artifacts/smoke/smoke_report.json")
    if smoke_rep_p.exists():
        try:
            with open(smoke_rep_p, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                s_commit = s_data.get("tested_code_commit")
                if s_commit != tested_commit:
                    issues.append(f"smoke_report.json tested_code_commit '{s_commit}' != '{tested_commit}'")
        except Exception as err:
            issues.append(f"Failed to parse smoke_report.json: {err}")
    else:
        issues.append("Missing artifacts/smoke/smoke_report.json")

    overall_passed = (len(issues) == 0)

    rel_report = {
        "status": "PASSED" if overall_passed else "FAILED",
        "tested_code_commit": tested_commit,
        "current_head_commit": current_head,
        "code_changes_since_tested_commit": code_changed_after,
        "working_tree_clean": working_tree_clean,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_statuses": report_statuses,
        "issue_count": len(issues),
        "issues": issues,
    }

    with open(reports_dir / "release_verification.json", "w", encoding="utf-8") as f:
        json.dump(rel_report, f, indent=2)

    return overall_passed


if __name__ == "__main__":
    t_commit = sys.argv[1] if len(sys.argv) > 1 else None
    passed = verify_release_state(t_commit)
    print(f"Release verification state: {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)
