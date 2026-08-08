import pytest
import json
from pathlib import Path
import hashlib

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def test_context_challenge_rgb_invariant():
    manifest_path = Path("data/manifests/context_challenge_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("Context challenge dataset not generated yet.")

    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    # Pair records by scene
    scenes = {}
    for r in records:
        sid = r["pair_id"]
        if sid not in scenes:
            scenes[sid] = []
        scenes[sid].append(r)

    for sid, pairs in scenes.items():
        assert len(pairs) == 2, f"Scene {sid} should have exactly 2 records"
        t1 = pairs[0] if pairs[0]["task_id"] == "task_1" else pairs[1]
        t2 = pairs[1] if pairs[1]["task_id"] == "task_2" else pairs[0]

        assert t1["query_rgb_path"] == t2["query_rgb_path"], f"RGB paths must exactly match for {sid}"
        hash1 = compute_sha256(t1["query_rgb_path"])
        hash2 = compute_sha256(t2["query_rgb_path"])
        assert hash1 == hash2, "RGB hash must be exactly identical"

        assert t1["state"] == t2["state"], "States must match"
        state = t1["state"]

        if state == "A":
            assert t1["label"] == "STOP"
            assert t2["label"] == "PROCEED"
        elif state == "B":
            assert t1["label"] == "PROCEED"
            assert t2["label"] == "STOP"
        elif state == "C":
            assert t1["label"] == "STOP"
            assert t2["label"] == "STOP"
        elif state == "D":
            assert t1["label"] == "PROCEED"
            assert t2["label"] == "PROCEED"

def test_context_challenge_counts_and_demos():
    manifest_path = Path("data/manifests/context_challenge_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("Context challenge dataset not generated yet.")

    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 80, f"Expected exactly 80 context records, got {len(records)}"

    t1_demos = set(r["demonstration_id"] for r in records if r["task_id"] == "task_1")
    t2_demos = set(r["demonstration_id"] for r in records if r["task_id"] == "task_2")

    assert len(t1_demos) == 3, f"Expected exactly 3 Task 1 demos, got {len(t1_demos)}"
    assert len(t2_demos) == 3, f"Expected exactly 3 Task 2 demos, got {len(t2_demos)}"

    states_count = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in records:
        states_count[r["state"]] += 1

    # Each state should have exactly 10 scenes * 2 tasks = 20 records
    assert states_count["A"] == 20
    assert states_count["B"] == 20
    assert states_count["C"] == 20
    assert states_count["D"] == 20

def test_causal_mask_and_occupancy():
    manifest_path = Path("data/manifests/context_challenge_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("Context challenge dataset not generated yet.")

    with open(manifest_path, "r") as f:
        records = [json.loads(line) for line in f]

    for r in records:
        assert "measured_lid_occupancy" in r
        assert "measured_target_occupancy" in r
        assert "causal_mask_path" in r

        state = r["state"]
        if state == "A":
            assert r["measured_lid_occupancy"] is True
            assert r["measured_target_occupancy"] is False
        elif state == "B":
            assert r["measured_lid_occupancy"] is False
            assert r["measured_target_occupancy"] is True
        elif state == "C":
            assert r["measured_lid_occupancy"] is True
            assert r["measured_target_occupancy"] is True
        elif state == "D":
            assert r["measured_lid_occupancy"] is False
            assert r["measured_target_occupancy"] is False
