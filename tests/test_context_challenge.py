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
        t1, t2 = pairs
        
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
