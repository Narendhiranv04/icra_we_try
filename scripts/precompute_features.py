import os
import json
import torch
import cv2
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import hashlib

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.vision_encoder import VisionEncoder
from src.learning.text_encoder import TextEncoder

logging.basicConfig(level=logging.INFO)

PARAPHRASES = {
    "task_1": [
        "Open the box.",
        "Open the box lid.",
        "Lift the lid of the box.",
        "Uncover the box by opening its lid."
    ],
    "task_2": [
        "Place object1 in the target region.",
        "Put object1 inside the marked target area.",
        "Move object1 into the destination region.",
        "Place the object into the indicated area."
    ],
    "generic": [
        "Perform the demonstrated task."
    ]
}

CACHE_SCHEMA_VERSION = "1.1"

def extract_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames == 0:
        logging.warning(f"No frames in {video_path}")
        return frames

    # Deterministic temporal sampling
    indices = [int(i * (total_frames - 1) / max(1, num_frames - 1)) for i in range(num_frames)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()

    # Fill if missing frames
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="learning_data/index.jsonl")
    parser.add_argument("--out_dir", default="learning_data/features")
    parser.add_argument("--num_demo_frames", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Force recomputation of features")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Text
    logging.info("Precomputing text features...")
    text_enc = TextEncoder(device=device)
    text_features = {}

    for task_id, phrases in PARAPHRASES.items():
        # Get embeddings with autocast
        with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            embs = text_enc(phrases).to(torch.float32).cpu()
        for p, emb in zip(phrases, embs):
            text_features[p] = emb

    torch.save(text_features, out_dir / "text_features.pt")

    # Free memory
    del text_enc
    if device == "cuda":
        torch.cuda.empty_cache()

    # Vision
    logging.info("Precomputing vision features...")
    vision_enc = VisionEncoder(device=device)

    with open(args.index, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    demos_processed = set()

    # Demo features
    logging.info("Processing demonstrations...")
    for rec in tqdm(records):
        demo_path = rec.get("demonstration_video_path")
        if not demo_path or not os.path.exists(demo_path):
            raise RuntimeError(f"Missing required demo video {demo_path} for sample_id={rec.get('sample_id')} pair_id={rec.get('pair_id')}")
        if demo_path in demos_processed:
            continue

        demo_id = rec["demonstration_id"]
        out_path = out_dir / f"demo_{demo_id}_features.pt"
        meta_path = out_dir / f"demo_{demo_id}_features.meta.json"

        file_hash = compute_sha256(demo_path)

        if not args.force and out_path.exists() and meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("source_sha256") == file_hash and meta.get("num_frames") == args.num_demo_frames:
                demos_processed.add(demo_path)
                continue

        demo_frames = extract_frames(demo_path, num_frames=args.num_demo_frames)
        if not demo_frames:
            raise RuntimeError(f"Failed to extract frames from {demo_path} for sample_id={rec.get('sample_id')}")

        with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            global_feat, patch_feat = vision_enc(demo_frames)
            # Shapes: global_feat (K, D), patch_feat (K, N, D)

        demo_id = rec["demonstration_id"]
        feat_dict = {
            "global": global_feat.to(torch.float32).cpu(),
            "patch": patch_feat.to(torch.float32).cpu()
        }
        torch.save(feat_dict, out_path)

        meta = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "demo_id": demo_id,
            "source_path": demo_path,
            "source_sha256": file_hash,
            "encoder_id": "dinov2_vitb14",
            "num_frames": args.num_demo_frames,
            "sampled_frame_indices": [int(i * (args.num_demo_frames - 1) / max(1, args.num_demo_frames - 1)) for i in range(args.num_demo_frames)]
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        demos_processed.add(demo_path)

    # Query features
    logging.info("Processing queries...")
    for rec in tqdm(records):
        sample_id = rec["sample_id"]
        rgb_path = rec.get("query_rgb_path")
        if not rgb_path or not os.path.exists(rgb_path):
            raise RuntimeError(f"Missing required query image {rgb_path} for sample_id={sample_id}")

        out_path = out_dir / f"query_{sample_id}_features.pt"
        meta_path = out_dir / f"query_{sample_id}_features.meta.json"

        file_hash = compute_sha256(rgb_path)

        if not args.force and out_path.exists() and meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("source_sha256") == file_hash:
                continue

        img = Image.open(rgb_path).convert("RGB")
        with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            global_feat, patch_feat = vision_enc([img])

        feat_dict = {
            "global": global_feat.squeeze(0).to(torch.float32).cpu(),
            "patch": patch_feat.squeeze(0).to(torch.float32).cpu()
        }
        torch.save(feat_dict, out_path)

        meta = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "sample_id": sample_id,
            "source_path": rgb_path,
            "source_sha256": file_hash,
            "encoder_id": "dinov2_vitb14"
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    logging.info("Precomputation finished.")

if __name__ == "__main__":
    main()
