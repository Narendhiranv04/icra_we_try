import json
from pathlib import Path
from src.generation.context_challenge_generator import ContextChallengeGenerator

def main():
    generator = ContextChallengeGenerator(output_dir="data/context_challenge")
    manifest_path = Path("data/manifests/context_challenge_manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    # 4 states, 10 physical scenes each = 40 scenes
    idx = 0
    for state in ["A", "B", "C", "D"]:
        for i in range(10):
            seed = 2000 + idx # Ensure deterministic distinct seeds
            scene_id = f"context_scene_{idx:03d}_{state}"
            records = generator.generate_scene(scene_id, state, seed=seed)
            all_records.extend(records)
            idx += 1
            
    with open(manifest_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(all_records)} records in {manifest_path}")

if __name__ == "__main__":
    main()
