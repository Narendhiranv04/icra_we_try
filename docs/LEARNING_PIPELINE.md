# Learning Pipeline

This document details the Representation Learning pipeline for the egocentric relational-precondition benchmark.

## Inference Inputs
At inference time, the model only has access to:
1. A natural language instruction.
2. A successful demonstration video (or a subset of frames).
3. The current query RGB image.

**Privileged Information (NOT USED AT INFERENCE):**
- Object IDs
- Segmentation masks
- Poses
- Simulator State

## Architecture

### Feature Encoders (Frozen)
- **Vision:** DINOv2 (`ViT-B/14`). Extracts global features (768-dim) and patch features (16x16).
- **Text:** SentenceTransformer (`all-MiniLM-L6-v2`). Extracts a single 384-dim language embedding.

### Multimodal Baselines
- **Query Only:** MLP directly from Query Global feature to prediction.
- **Pooled Multimodal:** Context vector created by mean-pooling demo global features concatenated with the text feature.

### Relational Model (Cross-Attention)
1. **Temporal Encoding:** Text token and Demonstration Global tokens are passed through a self-attention Transformer.
2. **Cross-Attention:** Query patch tokens attend to the temporal context tokens to extract relational compatibility representations.
3. **Latent Similarity:** The alignment compatibility $s$ is explicitly formulated as the cosine similarity between the L2-normalized predicted latent and the L2-normalized target query latent.
4. **Heatmap Decoding:** The spatial tokens are reshaped back to 16x16 and decoded via CNN into a causal violation heatmap.

## Preprocessing and Assignment
- **Demo Assignment:** Demos are deterministically assigned based on the query pair ID via `hashlib.sha256(pair_id.encode())` to ensure absolute stability across runs.
- **Cache Provenance Safety:** Cached `*.pt` files are verified against a `.meta.json` file storing the source data `SHA256` hash. If the underlying data changes, the cache safely re-evaluates or errors out.
- **Seed Isolation:** Train/val splits are controlled strictly by `split_seed`, while model architecture initialization and mini-batch shuffling are controlled by the model `seed`. This ensures identical train/val sets across multi-seed evaluations.
- **Fail-Hard Manifests:** The index generation process operates with strict fail-hard semantics on malformed metadata, demanding cleanly validated benchmark manifests.

## Losses
- **Classification:** `BCEWithLogitsLoss` on the STOP (1) / PROCEED (0) targets.
- **Ranking Loss:** `MarginRankingLoss`. Enforces `compatibility(PROCEED) > compatibility(STOP)` for matched causal pairs.
- **Heatmap Loss:** Sum of pixel-wise BCE and Dice loss.

## Diagnostics and Benchmarks

### Conditioning Diagnostics
Dynamic ablation modes evaluate how the model relies on the conditioning vectors.
- `wrong_instruction`: Swaps the instruction for the opposite task.
- `heldout_paraphrase`: Evaluates on a semantically equivalent but lexically disjoint phrase.
- `zero_text`: Zeroes out the text embedding.
- `wrong_demo`: Swaps the demonstration video to the opposite task.
- `zero_demo`: Zeroes out the visual demonstration features.

### Benchmark B: Context Challenge
A strictly controlled subset of scenes where a single query RGB image inherently represents **both tasks** but with inverted semantics (Task 1: STOP, Task 2: PROCEED). Models that bypass multi-modal conditioning (e.g. Query-Only) score exactly 0% on reversal metrics.

## Running Experiments

### Feature Precomputation
To avoid redundantly running DINOv2 and text models, run:
```bash
python scripts/build_learning_index.py
python scripts/precompute_features.py
```
This writes `*.pt` cached tensors to `learning_data/features/`.

### Training
```bash
python scripts/train_model.py --config configs/learning/query_only.yaml
python scripts/train_model.py --config configs/learning/pooled_multimodal.yaml
python scripts/train_model.py --config configs/learning/relational_heatmap.yaml
```

### Evaluation
```bash
python scripts/evaluate_model.py --dir learning_outputs/relational_heatmap_seed42
python scripts/evaluate_context_challenge.py --experiment-dir learning_outputs/relational_heatmap_seed42
```

### Multi-seed Protocol
For rigorous results, always use the 5-seed automated script:
```bash
./run_all_experiments.sh
```
This runs 5 model seeds (`11`, `23`, `42`, `67`, `101`), evaluates on Benchmark A (with ablations) and Benchmark B (Context Challenge), and generates aggregated bootstapped metrics.

## GPU Requirements
The architecture is designed to train on a 16 GB VRAM GPU. The use of frozen encoders heavily mitigates memory limits.
