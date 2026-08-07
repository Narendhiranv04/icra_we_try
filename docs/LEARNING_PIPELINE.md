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
1. **Temporal Encoding:** Text token and Demonstration Global tokens are passed through a 2-layer self-attention Transformer.
2. **Cross-Attention:** Query patch tokens attend to the temporal context tokens to extract relational compatibility representations.
3. **Heatmap Decoding:** The 256 spatial tokens are reshaped back to 16x16 and decoded via CNN into a causal violation heatmap.

## Losses
- **Classification:** `BCEWithLogitsLoss` on the STOP (1) / PROCEED (0) targets.
- **Ranking Loss:** `MarginRankingLoss`. We enforce that `compatibility(PROCEED) > compatibility(STOP)` for matched causal pairs.
- **Heatmap Loss:** Sum of pixel-wise BCE and Dice loss.

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
python scripts/evaluate_model.py --dir learning_outputs/relational_heatmap
```

## GPU Requirements
The architecture is designed to train on a 16 GB VRAM GPU. The use of frozen encoders heavily mitigates memory limits.
