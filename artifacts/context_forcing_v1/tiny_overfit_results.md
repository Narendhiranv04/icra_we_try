# Tiny Overfit Diagnostic Results

## Overview
This diagnostic tested if models could overfit a tiny 8-pair dataset. A model capable of using context (demo/text) to differentiate identical RGB inputs should achieve 100% accuracy on the train set. A model that ignores context (e.g. `query_only`) will fail to overfit because identical images map to different labels.

## Accuracy (Final Epoch)
| Model | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| query_only | 1.00 | 0.50 | Pass |
| pooled_multimodal | 1.00 | 0.50 | Pass |
| relational_heatmap | 1.00 | 0.25 | Pass |

## Gradient Norms (Epoch 1)
Checking if context-specific parameters receive gradients during the first backward pass:

### query_only
No gradient data found.

### pooled_multimodal
```json
{
  "context_mlp.0.weight": 0.1710619479417801,
  "context_mlp.0.bias": 0.012175679206848145,
  "context_mlp.2.weight": 0.1333666294813156,
  "context_mlp.2.bias": 0.04171845689415932,
  "classifier.0.weight": 1.8442399501800537,
  "classifier.0.bias": 0.08210064470767975,
  "classifier.3.weight": 0.8962962031364441,
  "classifier.3.bias": 0.26025390625
}
```

### relational_heatmap
```json
{
  "cross_attention.0.multihead_attn.in_proj_weight": 0.33956584334373474,
  "cross_attention.0.multihead_attn.in_proj_bias": 0.026005610823631287,
  "cross_attention.0.multihead_attn.out_proj.weight": 0.43368402123451233,
  "cross_attention.0.multihead_attn.out_proj.bias": 0.044665683060884476,
  "cross_attention.0.linear1.weight": 0.13583405315876007,
  "cross_attention.0.linear1.bias": 0.013968355022370815,
  "cross_attention.0.linear2.weight": 0.3218403458595276,
  "cross_attention.0.linear2.bias": 0.045725543051958084,
  "cross_attention.0.norm1.weight": 0.028640061616897583,
  "cross_attention.0.norm1.bias": 0.04640614241361618,
  "cross_attention.0.norm2.weight": 0.02874481864273548,
  "cross_attention.0.norm2.bias": 0.046671684831380844,
  "cross_attention.1.multihead_attn.in_proj_weight": 0.32553648948669434,
  "cross_attention.1.multihead_attn.in_proj_bias": 0.024910327047109604,
  "cross_attention.1.multihead_attn.out_proj.weight": 0.4155859053134918,
  "cross_attention.1.multihead_attn.out_proj.bias": 0.046533942222595215,
  "cross_attention.1.linear1.weight": 0.1491624414920807,
  "cross_attention.1.linear1.bias": 0.014860262162983418,
  "cross_attention.1.linear2.weight": 0.3315335214138031,
  "cross_attention.1.linear2.bias": 0.04710206016898155,
  "cross_attention.1.norm1.weight": 0.03133849427103996,
  "cross_attention.1.norm1.bias": 0.04824436828494072,
  "cross_attention.1.norm2.weight": 0.030010992661118507,
  "cross_attention.1.norm2.bias": 0.04836522787809372,
  "classifier.0.weight": 0.8151282668113708,
  "classifier.0.bias": 0.08365136384963989,
  "classifier.3.weight": 0.7364214658737183,
  "classifier.3.bias": 0.25048828125
}
```

