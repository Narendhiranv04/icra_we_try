#!/usr/bin/env python3
"""Intervention Model Training Script — Packet 5.

Trains intervention-conditioned feasibility models (B0, B0b, B3, V2)
using frozen pre-computed DINOv2 + MiniLM features.

Key design:
- Uses pre-computed feature cache (no encoder instantiation during training).
- Scene-balanced grouped pairwise logistic ranking loss.
- CPU training for authoritative smoke diagnostics.
- Fixed seed, no LR scheduler, no AMP.

Usage:
    python scripts/train_intervention_model.py \
        --config configs/intervention/learning/v2_smoke.yaml \
        [--device cpu] [--seed 42]
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.learning.intervention_config import InterventionTrainingConfig
from src.learning.intervention_dataset import (
    InterventionLearningDataset,
    InterventionGroupBatchSampler,
    collate_intervention_group,
)
from src.learning.intervention_loss import InterventionLoss, InterventionLossOutput
from src.learning.intervention_metrics import InterventionMetrics, InterventionMetricsOutput
from src.learning.models.intervention_model_registry import get_intervention_model
from src.learning.models.intervention_outputs import InterventionModelOutput


logger = logging.getLogger(__name__)


# ─── Provenance Helpers ────────────────────────────────────────────────────────

def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _compute_model_fingerprint(model: nn.Module) -> str:
    """Deterministic model-parameter fingerprint.

    Hashes: sorted(name, shape, dtype, raw CPU bytes) for each parameter.
    This is independent of PyTorch checkpoint file metadata.
    """
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        data = param.detach().cpu().contiguous().to(torch.float32)
        h.update(name.encode("utf-8"))
        h.update(str(param.shape).encode("utf-8"))
        h.update(str(param.dtype).encode("utf-8"))
        h.update(data.numpy().tobytes())
    return h.hexdigest()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ─── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full_dataset(
    model: nn.Module,
    dataset: InterventionLearningDataset,
    criterion: InterventionLoss,
    metrics_computer: InterventionMetrics,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on the entire dataset with shuffle=False.

    This is a TRAINING-SET diagnostic, NOT validation/test performance.
    """
    model.eval()

    sampler = InterventionGroupBatchSampler(
        dataset=dataset,
        scenes_per_batch=1,
        shuffle=False,
        seed=0,
    )

    all_post_logits = []
    all_ranking_scores = []
    all_post_feasible = []
    all_pre_feasible = []
    all_causal_effect = []
    all_group_ptrs = []
    current_offset = 0

    for batch_indices in sampler:
        items = [dataset[i] for i in batch_indices]
        batch = collate_intervention_group(items)

        # Move model_inputs to device
        model_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["model_inputs"].items()
        }
        supervision = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["supervision_targets"].items()
        }
        group_ptrs = batch["scene_group_ptrs"]

        output = model(model_inputs)

        # Accumulate (on CPU for metrics)
        all_post_logits.append(output.post_logit.cpu())
        all_ranking_scores.append(output.ranking_score.cpu())
        all_post_feasible.append(supervision["post_feasible"].cpu())
        all_pre_feasible.append(supervision["pre_feasible"].cpu())
        all_causal_effect.append(supervision["causal_effect"].cpu())

        # Adjust group pointers to global batch offset
        adjusted_ptrs = group_ptrs + current_offset
        all_group_ptrs.append(adjusted_ptrs[:-1])  # Exclude final pointer (will be rebuilt)
        current_offset += batch["batch_size"]

    # Reconstruct global tensors
    post_logit = torch.cat(all_post_logits)
    ranking_score = torch.cat(all_ranking_scores)
    post_feasible = torch.cat(all_post_feasible)
    pre_feasible = torch.cat(all_pre_feasible)
    causal_effect = torch.cat(all_causal_effect)

    # Rebuild global group pointers
    global_ptrs_list = list(torch.cat(all_group_ptrs).tolist()) + [current_offset]
    global_ptrs = torch.tensor(global_ptrs_list, dtype=torch.long)

    # Compute loss
    from src.learning.models.intervention_outputs import InterventionModelOutput as IMO
    full_output = IMO(
        post_logit=post_logit,
        ranking_score=ranking_score,
        relational_embedding=None,
        relational_tokens=None,
    )
    full_supervision = {
        "post_feasible": post_feasible,
        "pre_feasible": pre_feasible,
        "causal_effect": causal_effect,
    }

    loss_out = criterion(full_output, full_supervision, global_ptrs)
    metrics_out = metrics_computer.compute(full_output, full_supervision, global_ptrs)

    return {
        "total_loss": loss_out.total_loss.item(),
        "post_bce": loss_out.post_feasibility_loss.item(),
        "ranking_loss": loss_out.ranking_loss.item(),
        "ranking_pair_count": loss_out.ranking_pair_count,
        "valid_ranking_group_count": loss_out.valid_ranking_group_count,
        "post_accuracy": metrics_out.post_accuracy,
        "pairwise_ranking_accuracy": metrics_out.pairwise_ranking_accuracy,
        "top1_post_feasible_rate": metrics_out.top1_post_feasible_rate,
        "stop_top1_feasible_rate": metrics_out.stop_top1_feasible_rate,
        "proceed_top1_safe_rate": metrics_out.proceed_top1_safe_rate,
    }


# ─── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: InterventionTrainingConfig,
    initial_fingerprint: str,
    manifest_sha: str,
    cache_index_sha: str,
    out_path: Path,
) -> None:
    """Save a checkpoint with full provenance metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "provenance": {
                "model_name": config.model_name,
                "run_label": config.run_label if hasattr(config, "run_label") else config.model_name,
                "config": config.to_dict(),
                "seed": config.seed,
                "device": config.device,
                "manifest_sha256": manifest_sha,
                "feature_cache_index_sha256": cache_index_sha,
                "feature_cache_schema_version": "1.1.0",
                "initial_model_fingerprint_sha256": initial_fingerprint,
            },
        },
        out_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: InterventionTrainingConfig,
    manifest_sha: str,
    cache_index_sha: str,
) -> Dict[str, Any]:
    """Load a checkpoint with compatibility verification."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    prov = ckpt.get("provenance", {})

    # Compatibility checks
    ckpt_model_name = prov.get("config", {}).get("model_name", prov.get("model_name"))
    if ckpt_model_name != config.model_name:
        raise ValueError(
            f"Checkpoint model_name '{ckpt_model_name}' != "
            f"current config model_name '{config.model_name}'"
        )

    ckpt_manifest_sha = prov.get("manifest_sha256")
    if ckpt_manifest_sha and ckpt_manifest_sha != manifest_sha:
        raise ValueError(
            f"Checkpoint manifest SHA '{ckpt_manifest_sha}' != "
            f"current manifest SHA '{manifest_sha}'"
        )

    ckpt_cache_sha = prov.get("feature_cache_index_sha256")
    if ckpt_cache_sha and ckpt_cache_sha != cache_index_sha:
        raise ValueError(
            f"Checkpoint feature cache SHA '{ckpt_cache_sha}' != "
            f"current cache SHA '{cache_index_sha}'"
        )

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "provenance": prov,
    }


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(config: InterventionTrainingConfig) -> Dict[str, Any]:
    """Run full training loop with smoke diagnostics.

    Returns a result dict with all measured values for report generation.
    """
    _set_seeds(config.seed)

    # Determinism on CPU
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass  # Non-critical if env doesn't support

    device = torch.device(config.device)
    logger.info(f"Training device: {device}")
    logger.info(f"Run: {config.model_name} | seed={config.seed} | "
                f"lambda_feas={config.lambda_feas} | lambda_rank={config.lambda_rank}")

    # Load Dataset
    manifest_path = Path(config.manifest_path)
    features_dir = Path(config.features_dir)

    manifest_sha = _compute_file_sha256(manifest_path)
    cache_index_path = features_dir / "feature_cache_index.json"
    cache_index_sha = _compute_file_sha256(cache_index_path)

    dataset = InterventionLearningDataset(
        manifest_path=manifest_path,
        features_dir=features_dir,
        split="all",
    )
    logger.info(f"Dataset: {len(dataset)} records loaded.")

    # Build Model
    model = get_intervention_model(
        model_name=config.model_name,
        latent_dim=config.latent_dim,
        operator_embed_dim=config.operator_embed_dim,
        hidden_dim=config.hidden_dim,
        nhead=config.nhead,
        num_context_layers=config.num_context_layers,
        num_cross_layers=config.num_cross_layers,
        dropout=config.dropout,
    )
    model = model.to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {param_count:,}")

    # Initial fingerprint (before any gradient step)
    initial_fingerprint = _compute_model_fingerprint(model)
    logger.info(f"Initial model fingerprint: {initial_fingerprint}")

    # Loss and Metrics
    criterion = InterventionLoss(
        lambda_feas=config.lambda_feas,
        lambda_rank=config.lambda_rank,
    )
    metrics_computer = InterventionMetrics()

    # Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # ── Initial Evaluation (training-set diagnostic) ──
    logger.info("Running initial evaluation (full dataset, training-set diagnostic)...")
    initial_eval = evaluate_full_dataset(
        model, dataset, criterion, metrics_computer, device
    )
    logger.info(
        f"[INIT] total_loss={initial_eval['total_loss']:.4f} "
        f"post_bce={initial_eval['post_bce']:.4f} "
        f"ranking_loss={initial_eval['ranking_loss']:.4f} "
        f"pair_count={initial_eval['ranking_pair_count']} "
        f"valid_groups={initial_eval['valid_ranking_group_count']}"
    )

    # ── Training Epochs ──
    global_step = 0
    epoch_records = []

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        sampler = InterventionGroupBatchSampler(
            dataset=dataset,
            scenes_per_batch=config.scenes_per_batch,
            shuffle=True,
            seed=config.seed,
        )
        sampler.set_epoch(epoch)

        epoch_total_loss = 0.0
        epoch_batches = 0

        for batch_indices in sampler:
            items = [dataset[i] for i in batch_indices]
            batch = collate_intervention_group(items)

            model_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["model_inputs"].items()
            }
            supervision = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["supervision_targets"].items()
            }
            group_ptrs = batch["scene_group_ptrs"].to(device)

            optimizer.zero_grad()
            output = model(model_inputs)
            loss_out = criterion(output, supervision, group_ptrs)

            loss_out.total_loss.backward()

            if config.grad_clip_max_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_max_norm
                )

            optimizer.step()

            epoch_total_loss += loss_out.total_loss.item()
            epoch_batches += 1
            global_step += 1

            if global_step % config.log_every_n_batches == 0:
                logger.debug(
                    f"[E{epoch} S{global_step}] "
                    f"total={loss_out.total_loss.item():.4f} "
                    f"bce={loss_out.post_feasibility_loss.item():.4f} "
                    f"rank={loss_out.ranking_loss.item():.4f}"
                )

        avg_loss = epoch_total_loss / max(epoch_batches, 1)
        logger.info(f"[Epoch {epoch}/{config.num_epochs}] avg_train_loss={avg_loss:.4f}")

        epoch_records.append({"epoch": epoch, "avg_train_batch_loss": avg_loss})

        # Checkpoint (if configured)
        if (
            config.checkpoint_dir is not None
            and epoch % config.checkpoint_every_n_epochs == 0
        ):
            ckpt_path = (
                Path(config.checkpoint_dir)
                / f"{config.model_name}_seed{config.seed}_epoch{epoch:04d}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch, global_step, config,
                initial_fingerprint, manifest_sha, cache_index_sha, ckpt_path
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # ── Final Evaluation (training-set diagnostic) ──
    logger.info("Running final evaluation (full dataset, training-set diagnostic)...")
    final_eval = evaluate_full_dataset(
        model, dataset, criterion, metrics_computer, device
    )
    logger.info(
        f"[FINAL] total_loss={final_eval['total_loss']:.4f} "
        f"post_bce={final_eval['post_bce']:.4f} "
        f"ranking_loss={final_eval['ranking_loss']:.4f}"
    )

    # ── Finiteness Checks ──
    finite_losses = (
        torch.isfinite(torch.tensor(final_eval["total_loss"])).item()
        and torch.isfinite(torch.tensor(final_eval["post_bce"])).item()
        and torch.isfinite(torch.tensor(final_eval["ranking_loss"])).item()
    )

    # Check gradient finiteness from a fresh backward pass
    model.train()
    sampler_check = InterventionGroupBatchSampler(
        dataset=dataset, scenes_per_batch=1, shuffle=False
    )
    batch_indices_check = next(iter(sampler_check))
    items_check = [dataset[i] for i in batch_indices_check]
    batch_check = collate_intervention_group(items_check)
    model_inputs_check = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_check["model_inputs"].items()
    }
    supervision_check = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_check["supervision_targets"].items()
    }
    group_ptrs_check = batch_check["scene_group_ptrs"].to(device)

    optimizer.zero_grad()
    output_check = model(model_inputs_check)
    loss_check = criterion(output_check, supervision_check, group_ptrs_check)
    loss_check.total_loss.backward()

    finite_gradients = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.requires_grad
    )
    finite_parameters = all(
        torch.isfinite(p).all().item() for p in model.parameters()
    )

    # ── Checkpoint Roundtrip (if checkpoint dir configured) ──
    checkpoint_roundtrip_passed = None
    if config.checkpoint_dir is not None:
        ckpt_path = (
            Path(config.checkpoint_dir)
            / f"{config.model_name}_seed{config.seed}_final.pt"
        )
        save_checkpoint(
            model, optimizer, config.num_epochs, global_step, config,
            initial_fingerprint, manifest_sha, cache_index_sha, ckpt_path
        )

        # Load into fresh model and compare
        fresh_model = get_intervention_model(
            model_name=config.model_name,
            latent_dim=config.latent_dim,
            operator_embed_dim=config.operator_embed_dim,
            hidden_dim=config.hidden_dim,
            nhead=config.nhead,
            num_context_layers=config.num_context_layers,
            num_cross_layers=config.num_cross_layers,
            dropout=config.dropout,
        ).to(device)
        fresh_model.eval()

        load_checkpoint(
            ckpt_path, fresh_model, None, config, manifest_sha, cache_index_sha
        )

        # Compare outputs on the check batch
        model.eval()
        with torch.no_grad():
            out_orig = model(model_inputs_check)
            out_fresh = fresh_model(model_inputs_check)

        checkpoint_roundtrip_passed = torch.allclose(
            out_orig.post_logit, out_fresh.post_logit, atol=1e-6
        )
        logger.info(f"Checkpoint roundtrip: {'PASSED' if checkpoint_roundtrip_passed else 'FAILED'}")

    return {
        "model_name": config.model_name,
        "lambda_feas": config.lambda_feas,
        "lambda_rank": config.lambda_rank,
        "seed": config.seed,
        "device": config.device,
        "param_count": param_count,
        "initial_model_fingerprint_sha256": initial_fingerprint,
        "manifest_sha256": manifest_sha,
        "feature_cache_index_sha256": cache_index_sha,
        "epochs": config.num_epochs,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "finite_losses": finite_losses,
        "finite_gradients": finite_gradients,
        "finite_parameters": finite_parameters,
        "checkpoint_roundtrip_passed": checkpoint_roundtrip_passed,
        "epoch_records": epoch_records,
        "generalization_evaluation_performed": False,
        "smoke_dataset_used_for_training_and_diagnostics": True,
        "performance_claims_allowed": False,
    }


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train intervention feasibility model (Packet 5)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to write training result JSON",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (None = no checkpointing)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = InterventionTrainingConfig.from_yaml(args.config)

    # CLI overrides
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    result = train(config)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Training report written to: {args.report}")

    logger.info("=" * 70)
    logger.info(f"DONE | {result['model_name']} | epochs={result['epochs']}")
    logger.info(
        f"Initial: total={result['initial_eval']['total_loss']:.4f} "
        f"bce={result['initial_eval']['post_bce']:.4f} "
        f"rank={result['initial_eval']['ranking_loss']:.4f}"
    )
    logger.info(
        f"Final:   total={result['final_eval']['total_loss']:.4f} "
        f"bce={result['final_eval']['post_bce']:.4f} "
        f"rank={result['final_eval']['ranking_loss']:.4f}"
    )
    logger.info(
        f"Finite: losses={result['finite_losses']} "
        f"grads={result['finite_gradients']} "
        f"params={result['finite_parameters']}"
    )
    if result["checkpoint_roundtrip_passed"] is not None:
        logger.info(
            f"Checkpoint roundtrip: {'PASSED' if result['checkpoint_roundtrip_passed'] else 'FAILED'}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
