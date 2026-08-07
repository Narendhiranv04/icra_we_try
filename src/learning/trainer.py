import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import compute_metrics
import logging
import os

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    all_preds, all_targets, all_scores, all_pair_ids = [], [], [], []
    all_heat_preds, all_heat_targets = [], []
    
    for i, batch in enumerate(dataloader):
        text_feat = batch["text_feat"].to(device)
        query_global = batch["query_global"].to(device)
        query_patch = batch["query_patch"].to(device)
        demo_global = batch["demo_global"].to(device)
        demo_patch = batch["demo_patch"].to(device)
        targets = batch["label"].to(device)
        masks = batch["mask"].to(device)
        pair_ids = batch["pair_id"]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if hasattr(model, "forward"):
                # Based on model type
                import inspect
                sig = inspect.signature(model.forward)
                kwargs = {}
                if "text_feat" in sig.parameters: kwargs["text_feat"] = text_feat
                if "demo_global" in sig.parameters: kwargs["demo_global"] = demo_global
                if "query_global" in sig.parameters: kwargs["query_global"] = query_global
                if "query_patch" in sig.parameters: kwargs["query_patch"] = query_patch
                if "x" in sig.parameters: kwargs["x"] = query_global # for query-only
                
                out = model(**kwargs)
            else:
                out = model(query_global)
                
            logits, s, Z_R = None, None, None
            if isinstance(out, tuple):
                if len(out) == 3:
                    logits, s, Z_R = out
                else:
                    logits, s = out
            else:
                logits = out
                
            # If heatmap model is attached
            heat_logits = None
            if hasattr(model, "heatmap_decoder") and Z_R is not None:
                heat_logits = model.heatmap_decoder(Z_R)
                
            loss, loss_dict = criterion(logits, targets, s=s, pair_ids=pair_ids, heat_logits=heat_logits, heat_targets=masks if heat_logits is not None else None)
            loss = loss / accumulation_steps
            
        if scaler:
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        total_loss += loss_dict["loss"] * accumulation_steps
        
        if logits is not None:
            all_preds.extend(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            if s is not None:
                all_scores.extend(s.detach().cpu().numpy())
            all_pair_ids.extend(pair_ids)
            if heat_logits is not None:
                all_heat_preds.extend(torch.sigmoid(heat_logits).detach().cpu().numpy())
                all_heat_targets.extend(masks.cpu().numpy())
            
    metrics = compute_metrics(
        all_preds, all_targets, 
        all_scores if all_scores else None, 
        all_pair_ids if all_pair_ids else None,
        all_heat_preds if all_heat_preds else None,
        all_heat_targets if all_heat_targets else None
    )
    metrics["loss"] = total_loss / len(dataloader)
    return metrics

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_scores, all_pair_ids = [], [], [], []
    all_heat_preds, all_heat_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            text_feat = batch["text_feat"].to(device)
            query_global = batch["query_global"].to(device)
            query_patch = batch["query_patch"].to(device)
            demo_global = batch["demo_global"].to(device)
            targets = batch["label"].to(device)
            masks = batch["mask"].to(device)
            pair_ids = batch["pair_id"]
            
            with torch.cuda.amp.autocast():
                import inspect
                sig = inspect.signature(model.forward)
                kwargs = {}
                if "text_feat" in sig.parameters: kwargs["text_feat"] = text_feat
                if "demo_global" in sig.parameters: kwargs["demo_global"] = demo_global
                if "query_global" in sig.parameters: kwargs["query_global"] = query_global
                if "query_patch" in sig.parameters: kwargs["query_patch"] = query_patch
                if "x" in sig.parameters: kwargs["x"] = query_global
                
                out = model(**kwargs)
                
                logits, s, Z_R = None, None, None
                if isinstance(out, tuple):
                    if len(out) == 3:
                        logits, s, Z_R = out
                    else:
                        logits, s = out
                else:
                    logits = out
                    
                heat_logits = None
                if hasattr(model, "heatmap_decoder") and Z_R is not None:
                    heat_logits = model.heatmap_decoder(Z_R)
                    
                loss, loss_dict = criterion(logits, targets, s=s, pair_ids=pair_ids, heat_logits=heat_logits, heat_targets=masks if heat_logits is not None else None)
                
            total_loss += loss_dict["loss"]
            
            if logits is not None:
                all_preds.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                if s is not None:
                    all_scores.extend(s.cpu().numpy())
                all_pair_ids.extend(pair_ids)
                if heat_logits is not None:
                    all_heat_preds.extend(torch.sigmoid(heat_logits).cpu().numpy())
                    all_heat_targets.extend(masks.cpu().numpy())
                
    metrics = compute_metrics(
        all_preds, all_targets, 
        all_scores if all_scores else None, 
        all_pair_ids if all_pair_ids else None,
        all_heat_preds if all_heat_preds else None,
        all_heat_targets if all_heat_targets else None
    )
    metrics["loss"] = total_loss / max(1, len(dataloader))
    metrics["_raw_preds"] = all_preds
    metrics["_raw_targets"] = all_targets
    metrics["_raw_scores"] = all_scores
    metrics["_raw_pair_ids"] = all_pair_ids
    return metrics
