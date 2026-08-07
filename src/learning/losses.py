import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # logits: (B, H, W)
        # targets: (B, H, W)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1,2))
        union = probs.sum(dim=(1,2)) + targets.sum(dim=(1,2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class LearningLoss(nn.Module):
    def __init__(self, margin=0.2, lambda_cls=1.0, lambda_rank=0.5, lambda_heat=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.dice = DiceLoss()
        self.lambda_cls = lambda_cls
        self.lambda_rank = lambda_rank
        self.lambda_heat = lambda_heat
        
    def forward(self, logits, targets, s=None, pair_ids=None, heat_logits=None, heat_targets=None):
        l_cls = self.bce(logits.squeeze(-1), targets)
        
        loss = self.lambda_cls * l_cls
        l_rank = torch.tensor(0.0, device=logits.device)
        l_heat = torch.tensor(0.0, device=logits.device)
        
        # Ranking loss
        if s is not None and pair_ids is not None:
            # Pair logic
            s_pos = []
            s_neg = []
            pid_to_idx = {}
            for i, pid in enumerate(pair_ids):
                if pid not in pid_to_idx:
                    pid_to_idx[pid] = []
                pid_to_idx[pid].append(i)
                
            for pid, indices in pid_to_idx.items():
                if len(indices) == 2:
                    i1, i2 = indices
                    t1, t2 = targets[i1], targets[i2]
                    if t1 == 0 and t2 == 1:
                        s_pos.append(s[i1])
                        s_neg.append(s[i2])
                    elif t1 == 1 and t2 == 0:
                        s_pos.append(s[i2])
                        s_neg.append(s[i1])
            
            if s_pos:
                s_pos = torch.stack(s_pos)
                s_neg = torch.stack(s_neg)
                # s_pos should be > s_neg
                l_rank = self.margin_loss(s_pos, s_neg, torch.ones_like(s_pos))
                loss += self.lambda_rank * l_rank
                
        # Heatmap loss
        if heat_logits is not None and heat_targets is not None:
            l_mask_bce = self.bce(heat_logits, heat_targets)
            l_dice = self.dice(heat_logits, heat_targets)
            l_heat = l_mask_bce + l_dice
            loss += self.lambda_heat * l_heat
            
        return loss, {
            "l_cls": l_cls.item(),
            "l_rank": l_rank.item(),
            "l_heat": l_heat.item(),
            "loss": loss.item()
        }
