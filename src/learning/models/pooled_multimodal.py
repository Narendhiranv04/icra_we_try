import torch
import torch.nn as nn

class PooledMultimodalBaseline(nn.Module):
    def __init__(self, text_dim=384, vision_dim=768, latent_dim=256, use_text=True, use_demo=True):
        super().__init__()
        self.use_text = use_text
        self.use_demo = use_demo
        
        self.proj_t = nn.Linear(text_dim, latent_dim)
        self.proj_v = nn.Linear(vision_dim, latent_dim)
        
        self.context_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1)
        )
        
    def forward(self, text_feat, demo_global, query_global):
        """
        text_feat: (B, text_dim)
        demo_global: (B, K, vision_dim)
        query_global: (B, vision_dim)
        """
        B = text_feat.size(0)
        
        if self.use_text:
            z_t = self.proj_t(text_feat) # (B, 256)
        else:
            z_t = torch.zeros(B, self.proj_t.out_features, device=text_feat.device)
            
        if self.use_demo:
            z_d_seq = self.proj_v(demo_global) # (B, K, 256)
            z_d = z_d_seq.mean(dim=1) # (B, 256)
        else:
            z_d = torch.zeros(B, self.proj_v.out_features, device=demo_global.device)
            
        z_q = self.proj_v(query_global) # (B, 256)
        
        z_context = self.context_mlp(torch.cat([z_t, z_d], dim=-1))
        
        c = torch.cat([
            z_context,
            z_q,
            torch.abs(z_context - z_q),
            z_context * z_q
        ], dim=-1)
        
        logits = self.classifier(c)
        
        # Explicit latent ranking compatibility
        import torch.nn.functional as F
        z_c_norm = F.normalize(z_context, p=2, dim=-1)
        z_q_norm = F.normalize(z_q, p=2, dim=-1)
        s = torch.sum(z_c_norm * z_q_norm, dim=-1)
        
        return logits, s
