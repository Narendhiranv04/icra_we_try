import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, query, key_value):
        q2, _ = self.multihead_attn(query, key_value, key_value)
        query = query + self.dropout1(q2)
        query = self.norm1(query)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(q2)
        query = self.norm2(query)
        return query

class DemoLanguageConditionedRelationalModel(nn.Module):
    def __init__(self, text_dim=384, vision_dim=768, latent_dim=256, num_demo_frames=4):
        super().__init__()
        self.proj_t = nn.Linear(text_dim, latent_dim)
        self.proj_v_global = nn.Linear(vision_dim, latent_dim)
        self.proj_v_patch = nn.Linear(vision_dim, latent_dim)
        
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_demo_frames + 1, latent_dim))
        
        self.temporal_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model=latent_dim) for _ in range(2)
        ])
        
        self.cross_attention = nn.ModuleList([
            CrossAttentionLayer(d_model=latent_dim) for _ in range(2)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1)
        )
        
    def forward(self, text_feat, demo_global, query_patch):
        """
        text_feat: (B, text_dim)
        demo_global: (B, K, vision_dim)
        query_patch: (B, N, vision_dim)
        """
        B, K, _ = demo_global.shape
        
        z_t = self.proj_t(text_feat).unsqueeze(1) # (B, 1, latent_dim)
        z_d = self.proj_v_global(demo_global) # (B, K, latent_dim)
        
        # Concat [text, demo_1, ..., demo_K]
        seq = torch.cat([z_t, z_d], dim=1) # (B, 1+K, latent_dim)
        seq = seq + self.temporal_pos_embed[:, :1+K, :]
        
        Z_S = seq
        for layer in self.temporal_encoder:
            Z_S = layer(Z_S)
            
        # Context pool (can be mean or just the text token)
        # Using mean pool over all temporal tokens
        z_S = Z_S.mean(dim=1)
        
        # Query
        Z_Q = self.proj_v_patch(query_patch) # (B, N, latent_dim)
        
        # Cross Attention
        Z_R = Z_Q
        for layer in self.cross_attention:
            Z_R = layer(Z_R, Z_S)
            
        z_R = Z_R.mean(dim=1)
        
        logits = self.classifier(z_R)
        
        # Latent ranking objective
        z_S_norm = F.normalize(z_S, dim=-1)
        z_Q_norm = F.normalize(z_R, dim=-1)
        s = (z_S_norm * z_Q_norm).sum(dim=-1)
        
        return logits, s, Z_R
