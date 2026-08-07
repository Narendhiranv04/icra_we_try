import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalHeatmapDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_size=224, patch_grid_size=16):
        super().__init__()
        self.patch_grid_size = patch_grid_size
        self.output_size = output_size
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32x32
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64x64
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 128x128
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Need to get to 224x224 (from 128x128 we can interpolate)
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, Z_R):
        """
        Z_R: (B, N, latent_dim) where N = patch_grid_size^2
        """
        B, N, D = Z_R.shape
        # Reshape to (B, D, H, W)
        H = W = self.patch_grid_size
        x = Z_R.transpose(1, 2).view(B, D, H, W)
        
        out = self.decoder(x)
        # Interpolate to exactly output_size
        out = F.interpolate(out, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return out.squeeze(1) # (B, H, W)
