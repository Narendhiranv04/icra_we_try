import hashlib
import json
from typing import Any, Dict

import torch
import torch.nn as nn
import logging
import torchvision.transforms as T
from PIL import Image


class VisionEncoder(nn.Module):
    def __init__(self, model_name="dinov2_vitb14", device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logging.info(f"Loading Vision Encoder: {model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        
        # Freeze vision encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        
        # Standard DINOv2 transforms
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def extraction_signature(self) -> Dict[str, Any]:
        """Return the canonicalized extraction contract for this encoder.

        This is the single source of truth for vision feature extraction parameters.
        The returned dictionary is JSON-serializable and describes the exact
        preprocessing pipeline and model configuration used for feature extraction.
        """
        sig = {
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "num_patches": (224 // self.patch_size) ** 2,
            "resize": 224,
            "interpolation": "bicubic",
            "center_crop": 224,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        }
        # Compute deterministic signature hash from canonicalized JSON
        canonical_json = json.dumps(sig, sort_keys=True, separators=(",", ":"))
        sig["signature_sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return sig

    @torch.no_grad()
    def forward(self, images):
        """
        Extract features for a batch of images.
        Args:
            images: list of PIL Images or torch.Tensor (B, 3, H, W) in [0, 1]
        Returns:
            global_features: (B, embed_dim)
            patch_features: (B, num_patches, embed_dim)
        """
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            tensors = [self.transform(img) for img in images]
            x = torch.stack(tensors).to(self.device)
        else:
            x = images.to(self.device)
            # If already a tensor, assume it's preprocessed or needs transform?
            # Normally we pass PIL images during precomputation.
        
        ret = self.model.forward_features(x)
        cls_token = ret['x_norm_clstoken']
        patch_tokens = ret['x_norm_patchtokens']
        
        return cls_token, patch_tokens
