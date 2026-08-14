import hashlib
import json
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging


class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logging.info(f"Loading Text Encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Freeze the text encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.embed_dim = self.model.config.hidden_size

    def extraction_signature(self) -> Dict[str, Any]:
        """Return the canonicalized extraction contract for this encoder.

        This is the single source of truth for text feature extraction parameters.
        The returned dictionary is JSON-serializable and describes the exact
        tokenization, pooling, and model configuration used for feature extraction.
        """
        sig = {
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
            "tokenizer": self.model_name,
            "truncation": True,
            "pooling": "attention_mask_weighted_mean",
        }
        # Compute deterministic signature hash from canonicalized JSON
        canonical_json = json.dumps(sig, sort_keys=True, separators=(",", ":"))
        sig["signature_sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return sig

    @torch.no_grad()
    def forward(self, texts):
        """
        Extract features for a list of strings.
        Args:
            texts: list of strings
        Returns:
            torch.Tensor of shape (B, embed_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings
