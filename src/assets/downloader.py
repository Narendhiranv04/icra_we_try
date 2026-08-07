"""
Downloader utility for fetching or verifying YCB and GSO realistic asset files.
"""

from pathlib import Path
from typing import List, Dict, Optional
import os

from src.assets.asset_registry import AssetRegistry, AssetMetadata


class AssetDownloader:
    """Utility to verify and manage local availability of benchmark assets."""

    def __init__(self, registry: AssetRegistry):
        self.registry = registry

    def verify_and_ensure_assets(self, asset_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Check presence of target assets and report status.
        All core assets are bundled locally in assets/objects/meshes/.
        """
        targets = asset_names or list(self.registry.assets.keys())
        status = {}
        for name in targets:
            asset = self.registry.get_asset(name)
            if asset is None:
                status[name] = False
                continue
            status[name] = asset.exists_on_disk
        return status


def ensure_asset_available(asset_name: str) -> Path:
    """Ensure that the given asset exists locally and return its mesh path."""
    registry = AssetRegistry()
    asset = registry.get_asset(asset_name)
    if asset is None or not asset.exists_on_disk:
        raise FileNotFoundError(f"Asset '{asset_name}' not available on disk.")
    return asset.mesh_path
