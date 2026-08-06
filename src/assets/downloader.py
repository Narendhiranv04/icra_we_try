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
