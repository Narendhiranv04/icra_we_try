"""
Asset registry for querying and resolving YCB and GSO realistic 3D assets.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class AssetMetadata:
    name: str
    source: str
    source_id: str
    display_name: str
    category: str
    bounding_box_size: List[float]
    mass_kg: float
    mesh_path: str
    texture_path: str
    collision_type: str
    collision_params: dict
    default_friction: List[float]
    splits: List[str]
    task_roles: List[str]

    @property
    def exists_on_disk(self) -> bool:
        """Check if mesh and texture files exist locally."""
        mesh_ok = Path(self.mesh_path).is_file()
        tex_ok = Path(self.texture_path).is_file() if self.texture_path else True
        return mesh_ok and tex_ok


class AssetRegistry:
    """Registry for managing and retrieving YCB / GSO object metadata and assets."""

    def __init__(self, config_path: Union[str, Path] = "configs/assets_ycb_gso.yaml"):
        self.config_path = Path(config_path)
        self.assets: Dict[str, AssetMetadata] = {}
        self._load_config()

    def _load_config(self) -> None:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Asset config file not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_assets = data.get("assets", {})
        for name, entry in raw_assets.items():
            self.assets[name] = AssetMetadata(
                name=entry["name"],
                source=entry.get("source", "unknown"),
                source_id=entry.get("source_id", ""),
                display_name=entry.get("display_name", name),
                category=entry.get("category", "general"),
                bounding_box_size=entry.get("bounding_box_size", [0.1, 0.1, 0.1]),
                mass_kg=float(entry.get("mass_kg", 0.3)),
                mesh_path=entry.get("mesh_path", ""),
                texture_path=entry.get("texture_path", ""),
                collision_type=entry.get("collision_type", "box"),
                collision_params=entry.get("collision_params", {}),
                default_friction=entry.get("default_friction", [1.0, 0.005, 0.0001]),
                splits=entry.get("splits", ["id"]),
                task_roles=entry.get("task_roles", ["blocker"]),
            )

    def get_asset(self, asset_name: str) -> Optional[AssetMetadata]:
        """Retrieve metadata for a specific asset name."""
        return self.assets.get(asset_name)

    def list_assets(
        self,
        split: Optional[str] = None,
        role: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[AssetMetadata]:
        """List assets filtered by split, role, or source dataset."""
        results = []
        for asset in self.assets.values():
            if split and split not in asset.splits:
                continue
            if role and role not in asset.task_roles:
                continue
            if source and asset.source.lower() != source.lower():
                continue
            results.append(asset)
        return results

    def verify_all_assets_present(self) -> Dict[str, bool]:
        """Verify presence of all registered assets on disk."""
        return {name: asset.exists_on_disk for name, asset in self.assets.items()}
