"""
Unit tests for YCB and GSO asset registry and downloader.
"""

from pathlib import Path
from src.assets.asset_registry import AssetRegistry
from src.assets.downloader import AssetDownloader


def test_asset_registry_loading():
    registry = AssetRegistry("configs/assets_ycb_gso.yaml")
    assert len(registry.assets) > 0

    coffee_can = registry.get_asset("ycb_coffee_can")
    assert coffee_can is not None
    assert coffee_can.source == "ycb"
    assert coffee_can.category == "pantry"
    assert len(coffee_can.bounding_box_size) == 3


def test_asset_disk_verification():
    registry = AssetRegistry("configs/assets_ycb_gso.yaml")
    downloader = AssetDownloader(registry)
    status = downloader.verify_and_ensure_assets()
    
    # Check that registered YCB and GSO assets exist on disk
    assert status["ycb_coffee_can"] is True
    assert status["ycb_sugar_box"] is True
    assert status["gso_coffee_jar"] is True


def test_asset_filtering():
    registry = AssetRegistry("configs/assets_ycb_gso.yaml")
    id_assets = registry.list_assets(split="id")
    assert len(id_assets) > 0

    utensils = registry.list_assets(role="distractor")
    assert len(utensils) > 0
