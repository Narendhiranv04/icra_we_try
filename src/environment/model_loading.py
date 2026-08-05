"""
Model loading utilities for MuJoCo scene construction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mujoco


def load_assets_from_dir(assets_dir: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, bytes]:
    """Load binary asset files (OBJ, STL, PNG, etc.) into a dictionary for MuJoCo XML string parsing.
    
    Args:
        assets_dir: Single path or list of paths to directories containing assets.
        
    Returns:
        Dict mapping relative file paths to raw bytes.
    """
    asset_dict = {}
    dirs_to_search = []
    
    if isinstance(assets_dir, (list, tuple)):
        dirs_to_search.extend([Path(d) for d in assets_dir])
    else:
        dirs_to_search.append(Path(assets_dir))

    # Add gymnasium_robotics fetch assets if available
    try:
        import gymnasium_robotics
        p_base = Path(gymnasium_robotics.__file__).parent / "envs" / "assets"
        p_fetch = p_base / "fetch"
        p_stls = p_base / "stls" / "fetch"
        if p_fetch.exists():
            dirs_to_search.append(p_fetch)
        if p_stls.exists():
            dirs_to_search.append(p_stls)
    except ImportError:
        pass

    for d in dirs_to_search:
        if d.exists():
            for file_path in d.rglob("*"):
                if file_path.is_file():
                    name = file_path.name
                    if name not in asset_dict:
                        with open(file_path, "rb") as f:
                            asset_dict[name] = f.read()
    return asset_dict



def load_model_from_path(
    xml_path: Union[str, Path],
    assets_dir: Optional[Union[str, Path]] = None,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MjModel and MjData directly from an XML file path.
    
    Args:
        xml_path: Absolute or relative path to MJCF XML file.
        assets_dir: Optional directory containing referenced mesh/texture assets.
        
    Returns:
        Tuple of (mujoco.MjModel, mujoco.MjData).
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {xml_path}")
        
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_string = f.read()
        
    base_dir = assets_dir if assets_dir else xml_path.parent
    assets = load_assets_from_dir(base_dir)
    
    if assets:
        model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
    else:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        
    data = mujoco.MjData(model)
    return model, data


def load_model_from_string(
    xml_string: str,
    assets: Optional[Dict[str, bytes]] = None,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MjModel and MjData from an XML string.
    
    Args:
        xml_string: MJCF XML string content.
        assets: Dict mapping filename relative paths to binary content bytes.
        
    Returns:
        Tuple of (mujoco.MjModel, mujoco.MjData).
    """
    if assets:
        model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
    else:
        model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data
