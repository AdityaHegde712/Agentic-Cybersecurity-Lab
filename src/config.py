"""Central configuration and path resolution module for local and HPC environments.

Hierarchy:
1. Explicit CLI arguments (passed by callers)
2. Environment variables:
   - CYBER_PROCESSED_DIR: Path to processed parquet store directory
   - CYBER_RAW_HAI_DIR / HAI_RAW_DIR: Path to raw HAI-20.07 CSV directory
   - CYBER_CONFIG_PATH: Path to custom YAML config file
3. configs/default.yaml configuration file
4. Repository default relative paths
"""

from pathlib import Path
import os
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Lightweight fallback parser for basic nested YAML (no external deps)."""
    result: Dict[str, Any] = {}
    current_section = None

    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip().strip('"').strip("'")
            if not val:
                current_section = key
                if current_section not in result:
                    result[current_section] = {}
            else:
                if current_section:
                    result[current_section][key] = val
                else:
                    result[key] = val
    return result


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return built-in defaults."""
    target = config_path or Path(os.environ.get("CYBER_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not target.exists():
        return {}

    try:
        import yaml  # type: ignore
        with open(target, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        with open(target, "r", encoding="utf-8") as fh:
            return _parse_simple_yaml(fh.read())


_CONFIG = load_config()


def get_processed_dir() -> Path:
    """Resolve processed parquet dataset directory with environment & config precedence."""
    # 1. Environment variable override
    env_dir = os.environ.get("CYBER_PROCESSED_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # 2. Config file entry
    data_cfg = _CONFIG.get("data", {})
    if isinstance(data_cfg, dict) and "processed_dir" in data_cfg:
        cfg_val = data_cfg["processed_dir"]
        path = Path(cfg_val)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    # 3. Default relative to repository
    return (PROJECT_ROOT / "data" / "processed").resolve()


def get_raw_hai_dir() -> Path:
    """Resolve raw HAI-20.07 CSV directory with environment & config precedence."""
    # 1. Environment variable override
    env_dir = os.environ.get("CYBER_RAW_HAI_DIR") or os.environ.get("HAI_RAW_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # 2. Config file entry
    data_cfg = _CONFIG.get("data", {})
    if isinstance(data_cfg, dict) and "raw_hai_dir" in data_cfg:
        cfg_val = data_cfg["raw_hai_dir"]
        path = Path(cfg_val)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    # 3. Default relative to repository
    return (PROJECT_ROOT / "data" / "hai" / "raw" / "hai-20.07").resolve()


def get_hpc_config() -> Dict[str, Any]:
    """Return HPC allocation and queue metadata."""
    return _CONFIG.get("hpc", {
        "allocation_id": "TG-NAIRR260371",
        "system": "vista",
        "queues": {
            "gpu_prod": "gh",
            "gpu_dev": "gh-dev",
            "cpu_prod": "gg",
        }
    })
