import yaml
import os
import json
import logging
from typing import Dict, Any, Union, Optional

from .models import MochiWorkerConfig

logger = logging.getLogger(__name__)

def _parse_value(value: str) -> Any:
    """Attempt to parse string value to bool, int, float, or keep as string."""
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

def _get_worker_root_dir() -> str:
    """Get the root directory of the worker package."""
    # Get the directory of this file (config/manager.py)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the worker root directory
    worker_root_dir = os.path.dirname(current_file_dir)
    return worker_root_dir

class ConfigurationManager:
    def __init__(self, default_config_path: Optional[str] = None):
        self.config: MochiWorkerConfig = MochiWorkerConfig()

        path_to_load = default_config_path
        if not path_to_load:
            # Look for worker_config.yaml in the worker's root directory instead of CWD
            worker_root_dir = _get_worker_root_dir()
            worker_config_path = os.path.join(worker_root_dir, "worker_config.yaml")
            logger.debug(f"No config path provided, looking for config at: {worker_config_path}")
            if os.path.exists(worker_config_path):
                path_to_load = worker_config_path
            else:
                logger.info("No explicit config path provided and 'worker_config.yaml' not found in worker root directory. Using Pydantic defaults/env vars initially.")
        elif not os.path.isabs(path_to_load) and not os.path.exists(path_to_load):
            # If the provided path is relative and doesn't exist in CWD, try worker root dir
            worker_root_dir = _get_worker_root_dir()
            worker_config_path = os.path.join(worker_root_dir, path_to_load)
            logger.debug(f"Relative config path provided, checking worker root: {worker_config_path}")
            if os.path.exists(worker_config_path):
                path_to_load = worker_config_path
                logger.info(f"Found config file in worker root directory: {worker_config_path}")
            else:
                logger.debug(f"Config file not found in worker root either: {worker_config_path}")

        if path_to_load:
            logger.info(f"Attempting to load configuration from: {path_to_load}")
            try:
                self.load_from_file(path_to_load)
            except FileNotFoundError:
                logger.warning(f"Configuration file not found at {path_to_load}. Using Pydantic defaults/env vars.")
            except ValueError as e:
                logger.warning(f"Error loading configuration file {path_to_load}: {e}. Using Pydantic defaults/env vars.")
        else:
            logger.info("No configuration file path specified or found by default. Relying on Pydantic defaults and environment variables for MochiWorkerConfig.")


    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update dictionary d with u."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def _update_config_and_revalidate(self, update_data: Dict[str, Any]):
        """Updates the internal config dict and re-validates with Pydantic model."""
        current_config_dict = self.config.model_dump()
        updated_dict = self._deep_update(current_config_dict, update_data)
        self.config = MochiWorkerConfig.model_validate(updated_dict)


    def load_from_file(self, file_path: str) -> MochiWorkerConfig:
        """Load configuration from a YAML or JSON file, merging with existing config."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                try:
                    config_data = yaml.safe_load(f)
                    if not isinstance(config_data, dict):
                        raise ValueError("YAML file did not yield a dictionary.")
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML file {file_path}: {e}")
            elif file_path.endswith('.json'):
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing JSON file {file_path}: {e}")
            else:
                raise ValueError("Configuration file must be YAML or JSON (e.g., .yaml, .yml, .json)")
        
        if config_data:
            self._update_config_and_revalidate(config_data)
        return self.config

    def get_config(self) -> MochiWorkerConfig:
        """Get the current, validated configuration object."""
        return self.config

    def update_runtime_config(self, config_updates: Dict[str, Any]) -> MochiWorkerConfig:
        """Update configuration with runtime overrides and re-validate."""
        self._update_config_and_revalidate(config_updates)
        return self.config

