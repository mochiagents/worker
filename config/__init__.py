from .manager import ConfigurationManager
# from .schema import MochiWorkerConfig # Old import
from .models import (
    MochiWorkerConfig, # New import from models.py
    LLMConfigEntry as LLMConfig, # Alias for compatibility or direct use
    MCPToolServerConfig as MCPToolConfig, # Alias for compatibility or direct use
    ExecutionSettings as ExecutionConfig, # Alias for compatibility or direct use
    LoggingSettings as LoggingConfig, # Alias for compatibility or direct use
    PlannerSettings, # Added PlannerSettings
    JoinerSettings # Added JoinerSettings
)

# Define the default configuration file path relative to the project root
# This assumes the application is run from the project root directory.
# Adjust if the execution context is different.
DEFAULT_CONFIG_FILE = "worker_config.yaml" 

# Global instance of the Configuration Manager
# It will first load Pydantic defaults, then try the DEFAULT_CONFIG_FILE,
# and then can be further updated by environment variables or runtime calls.
config_manager = ConfigurationManager(default_config_path=DEFAULT_CONFIG_FILE)

# Optionally, load environment variables at startup.
# Services can also choose to call this explicitly if needed, or it can be part of app bootstrap.

def get_settings() -> MochiWorkerConfig:
    """Provides access to the globally managed MochiWorkerConfig instance."""
    return config_manager.get_config()

# Expose specific models if they are commonly imported via mochi.config
# This makes them available like: from mochi.config import LLMConfig
# from .schema import LLMConfig, MCPToolConfig, ExecutionConfig, LoggingConfig # Old imports

__all__ = [
    "config_manager",
    "get_settings",
    "MochiWorkerConfig",
    "LLMConfig", # Now aliased from models.LLMConfigEntry
    "MCPToolConfig", # Now aliased from models.MCPToolServerConfig
    "ExecutionConfig", # Now aliased from models.ExecutionSettings
    "LoggingConfig", # Now aliased from models.LoggingSettings
    "PlannerSettings", # Added PlannerSettings
    "JoinerSettings" # Added JoinerSettings
]
