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

# Global instance of the Configuration Manager
# It will automatically look for worker_config.yaml in the worker's root directory
config_manager = ConfigurationManager()

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
