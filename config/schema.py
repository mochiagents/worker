from pydantic import Field, field_validator, BaseModel
from typing import Dict, List, Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseModel):
    """Configuration for an LLM instance."""
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class MCPToolConfig(BaseModel):
    """Configuration for an MCP Tool Server registration."""
    server_id: str
    endpoint: str
    schema_endpoint: Optional[str] = None
    timeout: int = 30
    retries: int = 3


class ExecutionConfig(BaseModel):
    """Configuration for task execution parameters."""
    max_parallel_tasks: int = 5
    task_timeout: int = 60
    max_retries: int = 2

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    log_llm_calls: bool = True
    log_mcp_calls: bool = True
    log_file: Optional[str] = None

    @field_validator('level')
    @classmethod
    def validate_logging_level(cls, value: str) -> str:
        """Validate that the logging level is a recognized string."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value.upper() not in valid_levels:
            raise ValueError(
                f'Invalid logging level "{value}". Must be one of {valid_levels}.'
            )
        return value.upper()

class MochiWorkerConfig(BaseSettings):
    """Root configuration model for the Mochi Worker agent.
    Inherits from BaseSettings to automatically load from .env and environment variables.
    """
    agent_id: str = "mochi-worker-default"
    
    planner_llm: LLMConfig = Field(default_factory=LLMConfig)
    joiner_llm: LLMConfig = Field(default_factory=LLMConfig)
    
    mcp_tools: List[MCPToolConfig] = Field(default_factory=list)
    
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    additional_settings: Dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_prefix="MOCHI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
