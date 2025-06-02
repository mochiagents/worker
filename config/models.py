from typing import Optional, Dict, Any, Literal, List, Set
from pydantic import BaseModel, Field

# --- Default Stop Words (can be overridden by config) ---
DEFAULT_SEMANTIC_STOP_WORDS: Set[str] = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can',
    'could', 'may', 'might', 'must', 'and', 'or', 'but', 'if', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'just', 'don', 'shouldv', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'us', 'him', 'them'
}
# --- End Default Stop Words ---

class LLMConfigEntry(BaseModel):
    """Defines a configuration profile for an LLM instance."""
    provider: Literal["openai", "anthropic", "google", "ollama", "azure", "mock"] = Field(..., description="The LLM provider.")
    model: str = Field(..., description="The specific model name (e.g., 'gpt-4-turbo', 'claude-3-opus-20240229', 'mistral').")
    
    api_key: Optional[str] = Field(None, description="API key for the LLM provider. If None, it might be sourced from environment variables.")
    base_url: Optional[str] = Field(None, description="Base URL for custom or local LLM providers (e.g., Ollama, vLLM).")
    
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature for the LLM.")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum number of tokens to generate.")
    
    azure_api_version: Optional[str] = Field(None, description="Azure OpenAI API version (e.g., '2023-05-15'). Required if provider is 'azure'.")
    azure_deployment: Optional[str] = Field(None, description="Azure OpenAI deployment name. Required if provider is 'azure'.")

    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters.")

    class Config:
        extra = 'forbid'

class MCPToolServerConfig(BaseModel):
    """Configuration for a single MCP Tool Server that Mochi can connect to."""
    name: str = Field(..., description="A unique, descriptive name for this tool server configuration (e.g., 'web_search_v1', 'code_interpreter_prod'). Used as server_id.")
    endpoint_url: str = Field(..., description="The base HTTP(S) URL for the MCP Tool Server where RPC calls are made (e.g. /mcp).")
    description: Optional[str] = Field(None, description="A brief description of the tool server and its capabilities.")
    
    schema_discovery_url: Optional[str] = Field(None, description="Specific URL to fetch the tool schema document (e.g., OpenAPI spec or direct tools list) via HTTP GET. If set, this URL is tried first for schema discovery.")
    request_timeout_seconds: Optional[float] = Field(None, gt=0, description="Timeout in seconds for individual requests to this server. Overrides the McpClient's default timeout_request if set.")
    max_retries: Optional[int] = Field(None, ge=0, description="Maximum number of retries for failing requests (both HTTP GET for schema and RPC calls) to this server.")
    retry_delay_seconds: Optional[float] = Field(None, gt=0, description="Initial delay in seconds before the first retry for requests to this server. Subsequent retries might use a backoff strategy.")

    auth_token: Optional[str] = Field(None, description="Bearer token for authentication with this MCP server, sent as 'Authorization: Bearer <token>'.")
    custom_headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Custom HTTP headers to send with all requests to this MCP server (e.g., API keys as headers). These will be merged with auth_token if both are present.")
    
    mock_tool_schemas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional list of mock tool schemas (in dict format) this server would provide. Used if the actual server is unavailable or for testing planner logic.")

    class Config:
        extra = 'forbid'

class McpClientDefaultSettings(BaseModel):
    """Default settings for the MCP Client if not overridden by server-specific configs."""
    default_request_timeout_seconds: float = Field(default=10.0, gt=0, description="Default timeout for HTTP requests to MCP servers.")
    default_connect_timeout_seconds: float = Field(default=5.0, gt=0, description="Default timeout for establishing a connection to MCP servers.")
    sse_heartbeat_timeout_seconds: float = Field(default=300.0, gt=0, description="Timeout for SSE connection heartbeats.")
    sse_max_retries: int = Field(default=3, ge=0, description="Maximum retries for SSE connections.")
    sse_retry_delay_seconds: float = Field(default=1.0, gt=0, description="Initial delay for SSE connection retries.")

    class Config:
        extra = 'forbid'

class ExecutionSettings(BaseModel):
    """Settings related to task execution within the Mochi agent."""
    max_parallel_tasks: int = Field(default=3, gt=0, description="Maximum number of tasks the agent can execute in parallel.")
    max_retries: int = Field(default=2, ge=0, description="Default maximum number of retries for a failing MCP tool call, if not overridden per server.")
    initial_retry_delay_seconds: float = Field(default=1.0, gt=0, description="Default initial delay in seconds before the first retry for MCP tool calls, if not overridden per server.")
    retry_backoff_factor: float = Field(default=2.0, gt=1.0, description="Default multiplier for increasing retry delay (e.g., 2.0 for exponential backoff), if not overridden per server.")
    default_task_execution_timeout_seconds: float = Field(default=60.0, gt=0, description="Default timeout for a single task execution.")
    min_task_execution_timeout_seconds: float = Field(default=5.0, gt=0, description="Minimum allowable timeout for a single task execution.")
    max_task_execution_timeout_seconds: float = Field(default=300.0, gt=0, description="Maximum allowable timeout for a single task execution.")
    
    class Config:
        extra = 'forbid'

class PlannerSettings(BaseModel):
    """Settings specific to the Planner component."""
    llm_profile_name: Optional[str] = Field(None, description="Name of the LLM profile (from top-level llm_profiles) to use for the Planner.")
    use_tool_recommendation: bool = Field(True, description="Whether the planner should use an LLM to recommend relevant tools before full DAG generation.")
    tool_recommendation_llm_profile_name: Optional[str] = Field(None, description="LLM profile for tool recommendation, if different from main planner LLM. If None, uses planner's llm_profile_name.")
    embedding_model_name: Optional[str] = Field(
        default='all-MiniLM-L6-v2', 
        description="Name of the sentence-transformer model to use for semantic tool recommendation. Set to None to potentially disable semantic search if embedding model fails to load."
    )
    semantic_search_include_server_description: bool = Field(
        default=False,
        description="If True, the server's own description will also be embedded and scored during semantic tool recommendation, in addition to its tools."
    )
    semantic_stop_words: Optional[List[str]] = Field(
        default=None,
        description="List of stop words for semantic processing. If null, internal defaults are used."
    )
    tool_recommendation_top_n: int = Field(
        default=5, 
        ge=1,
        description="Number of top server schemas to recommend based on semantic scores."
    )
    semantic_score_min_threshold: float = Field(
        default=0.1,
        ge=0.0, le=1.0,
        description="Minimum semantic similarity score for a tool/server to be considered relevant (0.0 to 1.0)."
    )
    default_planning_strategy: Optional[Literal["simple_sequential", "tool_recommendation_first"]] = Field(
        default="tool_recommendation_first", 
        description="Default strategy for planning. 'simple_sequential' might try to directly generate a full plan. 'tool_recommendation_first' emphasizes using tool recommendation before DAG construction."
    )
    max_dag_depth: Optional[int] = Field(default=5, ge=1, description="Maximum depth of the task dependency graph the planner should aim to create. Helps prevent overly complex initial plans.")
    max_sequential_tasks_per_tool: Optional[int] = Field(default=1, ge=1, description="Maximum number of sequential tasks using the exact same tool the planner should generate. Encourages tool diversity or breaking down large tool use into distinct steps.")

    class Config:
        extra = 'forbid'

class JoinerSettings(BaseModel):
    """Settings specific to the Joiner component."""
    llm_profile_name: Optional[str] = Field(None, description="Name of the LLM profile (from top-level llm_profiles) to use for the Joiner.")
    max_replanning_triggers: int = Field(default=1, ge=0, le=5, description="Maximum number of times the Joiner can explicitly trigger a replanning phase. This is separate from the agent's global max_replanning_loops.")
    joiner_response_format_instructions: Optional[str] = Field(None, description="Optional custom instructions for the Joiner's LLM on how to format its response, if the default internal prompt needs adjustment for specific LLMs or use cases.")

    class Config:
        extra = 'forbid'

class LoggingSettings(BaseModel):
    """Configuration for logging within the Mochi agent."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level for the Mochi agent.")
    format: Optional[str] = Field(default="%(asctime)s - %(name)s - %(levelname)s - [%(event_type)s] - %(message)s", alias="format", description="Logging format string.")
    date_format: Optional[str] = Field(default="%Y-%m-%d %H:%M:%S", alias="date_format", description="Date format string for logs.")
    log_to_console: bool = Field(default=True, description="Whether to output logs to the console (stdout/stderr).")
    log_file: Optional[str] = Field(None, description="Path to a log file. If None, logs to console only.")
    log_file_max_bytes: int = Field(default=10485760, gt=0, description="Maximum size in bytes for a log file before rotation.")
    log_file_backup_count: int = Field(default=3, ge=0, description="Number of backup log files to keep during rotation.")
    log_mcp_calls: bool = Field(default=True, description="Enable detailed logging of MCP request/response payloads.")
    
    class Config:
        extra = 'forbid'
        populate_by_name = True

class AgentGeneralSettings(BaseModel):
    """General operational settings for the Mochi agent."""
    max_replanning_loops: int = Field(default=3, ge=0, description="Maximum number of replanning loops the agent will attempt before stopping.")
    max_dag_generation_attempts: int = Field(default=3, ge=0, description="Maximum number of DAG generation attempts the agent will attempt before stopping.")
    max_context_tokens: int = Field(default=4096, ge=128, description="Maximum number of tokens from conversation context to include in prompts for LLMs.")
    state_persistence_dir: Optional[str] = Field(None, description="Directory to persist agent states. If None, state is in-memory only.")
    max_replanning_cycles: int = Field(default=3, ge=0, description="Maximum number of replanning cycles allowed for a single user query (potentially distinct from global loops).")
    include_debug_info_in_response: bool = Field(default=False, description="Whether to include detailed debug information in the final response.")
    enable_heartbeat: bool = Field(default=False, description="If true, the agent will send periodic heartbeats (implementation specific).")
    heartbeat_interval_seconds: int = Field(default=60, gt=0, description="Interval in seconds for agent heartbeats (if enabled).")

    class Config:
        extra = 'forbid'

class MochiWorkerConfig(BaseModel):
    """Root configuration model for the Mochi worker agent."""
    agent_id: str = Field(default="mochi-worker-default-01", description="A unique identifier for this Mochi worker agent instance.")
    
    llm_profiles: Dict[str, LLMConfigEntry] = Field(default_factory=dict, description="Dictionary of named LLM configuration profiles.")
    
    mcp_tool_servers: List[MCPToolServerConfig] = Field(default_factory=list, description="List of configured MCP Tool Servers the agent can connect to.")
    
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings, description="Settings for task execution.")
    planner: PlannerSettings = Field(default_factory=PlannerSettings, description="Settings for the Planner component.")
    joiner: JoinerSettings = Field(default_factory=JoinerSettings, description="Settings for the Joiner component.")
    logging: LoggingSettings = Field(default_factory=LoggingSettings, description="Logging configuration.")
    agent_settings: AgentGeneralSettings = Field(default_factory=AgentGeneralSettings, description="General agent operational settings.")
    mcp_client_defaults: McpClientDefaultSettings = Field(default_factory=McpClientDefaultSettings, description="Default settings for MCP client connections.")

    class Config:
        extra = 'forbid'
        validate_assignment = True

    def get_llm_profile(self, profile_name: str) -> Optional[LLMConfigEntry]:
        return self.llm_profiles.get(profile_name)

    def get_mcp_tool_server_config(self, server_name_or_id: str) -> Optional[MCPToolServerConfig]:
        for server_config in self.mcp_tool_servers:
            if server_config.name == server_name_or_id:
                return server_config
        return None
