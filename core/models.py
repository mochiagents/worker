# Pydantic models for core data structures used across Mochi.
"""This module defines the core Pydantic models used throughout the Mochi agent framework,
including data structures for tasks, agent state, tool schemas, and execution results.
These models provide data validation, serialization, and clear type hinting.
"""
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict
from ..core.logging import MochiLogger # ADDED IMPORT

# Forward declaration or ensure ToolExecutionResult is defined before TaskNode if not already.
# For now, assuming ToolExecutionResult will be defined later in the file, Pydantic handles forward references.

class TaskNode(BaseModel):
    """Represents a single node in the task Directed Acyclic Graph (DAG).
    Each node defines a specific tool to be called with given inputs and dependencies.
    """
    id: str = Field(..., description="Unique identifier for this task node within the DAG.")
    server_id: Optional[str] = Field(default=None, description="Identifier of the MCP server that hosts the tool. Can be null for non-tool tasks like 'direct_answer'.")
    tool_name: str = Field(..., description="Name of the tool to be executed on the specified server, or a special name like 'direct_answer'.")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of inputs for the tool. Values can be literals or references to outputs of other tasks (e.g., '$result.task_id.output_path').")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs that must be completed before this task can start.")
    title: Optional[str] = Field(None, description="Optional human-readable title or brief summary of the task.")
    priority: Optional[str] = Field("medium", description="Priority of the task (e.g., 'high', 'medium', 'low'). Defaults to 'medium'.")
    error: Optional[str] = Field(None, description="Error message if the task failed.")
    raw_output: Optional['ToolExecutionResult'] = Field(None, description="The raw output from the tool execution, stored as a ToolExecutionResult model.")
    status: str = Field(default="pending", description="Runtime status of the task (e.g., pending, ready_to_run, in_progress, completed_success, completed_failure, skipped).")
    execution_attempts: int = Field(default=0, description="Number of times execution has been attempted for this task.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "id": "task_2",
            "server_id": "weather_service_v1",
            "tool_name": "get_current_weather",
            "inputs": {"location": "London, UK", "units": "celsius"},
            "dependencies": ["task_1"],
            "title": "Fetch London Weather",
            "priority": "medium",
            "status": "pending",
            "execution_attempts": 0
        }
    })

class TaskDAG(BaseModel):
    """Represents the entire task Directed Acyclic Graph (DAG) for a given agent run.
    It consists of a list of TaskNode objects.
    """
    tasks: List[TaskNode] = Field(default_factory=list, description="List of task nodes defining the execution graph.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "tasks": [
                {
                    "id": "task_1", "server_id": "web_tools", "tool_name": "search", 
                    "inputs": {"query": "latest AI news"}, "dependencies": [], "title": "Search for AI News", "priority": "medium"
                },
                {
                    "id": "task_2", "server_id": "text_analysis", "tool_name": "summarize", 
                    "inputs": {"text": "$result.task_1.search_results[0].content"}, "dependencies": ["task_1"], "title": "Summarize First News Article", "priority": "high"
                }
            ]
        }
    })

class StructuredError(BaseModel):
    """A structured error object for more detailed error reporting."""
    error_type: str = Field(description="A specific error code or type, e.g., 'InputResolutionError', 'ToolTimeoutError'.")
    message: str = Field(description="A human-readable error message.")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary for additional error details, like problematic field names or values.")
    is_retryable: Optional[bool] = Field(None, description="Hint whether this error might be resolved by a simple retry.")
    is_repairable: Optional[bool] = Field(None, description="Hint whether this error might be resolved by DAG repair.")

class ToolExecutionResult(BaseModel):
    """Standardized result structure for tool execution via MCP."""
    task_id: str
    status: str # e.g., "success", "failure", "pending", "running"
    output: Optional[Any] = None
    error: Optional[Union[str, StructuredError]] = Field(None, description="Error message or structured error object if execution failed.")
    # Add other relevant fields like timestamps, logs, etc. as needed

    model_config = ConfigDict(json_schema_extra = {
        "example_success": {
            "task_id": "task_abc",
            "status": "success",
            "output": {"temperature": 25, "conditions": "sunny"}
        },
        "example_failure": {
            "task_id": "task_xyz",
            "status": "failure",
            "error": "API call timed out after 3 retries."
        }
    })

class Phase(BaseModel):
    """Represents a single high-level phase in a hierarchical plan."""
    phase_id: str = Field(..., description="Unique identifier for this phase.")
    description: str = Field(..., description="Detailed description of what this phase aims to achieve.")
    status: str = Field(default="pending", description="Runtime status of the phase (e.g., pending, in_progress, completed_success, completed_failure).")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "phase_id": "phase_1_research",
            "description": "Gather all relevant documents and prior art for the new product design.",
            "status": "pending"
        }
    })

class HierarchicalPlan(BaseModel):
    """Represents a high-level plan composed of multiple phases."""
    phases: List[Phase] = Field(default_factory=list, description="List of phases that make up the plan.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "phases": [
                {
                    "phase_id": "phase_1_research",
                    "description": "Gather all relevant documents and prior art for the new product design."
                },
                {
                    "phase_id": "phase_2_design",
                    "description": "Develop three distinct design mockups based on the research."
                }
            ]
        }
    })

class AgentState(BaseModel):
    """Represents the overall state of the Mochi agent during an execution run.
    This model is intended to be potentially serialized and deserialized to save/load agent progress.
    """
    original_query: Optional[str] = Field(None, description="The initial natural language query from the user.")
    hierarchical_plan: Optional[HierarchicalPlan] = Field(None, description="The high-level hierarchical plan, if applicable.")
    current_phase_id: Optional[str] = Field(None, description="ID of the currently executing phase in a hierarchical plan.") 
    current_phase_description: Optional[str] = Field(None, description="Description of the currently executing phase.") 
    # Stores accumulated outputs from all previously completed phases, keyed by GLOBAL task ID.
    accumulated_global_task_outputs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Outputs from all completed phases, mapping GLOBAL_TASK_ID to its actual output value. Used for cross-phase $result resolution.")
    
    # NEW FIELDS START
    conversation_context: Optional[str] = Field(None, description="The conversation history leading up to the current query.")
    config: Optional[Dict[str, Any]] = Field(None, exclude=True, description="Runtime configuration for the agent.")
    run_id: Optional[str] = Field(None, description="The unique ID for the current agent run.")
    
    planner_instance: Optional[Any] = Field(None, exclude=True, description="Instance of the Planner.")
    task_fetching_unit_instance: Optional[Any] = Field(None, exclude=True, description="Instance of the TaskFetchingUnit.")
    joiner_instance: Optional[Any] = Field(None, exclude=True, description="Instance of the Joiner.")
    mcp_clients: Optional[Dict[str, Any]] = Field(None, exclude=True, description="Dictionary of MCP client instances.")
    dag_editor_instance: Optional[Any] = Field(None, exclude=True, description="Instance of the DAGEditor.")
    # NEW FIELDS END
    
    logger: Optional[MochiLogger] = Field(default=None, exclude=True, description="Instance of MochiLogger for logging within graph execution. Excluded from serialization.") # ADDED LOGGER FIELD HERE

    replan_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contextual information for replanning, such as previous errors or partial results.") # ADDED FIELD

    dag_repair_error: Optional[str] = Field(None, description="Error message specifically from a DAG repair attempt, if any.") # ADDED FIELD

    dag_repair_instructions: Optional[Dict[str, Any]] = Field(default=None, description="Instructions for DAG repair, if a repair attempt failed and needs to be retried with modifications.") # ADDED FIELD
    current_task_id_to_execute: Optional[str] = Field(None, description="The ID of the specific task that the executor should run next. Used by specific execution strategies.") # ADDED FIELD

    planner_output: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Output from the planner node, can contain query_type or error info.") # ADDED FIELD

    joiner_error: Optional[str] = Field(None, description="Error message from the joiner node, if any.") # ADDED FIELD
    repair_instructions_available: bool = Field(False, description="Flag indicating if repair instructions are available from the joiner.") # ADDED FIELD

    planner_error: Optional[str] = Field(None, description="Error message specifically from the planner node, if any.") # ADDED FIELD

    task_dag: Optional[TaskDAG] = Field(None, description="The Pydantic model representando the current task DAG (could be for a single phase or the entire plan).")
    task_results: Dict[str, ToolExecutionResult] = Field(default_factory=dict, description="Dictionary mapping LOCAL task IDs (within the current DAG) to their execution results (ToolExecutionResult models).")
    task_statuses: Dict[str, str] = Field(default_factory=dict, description="Dictionary mapping LOCAL task IDs (within the current DAG) to their current status string (e.g., pending, in_progress, completed, failed).")
    final_response: Optional[str] = None
    error_message: Optional[str] = None
    all_completed: bool = False
    has_ready_tasks: bool = False
    execution_error: Optional[str] = None
    current_iteration_log: List[str] = Field(default_factory=list)
    replanning_cycles: int = Field(default=0, description="Number of replanning cycles attempted for the current query/phase DAG.")
    overall_status: str = Field(default="starting", description="Overall status of the agent for the query (e.g., starting, planning, executing_phase, requires_clarification, failed, completed)") # Added
    needs_replanning: bool = Field(False, description="Flag from Joiner if current plan failed and needs replanning.") # MODIFIED: Made non-optional
    
    # max_replanning_cycles_for_query: int = Field(default=3, description="Overall replanning limit for the entire query across all phases.") # Consider if needed
    # current_query_replanning_attempts: int = Field(default=0, description="Total replanning attempts for the current query.") # Consider if needed

    model_config = ConfigDict(
        arbitrary_types_allowed=True, # ENSURE this is True
        json_schema_extra = {
        "example": {
            "original_query": "Plan a trip to Paris and book a flight.",
            "hierarchical_plan": {"phases": [{"phase_id": "phase_1_research", "description": "Research Paris options", "status": "pending"}]}, 
            "current_phase_id": "phase_1_research",
            "current_phase_description": "Research Paris options",
            "accumulated_global_task_outputs": {"intro_task_abc": "Welcome message generated"},
            "task_dag": {"tasks": [{"id": "task_1", "server_id": "planner", "tool_name": "create_trip_plan", "inputs": {"destination": "Paris"}}]}, 
            "task_results": {},
            "task_statuses": {"task_1": "pending"},
            "final_response": None,
            "error_message": None,
            "all_completed": False,
            "has_ready_tasks": False,
            "execution_error": None,
            "current_iteration_log": [],
            "replanning_cycles": 0,
            "overall_status": "planning"
        }
    })

class ToolParameter(BaseModel):
    """Describes a single parameter within a tool's input or output JSON schema.
    This can be used for dynamic form generation or detailed tool documentation.
    """
    name: str = Field(..., description="The name of the parameter.")
    type: str = Field(..., description="The JSON schema type of the parameter (e.g., string, integer, boolean, object, array).")
    description: Optional[str] = Field(None, description="A human-readable description of what the parameter is for.")
    required: bool = Field(False, description="Whether this parameter is required for the tool to function.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "name": "location",
            "type": "string",
            "description": "The city and state, e.g., San Francisco, CA",
            "required": True
        }
    })

class ToolSchema(BaseModel):
    """Describes the schema for a single tool, including its name, description, and input/output JSON schemas.
    """
    tool_name: str = Field(..., description="The unique name of the tool, corresponding to TaskNode.tool_name.")
    description: Optional[str] = Field(None, description="A human-readable description of what the tool does.")
    input_schema: List[ToolParameter] = Field(default_factory=list, alias="inputs", description="List of input parameter definitions for the tool.") 
    output_schema: Dict[str, Any] = Field(default_factory=dict, alias="output", description="JSON schema definition for the tool's output.")

    model_config = ConfigDict(
        populate_by_name=True, # Allows use of alias for population
        json_schema_extra = {
        "example": {
            "tool_name": "get_weather",
            "description": "Fetches the current weather for a given location.",
            "inputs": [
                {
                    "name": "location", 
                    "type": "string", 
                    "description": "City to get weather for",
                    "required": True
                },
                {
                    "name": "units", 
                    "type": "string", 
                    "description": "Temperature units (e.g., celsius, fahrenheit)",
                    "required": False
                }
            ],
            "output": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "integer"},
                    "conditions": {"type": "string"}
                }
            }
        }
    })

class ServerToolSchemaGroup(BaseModel):
    """Groups tool schemas by the MCP server that provides them.
    This is useful for presenting available tools to the Planner or a UI.
    """
    server_id: str = Field(..., description="Unique identifier of the MCP server.")
    description: Optional[str] = Field(None, description="Optional description of the server or the group of tools it provides.")
    tools: List[ToolSchema] = Field(default_factory=list, description="List of ToolSchema objects available on this server.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "server_id": "weather_tools_v2",
            "description": "Provides weather and forecast information.",
            "tools": [
                {
                    "tool_name": "get_current_weather", "description": "Gets current weather.",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
                }
            ]
        }
    })

# --- BEGIN DAG REPAIR ACTION MODELS ---

class RepairActionBase(BaseModel):
    action_type: Literal[
        "MODIFY_TASK_INPUTS", 
        "MODIFY_TASK_TOOL", 
        "MODIFY_TASK_DEPENDENCIES", 
        "ADD_TASK", 
        "DELETE_TASK",
        "NO_REPAIR_POSSIBLE"
    ]

class ModifyTaskInputsRepairAction(RepairActionBase):
    action_type: Literal["MODIFY_TASK_INPUTS"] = "MODIFY_TASK_INPUTS"
    task_id: str = Field(..., description="The ID of the task whose inputs are to be modified.")
    updated_inputs: Dict[str, Any] = Field(..., description="The new set of inputs for the task.")

class ModifyTaskToolRepairAction(RepairActionBase):
    action_type: Literal["MODIFY_TASK_TOOL"] = "MODIFY_TASK_TOOL"
    task_id: str = Field(..., description="The ID of the task whose tool is to be modified.")
    new_tool_name: str = Field(..., description="The new tool name for the task.")
    new_server_id: Optional[str] = Field(None, description="The new server ID for the task, if applicable.")
    updated_inputs: Optional[Dict[str, Any]] = Field(None, description="Optional new inputs if the tool change requires it.")

class ModifyTaskDependenciesRepairAction(RepairActionBase):
    action_type: Literal["MODIFY_TASK_DEPENDENCIES"] = "MODIFY_TASK_DEPENDENCIES"
    task_id: str = Field(..., description="The ID of the task whose dependencies are to be modified.")
    updated_dependencies: List[str] = Field(..., description="The new list of dependency task IDs.")

class AddTaskRepairAction(RepairActionBase):
    action_type: Literal["ADD_TASK"] = "ADD_TASK"
    # Reuses the existing TaskNode model for defining the new task.
    # The LLM will need to be instructed to fill this out completely, 
    # including a unique ID, server_id (if applicable), tool_name, inputs, and dependencies.
    new_task_definition: TaskNode = Field(..., description="The full definition of the new task to be added.")

class DeleteTaskRepairAction(RepairActionBase):
    action_type: Literal["DELETE_TASK"] = "DELETE_TASK"
    task_id: str = Field(..., description="The ID of the task to be deleted.")

class NoRepairPossibleAction(RepairActionBase):
    action_type: Literal["NO_REPAIR_POSSIBLE"] = "NO_REPAIR_POSSIBLE"
    reason: Optional[str] = Field(None, description="Optional reason why no repair is possible.")

# Union of all possible repair action types
AnyRepairAction = Union[
    ModifyTaskInputsRepairAction,
    ModifyTaskToolRepairAction,
    ModifyTaskDependenciesRepairAction,
    AddTaskRepairAction,
    DeleteTaskRepairAction,
    NoRepairPossibleAction
]

class DAGRepairActions(BaseModel):
    """Top-level model for a list of DAG repair actions proposed by the LLM."""
    repair_actions: List[AnyRepairAction] = Field(..., description="A list of repair actions to be applied to the DAG.")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "repair_actions": [
                {
                    "action_type": "MODIFY_TASK_INPUTS",
                    "task_id": "task_search_china_gdp",
                    "updated_inputs": {"query": "China GDP per capita 2023"}
                },
                {
                    "action_type": "ADD_TASK",
                    "new_task_definition": {
                        "id": "task_extract_gdp_value",
                        "server_id": None, # Or appropriate server_id
                        "tool_name": "direct_answer",
                        "inputs": {"answer_text": "Extract the numerical GDP value from $result.task_search_china_gdp.results[0].content"},
                        "dependencies": ["task_search_china_gdp"],
                        "title": "Extract GDP Value from Search"
                    }
                }
            ]
        }
    })

# --- END DAG REPAIR ACTION MODELS ---

# Explicitly rebuild models to resolve forward references.

# Rebuild in an order that respects dependencies for forward references:
# ToolExecutionResult is referenced by TaskNode.
# TaskNode is part of TaskDAG.
# TaskDAG and ToolExecutionResult are part of AgentState.

# For other models, rebuild if they have forward refs or complex nesting.
StructuredError.model_rebuild()
ToolParameter.model_rebuild()
ToolSchema.model_rebuild()
ServerToolSchemaGroup.model_rebuild()
Phase.model_rebuild()
HierarchicalPlan.model_rebuild()

# Critical rebuild order for the issue at hand:
ToolExecutionResult.model_rebuild() # Definition that TaskNode.raw_output refers to
TaskNode.model_rebuild()          # Contains forward ref: raw_output: Optional['ToolExecutionResult']
TaskDAG.model_rebuild()           # Contains List[TaskNode]
AgentState.model_rebuild()        # Contains TaskDAG and task_results: Dict[str, ToolExecutionResult]

# Repair action models (rebuild if they also have internal forward refs or complex structures)
ModifyTaskInputsRepairAction.model_rebuild()
ModifyTaskToolRepairAction.model_rebuild()
ModifyTaskDependenciesRepairAction.model_rebuild()
AddTaskRepairAction.model_rebuild() # new_task_definition: TaskNode
DeleteTaskRepairAction.model_rebuild()
NoRepairPossibleAction.model_rebuild()
DAGRepairActions.model_rebuild()