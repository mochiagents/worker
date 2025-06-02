from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, Literal, List
from enum import Enum

# Pydantic models for Mochi's external API layer.
"""This module defines Pydantic models used for request and response validation
and serialization in Mochi's external-facing API (e.g., a FastAPI application).
It includes models for submitting tasks/queries to the agent, checking status,
and retrieving results, as well as models for worker-style task management if applicable.
"""

# --- Enums for Status and Actions ---

class TaskStatus(str, Enum):
    """Possible statuses for a task managed by a worker or a sub-component of the agent."""
    ACCEPTED = "accepted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ControlAction(str, Enum):
    """Allowed lifecycle control actions for the worker."""
    PAUSE = "pause"
    RESUME = "resume"
    SHUTDOWN = "shutdown"

# --- Request Models ---

class TaskRequest(BaseModel):
    """Schema for submitting a new pre-defined task to a worker component.
    This is distinct from submitting a general query to the main Mochi agent.
    """
    task_id: str = Field(..., description="Unique identifier for the task.")
    description: str = Field(..., description="Description of the task to be performed.")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context or parameters required for the task.")
    priority: Optional[str] = Field("medium", description="Task priority (e.g., high, medium, low).")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "task_id": "task-12345",
            "description": "Summarize the provided document at resource_uri mcp://example/doc.txt",
            "context": {"resource_uri": "mcp://example/doc.txt"},
            "priority": "high"
        }
    })

class ControlActionRequest(BaseModel):
    """Schema for sending a lifecycle control action (e.g., pause, resume) to a worker component."""
    action: ControlAction = Field(..., description="The control action to perform.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "action": "pause"
        }
    })

class InitializeRequest(BaseModel):
    """Schema for initializing or reconfiguring a worker component with specific settings."""
    config: Dict[str, Any] = Field(..., description="Configuration dictionary for the worker.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "config": {
                "llm_model": "claude-3-opus-20240229",
                "max_retries": 3
            }
        }
    })

# --- Response Models ---

class TaskResponse(BaseModel):
    """Schema for simple responses related to worker task submissions (e.g., acceptance)."""
    task_id: str = Field(..., description="Identifier of the task.")
    status: TaskStatus = Field(..., description="Current status of the task submission.")
    message: Optional[str] = Field(None, description="Optional message regarding the task status.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "task_id": "task-12345",
            "status": "accepted",
            "message": "Task accepted for processing."
        }
    })

class TaskResult(BaseModel):
    """Schema for returning the final result or error of a worker-style task."""
    task_id: str = Field(..., description="Identifier of the task.")
    status: TaskStatus = Field(..., description="Final status of the task (typically COMPLETED or FAILED).")
    result: Optional[Dict[str, Any]] = Field(None, description="The result output from the task if successful.")
    error: Optional[str] = Field(None, description="Error message if the task failed.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "task_id": "task-12345",
            "status": "completed",
            "result": {"summary": "The document discusses..."},
            "error": None
        }
    })

class ControlActionResponse(BaseModel):
    """Schema for confirming the outcome of a control action on a worker component."""
    status: str = Field(..., description="Message indicating the result of the control action.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "status": "pause successful"
        }
    })

class InitializeResponse(BaseModel):
    """Schema for confirming worker component initialization or reconfiguration."""
    status: Literal["initialized", "already_initialized", "reconfigured"] = Field(..., description="Status of the initialization or reconfiguration attempt.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "status": "initialized"
        }
    })

# --- Agent Interaction API Models ---

class AgentQueryRequest(BaseModel):
    """Schema for submitting a general query to the Mochi agent for planning and execution."""
    query: str = Field(..., min_length=1, description="The natural language query or task for the agent.")
    user_id: Optional[str] = Field(None, description="Optional user identifier.")
    session_id: Optional[str] = Field(None, description="Optional session identifier for context.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "query": "What is the weather in London and what is 2+2?",
            "user_id": "user-abc-123",
            "session_id": "session-xyz-789"
        }
    })

class AgentQueryResponse(BaseModel):
    """Schema for the immediate response after submitting a query to the Mochi agent."""
    agent_run_id: str = Field(..., description="Unique identifier for this agent run.")
    status: str = Field(..., description="Status of the query submission, e.g., 'received', 'queued'.")
    message: Optional[str] = Field(None, description="Optional message regarding the submission.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "agent_run_id": "run-f0ec-4a27-b3c8-adea120ddf71",
            "status": "received",
            "message": "Query received and queued for processing."
        }
    })

class AgentRunStatus(str, Enum):
    """Possible statuses for an agent run."""
    RECEIVED = "received"
    PLANNING = "planning"
    EXECUTING = "executing"
    JOINING = "joining_results"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentRunStatusResponse(BaseModel):
    """Schema for querying the status of an ongoing or completed agent run."""
    agent_run_id: str = Field(..., description="Identifier of the agent run.")
    overall_status: AgentRunStatus = Field(..., description="Current overall status of the agent run.")
    current_task_id: Optional[str] = Field(None, description="ID of the task currently being executed (if applicable).")
    current_task_description: Optional[str] = Field(None, description="Description or name of the current task.")
    message: Optional[str] = Field(None, description="Additional status message or progress details.")

    model_config = ConfigDict(json_schema_extra = {
        "example": {
            "agent_run_id": "run-f0ec-4a27-b3c8-adea120ddf71",
            "overall_status": "executing",
            "current_task_id": "task-002",
            "current_task_description": "Search for current weather in London",
            "message": "Executor processing tool call..."
        }
    })

class AgentRunResultResponse(BaseModel):
    """Schema for the final result of an agent run."""
    agent_run_id: str = Field(..., description="Identifier of the agent run.")
    overall_status: AgentRunStatus = Field(..., description="Final status of the agent run (e.g., completed, failed).")
    final_response: Optional[Any] = Field(None, description="The final synthesized response from the agent if successful.")
    error_message: Optional[str] = Field(None, description="Error message if the agent run failed.")

    model_config = ConfigDict(json_schema_extra = {
        "example_success": {
            "agent_run_id": "run-f0ec-4a27-b3c8-adea120ddf71",
            "overall_status": "completed",
            "final_response": "The weather in London is 15°C and partly cloudy. 2+2 equals 4.",
            "error_message": None
        },
        "example_failure": {
            "agent_run_id": "run-f0ec-4a27-b3c8-adea120ddf71",
            "overall_status": "failed",
            "final_response": None,
            "error_message": "Planner failed to generate a valid DAG after 3 attempts."
        }
    })

class ApiErrorDetail(BaseModel):
    loc: Optional[List[str | int]] = None
    msg: str
    type: str

class ApiErrorResponse(BaseModel):
    """Generic error response model for API failures (e.g., validation errors)."""
    detail: str | List[ApiErrorDetail]

    model_config = ConfigDict(json_schema_extra = {
        "example_validation_error": {
            "detail": [
                {
                    "loc": ["body", "query"],
                    "msg": "Field required",
                    "type": "missing"
                }
            ]
        },
        "example_generic_error": {
            "detail": "An unexpected internal server error occurred."
        }
    }) 