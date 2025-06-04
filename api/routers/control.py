from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel
from typing import Optional

from ..models import InitializeRequest, InitializeResponse, ControlActionRequest, ControlActionResponse, ControlAction
from ...agent.manager import AgentManager, get_agent_manager, InvalidStateError
from ...config import MochiWorkerConfig

router = APIRouter()

class AgentQueryRequest(BaseModel):
    query: str
    conversation_context: Optional[str] = None

@router.post("/initialize", response_model=InitializeResponse)
async def initialize_worker(request: InitializeRequest, agent_manager: AgentManager = Depends(get_agent_manager)):
    """Initializes the worker agent with the provided configuration."""
    try:
        worker_config_data = request.config if request.config is not None else {}
        try:
            worker_config = MochiWorkerConfig(**worker_config_data)
        except Exception as pydantic_error:
            print(f"Error creating MochiWorkerConfig from request data: {pydantic_error}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid configuration data: {pydantic_error}"
            )
        
        success, message = agent_manager.initialize(worker_config)
        return InitializeResponse(status=message)
    except Exception as e:
        print(f"Error during worker initialization: {e}")
        if isinstance(e, HTTPException) and e.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
            raise
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to initialize worker: {e}")

@router.post("/action", response_model=ControlActionResponse)
async def control_worker_action(request: ControlActionRequest, agent_manager: AgentManager = Depends(get_agent_manager)):
    """Performs a lifecycle action (pause, resume, shutdown) on the worker."""
    try:
        status_message = agent_manager.perform_action(request.action)
        return ControlActionResponse(status=status_message)
    except InvalidStateError as e:
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        print(f"Error performing control action {request.action.value}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to perform action {request.action.value}.")

@router.post("/agent/query")
async def agent_query(
    request_data: AgentQueryRequest,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Submits a query to the Mochi agent for processing."""
    try:
        result = await agent_manager.process_query(
            query=request_data.query,
            conversation_context=request_data.conversation_context
        )
        return result
    except InvalidStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent query failed: {e}") 