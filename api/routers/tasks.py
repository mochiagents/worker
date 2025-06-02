from fastapi import APIRouter, HTTPException, status, Path, Depends
from typing import List, Optional

from ..models import TaskRequest, TaskResponse, TaskResult, TaskStatus
from ...agent.manager import AgentManager, get_agent_manager, TaskNotFoundError, InvalidStateError

router = APIRouter()

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(task: TaskRequest, agent_manager: AgentManager = Depends(get_agent_manager)):
    """Accepts a new task for the worker agent."""
    try:
        agent_manager.submit_new_task(task.model_dump())
        return TaskResponse(task_id=task.task_id, status=TaskStatus.ACCEPTED, message="Task accepted and queued.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except InvalidStateError as e:
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        print(f"Error submitting task {task.task_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to submit task.")

@router.get("/{task_id}/status", response_model=TaskResponse)
async def get_task_status(
    task_id: str = Path(..., title="The ID of the task to get status for"),
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Retrieves the current status of a specific task."""
    try:
        current_status = agent_manager.get_task_current_status(task_id)
        return TaskResponse(task_id=task_id, status=current_status)
    except TaskNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except InvalidStateError as e:
         raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        print(f"Error getting status for task {task_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get task status.")

@router.get("/{task_id}/result", response_model=TaskResult)
async def get_task_result(
    task_id: str = Path(..., title="The ID of the task to get the result for"),
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Retrieves the final result or error for a completed or failed task."""
    status_val: Optional[TaskStatus] = None
    result_val: Optional[dict] = None
    error_val: Optional[str] = None

    try:
        status_val, result_val, error_val = agent_manager.get_task_final_result(task_id)
        return TaskResult(task_id=task_id, status=status_val, result=result_val, error=error_val)
    except TaskNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except InvalidStateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        print(f"Error getting result for task {task_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get task result due to an unexpected error.") 