from fastapi import APIRouter, HTTPException, Depends, Request, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, AsyncGenerator
import uuid
import json
import asyncio

class ManagedQueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False

router = APIRouter()

async def _event_stream_generator(
    queue: asyncio.Queue,
    agent_call_task: asyncio.Task,
    initial_conv_id: str,
    request: Request
) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events from an asyncio.Queue."""
    first_event_sent = False
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if agent_call_task.done() and queue.empty():
                    break
                continue
            
            if not first_event_sent:
                if isinstance(event, dict) and "conversation_id" not in event:
                    event["conversation_id"] = initial_conv_id
                elif not isinstance(event, dict):
                    event = {"data": event, "conversation_id": initial_conv_id}
                first_event_sent = True

            if event is None:
                break
            
            yield f"data: {json.dumps(event)}\n\n"
            queue.task_done()
            
            if isinstance(event, dict) and event.get("event_type") == "agent_run_end":
                break
                
    except asyncio.CancelledError:
        if hasattr(request.app.state.mochi_agent, 'logger') and request.app.state.mochi_agent.logger:
            request.app.state.mochi_agent.logger.info("Streaming client disconnected or task cancelled.", event_type="API_STREAM_CANCELLED")
        else:
            print("Streaming client disconnected or task cancelled.")
    finally:
        if not agent_call_task.done():
            agent_call_task.cancel()
        try:
            await agent_call_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if hasattr(request.app.state.mochi_agent, 'logger') and request.app.state.mochi_agent.logger:
                request.app.state.mochi_agent.logger.error(f"Error during agent task cleanup after stream cancellation: {e}", exc_info=True, event_type="API_STREAM_CLEANUP_ERROR")
            else:
                print(f"Error during agent task cleanup after stream cancellation: {e}")

@router.post("/query", summary="Run a managed query with conversation context")
async def run_managed_conversation_query(
    request_data: ManagedQueryRequest,
    request: Request
):
    """
    Accepts a query and an optional conversation_id. 
    Manages conversation history automatically.
    Returns the agent's response, potentially as a stream of events.
    """
    mochi_agent = request.app.state.mochi_agent
    if not mochi_agent:
        raise HTTPException(status_code=500, detail="Mochi Agent not available.")

    conv_id = request_data.conversation_id if request_data.conversation_id else str(uuid.uuid4())

    if request_data.stream:
        event_queue = asyncio.Queue()

        async def stream_callback_adapter(event_data: Dict[str, Any]):
            await event_queue.put(event_data)

        agent_task = asyncio.create_task(
            mochi_agent.run_managed_query(
                query=request_data.query,
                conversation_id=conv_id,
                user_info=request_data.user_info,
                session_data=request_data.session_data,
                stream_callback=stream_callback_adapter
            )
        )
        return StreamingResponse(
            _event_stream_generator(event_queue, agent_task, conv_id, request),
            media_type="text/event-stream"
        )
    else:
        try:
            result = await mochi_agent.run_managed_query(
                query=request_data.query,
                conversation_id=conv_id,
                user_info=request_data.user_info,
                session_data=request_data.session_data
            )
            
            if isinstance(result, dict):
                result["conversation_id"] = conv_id
            else:
                return {"response": result, "conversation_id": conv_id}
            return result
        except Exception as e:
            if hasattr(mochi_agent, 'logger') and mochi_agent.logger:
                mochi_agent.logger.error(f"API error in /conversations/query (non-stream): {e}", exc_info=True, event_type="API_CONVERSATION_ERROR_NON_STREAM")
            else:
                print(f"API error in /conversations/query (non-stream): {e}")
            raise HTTPException(status_code=500, detail=f"Error processing managed query: {str(e)}")

