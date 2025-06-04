# This file is used by the LangGraph StateGraph to define the schema of the agent state.
# It should refer to the Pydantic model defined in worker.core.models.

from worker.core.models import AgentState

__all__ = ['AgentState']
