from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    initialize_state,
    planner_node,
    task_fetching_unit_node,
    executor_module,
    joiner_node,
    format_response,
    handle_no_plan_query,
    dag_repair_node
)
from typing import Literal, Union
from worker.core.logging import MochiLogger
from worker.config import get_settings

module_logger = MochiLogger(config=get_settings().logging)

workflow = StateGraph(AgentState)

workflow.add_node("initialize", initialize_state)
workflow.add_node("planner", planner_node)
workflow.add_node("fetch_and_execute_tasks", task_fetching_unit_node)
workflow.add_node("execute_single_task", executor_module)
workflow.add_node("join", joiner_node)
workflow.add_node("respond", format_response)
workflow.add_node("handle_no_plan_query", handle_no_plan_query)
workflow.add_node("dag_repair", dag_repair_node)

def route_after_planner(state: AgentState) -> Literal["fetch_and_execute_tasks", "handle_no_plan_query", END]:
    """Determines the next step after the planner runs.
    Checks the query_type and routes accordingly.
    """
    active_logger: MochiLogger = state.logger or module_logger
    
    planner_output = state.planner_output or {}
    query_type = planner_output.get("query_type")

    if query_type == "NO_PLAN":
        active_logger.info("Query type is NO_PLAN. Routing from planner to handle_no_plan_query.", event_type="GRAPH_ROUTING")
        return "handle_no_plan_query"
    
    elif query_type in ["COMPLEX", "SIMPLE", "COMPLEX_PHASE", "SIMPLE_PHASE"]:
        if state.planner_error:
            active_logger.error(f"Routing from planner to END due to planner_error: {state.planner_error}", event_type="GRAPH_ROUTING_ERROR")
            return END
        elif state.task_dag:
            active_logger.info(f"Query type is {query_type}. Routing from planner to fetch_and_execute_tasks.", event_type="GRAPH_ROUTING")
            return "fetch_and_execute_tasks"
        else:
            active_logger.error(f"Query type {query_type} but no DAG and no planner_error. Routing to END.", event_type="GRAPH_ROUTING_ERROR")
            return END
    else:
        active_logger.error(f"Unknown or missing query_type ('{query_type}') after planner. Routing to END.", event_type="GRAPH_ROUTING_ERROR")
        return END

def route_after_fetch_and_execute(state: AgentState) -> Literal["join", END]:
    """Determines the next step after the task fetching and execution unit runs."""
    active_logger: MochiLogger = state.logger or module_logger
    execution_error = state.execution_error

    if execution_error:
        active_logger.error(f"Routing from fetch_and_execute_tasks to END due to execution_error: {execution_error}", event_type="GRAPH_ROUTING_ERROR")
        return END 
    else:

        active_logger.info("Routing from fetch_and_execute_tasks to join.", event_type="GRAPH_ROUTING")
        return "join"

def route_after_joiner(state: AgentState) -> Literal["planner", "respond", "dag_repair", END]:
    """Determines the next step after the joiner runs."""
    active_logger: MochiLogger = state.logger or module_logger
    
    joiner_error = state.joiner_error
    needs_replanning = getattr(state, "needs_replanning", False)
    repair_instructions_available = getattr(state, "repair_instructions_available", False)

    if joiner_error:
        active_logger.error(f"Routing from joiner to END due to joiner_error: {joiner_error}", event_type="GRAPH_ROUTING_ERROR")
        return END
    elif repair_instructions_available:
        active_logger.info("Routing from joiner to dag_repair for targeted DAG modification.", event_type="GRAPH_ROUTING_REPAIR")
        return "dag_repair"
    elif needs_replanning:
        active_logger.info("Routing from joiner back to planner for replanning.", event_type="GRAPH_ROUTING_REPLAN")
        return "planner"
    else:
        active_logger.info("Routing from joiner to respond for final formatting.", event_type="GRAPH_ROUTING")
        return "respond"

def route_after_dag_repair(state: AgentState) -> Literal["fetch_and_execute_tasks", "planner", END]:
    """Determines the next step after the DAG repair node runs."""
    active_logger: MochiLogger = state.logger or module_logger
    dag_repair_error = state.dag_repair_error
    current_dag = state.task_dag

    if dag_repair_error:
        active_logger.error(f"DAG repair failed: {dag_repair_error}. Routing to planner for full replan.", event_type="GRAPH_ROUTING_REPAIR_FAILURE")
        # Fallback to full replan if repair itself fails
        # Ensure needs_replanning is set if we go back to planner from here due to repair failure
        state.needs_replanning = True
        state.error_message = (state.error_message or "") + f"; DAG Repair Failed: {dag_repair_error}"
        return "planner" 
    elif current_dag and current_dag.tasks: # Check if DAG exists and has tasks after repair
        active_logger.info("DAG repair successful or no repair needed. Routing to fetch_and_execute_tasks.", event_type="GRAPH_ROUTING_REPAIR_SUCCESS")
        return "fetch_and_execute_tasks"
    else:
        # If DAG is empty or None after repair attempt (should not happen if repair logic is correct and returns original on failure)
        # Or if repair node was skipped but we still ended up here (unlikely)
        active_logger.error("DAG is empty or None after repair node, or repair node skipped. Routing to planner.", event_type="GRAPH_ROUTING_ERROR")
        state.needs_replanning = True
        state.error_message = (state.error_message or "") + "; Invalid state after DAG repair: No executable DAG."
        return "planner"

# --- Build Graph Edges --- 

# Entry point
workflow.set_entry_point("initialize")

# Standard transitions
workflow.add_edge("initialize", "planner")

# Conditional transition from planner
workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "fetch_and_execute_tasks": "fetch_and_execute_tasks",
        "handle_no_plan_query": "handle_no_plan_query",
        # "handle_error": "handle_error", # Map to error node if defined
        END: END # Route to end if error occurs for now. END is a special object from langgraph.graph
    }
)

workflow.add_conditional_edges(
    "fetch_and_execute_tasks",
    route_after_fetch_and_execute,
    {
        "join": "join",
        END: END
    }
)

workflow.add_conditional_edges(
    "join",
    route_after_joiner,
    {
        "planner": "planner",
        "respond": "respond",
        "dag_repair": "dag_repair",
        END: END
    }
)

workflow.add_conditional_edges(
    "dag_repair",
    route_after_dag_repair,
    {
        "fetch_and_execute_tasks": "fetch_and_execute_tasks",
        "planner": "planner",
        END: END # Should ideally not route to END directly from here unless unrecoverable
    }
)

workflow.add_edge("respond", END)

workflow.add_edge("handle_no_plan_query", "join")

# Compile the graph
app = workflow.compile()

module_logger.info("Mochi worker agent graph compiled successfully.")
