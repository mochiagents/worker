from typing import Dict, Any, List, Optional, Union
from .state import AgentState
from worker.agent.planner import Planner
from worker.agent.execution import TaskFetchingUnit
from worker.core.models import TaskDAG, TaskNode, ToolExecutionResult
from worker.core.logging import MochiLogger
from worker.agent.mcp_client import McpError
from worker.agent.dag_editor import DAGEditor
from worker.config import get_settings
import json

# Create module-level MochiLogger with proper config
module_logger = MochiLogger(config=get_settings().logging)

NODE_DESCRIPTIONS = {
    "planner_node": "Analyzes the user query and generates a plan (DAG) of tasks.",
    "task_fetching_node": "Identifies the next available task(s) from the plan to be executed.",
    "task_execution_node": "Executes a single task using the appropriate tool via an MCP client.",
    "joiner_node": "Synthesizes results from executed tasks, formulates a response, and determines if replanning is needed.",
    # Add other node names and descriptions as needed
}

def initialize_state(state: AgentState) -> Dict[str, Any]:
    """
    Initializes or resets the agent state for a new execution run or replanning.
    Preserves essential configuration, input fields, and pre-initialized service instances.
    """
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("Initialize_state: MochiLogger not found in input state. Using default module_logger for this node.")

    active_logger.info("--- Initializing State (Graph Entry) ---", event_type="GRAPH_STATE_INIT")

    preserved_fields = {
        "original_query": state.original_query,
        "conversation_context": state.conversation_context,
        "logger": active_logger,
        "config": state.config,
        "run_id": state.run_id,
        "planner_instance": state.planner_instance,
        "task_fetching_unit_instance": state.task_fetching_unit_instance,
        "joiner_instance": state.joiner_instance,
        "mcp_clients": state.mcp_clients,
        "dag_editor_instance": state.dag_editor_instance
    }

    reset_fields = {
        "task_dag": None,
        "generated_dag": None, 
        "planner_error": None,
        "task_statuses": {},
        "task_results": {},
        "error_message": None,
        "error": None,
        "final_response": None,
        "needs_replanning": False,
        "repair_instructions_available": False,
        "dag_repair_instructions": None,
        "dag_repair_error": None,
        "current_task_id_to_execute": None,
        "has_ready_tasks": False,
        "all_completed": False,
    }

    update_dict = {**reset_fields, **preserved_fields}

    for key in ["planner_instance", "task_fetching_unit_instance", "joiner_instance", "logger", "config", "dag_editor_instance"]:
        if update_dict.get(key) is None:
            if key == "dag_editor_instance":
                active_logger.info(f"Initialize_state: Optional component '{key}' is missing from initial state. This may be expected if DAG repair is not yet fully integrated.", event_type="GRAPH_STATE_INFO")
            active_logger.warning(f"Initialize_state: Critical component '{key}' is missing from initial state.", event_type="GRAPH_STATE_WARN")

    active_logger.info("State initialization complete. Preserved/Reset fields combined.", event_type="GRAPH_STATE_INIT_COMPLETE")
    return update_dict

async def planner_node(state: AgentState) -> Dict[str, Any]:
    """Node function representing the Planner module (LLM Call)."""
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("Planner_node: MochiLogger not found in input state. Using default module_logger.")

    active_logger.info("--- Planner Node: Executing ---", event_type="NODE_EXEC_START", metadata={"node_name": "planner_node"})

    planner_instance: Optional[Planner] = state.planner_instance
    original_query: Optional[str] = state.original_query
    conversation_context: Optional[str] = state.conversation_context
    replan_context: Optional[Dict[str, Any]] = state.replan_context
    dag_repair_error: Optional[str] = state.dag_repair_error
    dag_repair_instructions: Optional[Dict[str, Any]] = state.dag_repair_instructions

    current_phase_id: Optional[str] = state.current_phase_id
    current_phase_description: Optional[str] = state.current_phase_description

    # Determine the primary query for the planner
    query_for_planner: Optional[str] = original_query
    # If in a specific phase, use the phase description as the primary query for DAG generation.
    # The original query can be passed as part of the conversation_context or a dedicated field later if needed.
    if current_phase_id and current_phase_description:
        query_for_planner = current_phase_description
        active_logger.info(f"Planner node: Operating in phase '{current_phase_id}'. Using phase description as query: '{current_phase_description[:100]}...'", event_type="PLANNER_PHASE_CONTEXT", metadata={"node_name": "planner_node"})
    else:
        active_logger.info(f"Planner node: Not in a specific phase or phase description missing. Using original query: '{original_query[:100] if original_query else 'N/A'}...'", event_type="PLANNER_ORIGINAL_QUERY_CONTEXT", metadata={"node_name": "planner_node"})


    initial_classified_query_type = state.planner_output.get("query_type") if state.planner_output else None
    
    # Check if a DAG already exists from main.py for the current phase and no replan/repair is needed
    # state.task_dag is the DAG potentially pre-loaded by main.py for the current phase.
    # state.generated_dag is the DAG generated by a previous run of this planner_node within the same graph loop.
    # replan_context implies a replan is needed.
    # dag_repair_error or dag_repair_instructions implies a repair is needed.
    # state.current_phase_id ensures this logic applies only during phased execution.
    if state.task_dag and state.current_phase_id and not replan_context and not dag_repair_error and not dag_repair_instructions:
        active_logger.info(f"Planner node: Using pre-existing DAG for phase '{current_phase_id}'. Skipping DAG regeneration.", event_type="PLANNER_USING_EXISTING_DAG", metadata={"node_name": "planner_node"})
        return {
            "task_dag": state.task_dag, # Use the DAG passed in state, likely from main.py
            "generated_dag": state.task_dag, # Consistent with task_dag
            "planner_error": None,
            "logger": active_logger,
            "planner_output": state.planner_output or {"query_type": initial_classified_query_type, "status": "Used pre-existing DAG for phase.", "error": None}
        }

    final_return_value = {
        "task_dag": None,
        "generated_dag": None,
        "planner_error": "PLANNER_NODE_UNINITIALIZED_ERROR",
        "logger": active_logger,
        "planner_output": {"query_type": initial_classified_query_type, "status": "Error: Uninitialized", "error": "PLANNER_NODE_UNINITIALIZED_ERROR"}
    }

    if not planner_instance:
        error_msg = "Planner instance not found in agent state."
        active_logger.error(error_msg, event_type="NODE_CONFIG_ERROR", metadata={"node_name": "planner_node"})
        final_return_value["planner_error"] = error_msg
        return final_return_value

    if not query_for_planner: # Check query_for_planner instead of original_query directly here
        error_msg = "Input query (original or phase description) not available for planner."
        active_logger.error(error_msg, event_type="NODE_INPUT_ERROR", metadata={"node_name": "planner_node"})
        final_return_value["planner_error"] = error_msg
        return final_return_value

    try:
        # Prepare an enriched conversation context if we are using phase description as query
        effective_conversation_context = conversation_context
        if current_phase_id and current_phase_description and original_query and query_for_planner == current_phase_description:
            context_prefix = f"Original user query (overall goal): {original_query}\n\nCurrently working on phase '{current_phase_id}': {current_phase_description}\n\nPrior conversation:\n"
            effective_conversation_context = context_prefix + (conversation_context if conversation_context else "No prior conversation.")

        active_logger.info(f"Planner node: Calling planner_instance.generate_dag with query_for_planner: '{query_for_planner[:100]}...'", event_type="PLANNER_INVOKE", metadata={"node_name": "planner_node"})

        generated_dag_model, dag_generation_status_msg = await planner_instance.generate_dag(
            query=query_for_planner, # Use the determined query for planner
            conversation_context=effective_conversation_context, # Pass potentially enriched context
            replan_context=replan_context, 
            failed_repair_error=dag_repair_error, 
            failed_repair_instructions=dag_repair_instructions
        )
        
        # The query_type for routing is initial_classified_query_type.
        # dag_generation_status_msg is just a status from the generate_dag call.
        log_query_type_for_event = initial_classified_query_type if initial_classified_query_type else "Unknown_initial"

        active_logger.info(f"Planner node: planner_instance.generate_dag call COMPLETED. Status: {dag_generation_status_msg}, Initial Classified Type: {log_query_type_for_event}, DAG generated: {generated_dag_model is not None}", event_type="PLANNER_INVOKE_SUCCESS", metadata={"node_name": "planner_node", "status": dag_generation_status_msg, "initial_query_type": log_query_type_for_event})

        final_return_value["task_dag"] = generated_dag_model
        final_return_value["planner_output"] = {"query_type": initial_classified_query_type, "status": dag_generation_status_msg, "error": None} 
        final_return_value["generated_dag"] = generated_dag_model
        final_return_value["planner_error"] = None

    except McpError as mcpe:
        error_msg = f"Planner node: McpError during DAG generation: {mcpe}"
        active_logger.error(error_msg, exc_info=True, event_type="PLANNER_MCP_ERROR", metadata={"node_name": "planner_node"})
        final_return_value["planner_error"] = str(mcpe)
        final_return_value["planner_output"] = {"query_type": initial_classified_query_type, "status": "Error: McpError", "error": str(mcpe)}
    except ValueError as ve:
        error_msg = f"Planner node: ValueError during DAG generation: {ve}"
        active_logger.error(error_msg, exc_info=True, event_type="PLANNER_VALIDATION_ERROR", metadata={"node_name": "planner_node"})
        final_return_value["planner_error"] = str(ve)
        final_return_value["planner_output"] = {"query_type": initial_classified_query_type, "status": "Error: ValueError", "error": str(ve)}
    except Exception as e_outer:
        error_msg = f"Planner node: Unexpected exception during DAG generation: {e_outer}"
        active_logger.error(error_msg, exc_info=True, event_type="PLANNER_UNEXPECTED_ERROR", metadata={"node_name": "planner_node"})
        final_return_value["planner_error"] = str(e_outer)
        final_return_value["planner_output"] = {"query_type": initial_classified_query_type, "status": "Error: Unexpected Exception", "error": str(e_outer)}
    
    dag_task_count = 0
    if final_return_value.get('task_dag') and isinstance(final_return_value['task_dag'], TaskDAG):
        dag_task_count = len(final_return_value['task_dag'].tasks)
    
    returned_query_type = final_return_value.get("planner_output", {}).get("query_type")
    
    active_logger.info(f"Planner node: Returning: query_type='{returned_query_type}', dag_tasks_count={dag_task_count}, error='{final_return_value['planner_error']}'", event_type="NODE_EXEC_COMPLETE", metadata={"node_name": "planner_node", "query_type": returned_query_type, "dag_task_count": dag_task_count, "error": final_return_value['planner_error']}) # type: ignore
    return final_return_value

async def task_fetching_unit_node(state: AgentState) -> Dict[str, Any]:
    """Node responsible for managing DAG execution flow: identifying ready tasks, 
       delegating their execution, and updating overall DAG status.
    """
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("TaskFetchingUnit_node: MochiLogger not found in input state. Using default module_logger.")

    active_logger.info("--- Task Fetching Unit Node: Executing ---", event_type="NODE_EXEC_START", metadata={"node_name": "task_fetching_unit_node"})

    tfu_instance: Optional[TaskFetchingUnit] = state.task_fetching_unit_instance
    current_dag: Optional[TaskDAG] = state.task_dag
    return_update: Dict[str, Any] = {
        "task_statuses": state.task_statuses,
        "task_results": state.task_results,
        "execution_error": None,
        "all_completed": False, # Default to false, TFU will determine this
        "has_ready_tasks": False, # Default to false
        "current_task_id_to_execute": state.current_task_id_to_execute, # Preserve if already set, TFU might clear/update
        "logger": active_logger
    }

    if not tfu_instance:
        error_msg = "TaskFetchingUnit instance not found in agent state."
        active_logger.error(error_msg, event_type="NODE_CONFIG_ERROR", metadata={"node_name": "task_fetching_unit_node"})
        return_update["execution_error"] = error_msg
        return return_update

    if not current_dag or not current_dag.tasks:
        active_logger.info("TaskFetchingUnit node: No DAG or no tasks in DAG. Nothing to process.", event_type="TFU_NO_DAG", metadata={"node_name": "task_fetching_unit_node"})
        return_update["all_completed"] = True # No tasks means all (zero) are technically completed
        return return_update
    
    try:
        active_logger.info("TaskFetchingUnit node: Calling tfu_instance.process_dag.", event_type="TFU_PROCESS_INVOKE", metadata={"node_name": "task_fetching_unit_node"})
        
        processed_state: AgentState = await tfu_instance.process_dag(current_dag, state)

        return_update["task_statuses"] = processed_state.task_statuses
        return_update["task_results"] = processed_state.task_results
        return_update["execution_error"] = processed_state.execution_error
        return_update["all_completed"] = processed_state.all_completed
        return_update["has_ready_tasks"] = processed_state.has_ready_tasks
        # current_task_id_to_execute is not explicitly set by the new TFU process_dag loop,
        # so we don't update it from processed_state unless TFU starts managing it.
        # If it was in the original state, it's preserved in return_update's initialization.

        active_logger.info("TaskFetchingUnit node: tfu_instance.process_dag call COMPLETED.", event_type="TFU_PROCESS_SUCCESS", metadata={"node_name": "task_fetching_unit_node"})

    except McpError as mcpe:
        error_msg = f"TaskFetchingUnit node: McpError during DAG processing: {mcpe}"
        active_logger.error(error_msg, exc_info=True, event_type="TFU_MCP_ERROR", metadata={"node_name": "task_fetching_unit_node"})
        return_update["execution_error"] = str(mcpe)
        return_update["all_completed"] = False
        return_update["has_ready_tasks"] = False
    except Exception as e_outer:
        error_msg = f"TaskFetchingUnit node: Unexpected exception during DAG processing: {e_outer}"
        active_logger.error(error_msg, exc_info=True, event_type="TFU_UNEXPECTED_ERROR", metadata={"node_name": "task_fetching_unit_node"})
        return_update["execution_error"] = str(e_outer)
        return_update["all_completed"] = False
        return_update["has_ready_tasks"] = False

    log_meta = {
        "node_name": "task_fetching_unit_node",
        "all_completed": return_update.get("all_completed"),
        "has_ready_tasks": return_update.get("has_ready_tasks"),
        "error": return_update.get("execution_error")
    }
    active_logger.info(f"TaskFetchingUnit node: Returning. All completed: {log_meta['all_completed']}, Has ready: {log_meta['has_ready_tasks']}, Error: '{log_meta['error']}'", 
        event_type="NODE_EXEC_COMPLETE", metadata=log_meta)
    return return_update


async def executor_module(state: AgentState) -> Dict[str, Any]:
    """Node function representing the Executor module.
    This node is responsible for executing a single, specific task if identified
    by a preceding node (e.g., if task_fetching_unit_node was designed to yield one task at a time).
    With the current TaskFetchingUnit.process_dag handling batch execution, this node's role
    might be for specific scenarios like single task retries if the graph routes here.
    """
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("Executor_module: MochiLogger not found in input state. Using default module_logger.")
    
    active_logger.info("--- Executor Module: Executing ---", event_type="NODE_EXEC_START", metadata={"node_name": "executor_module"})

    tfu_instance: Optional[TaskFetchingUnit] = state.task_fetching_unit_instance
    task_id_to_execute: Optional[str] = state.current_task_id_to_execute
    current_dag: Optional[TaskDAG] = state.task_dag
    task_results: Dict[str, ToolExecutionResult] = state.task_results
    task_statuses: Dict[str, str] = state.task_statuses

    return_update: Dict[str, Any] = {
        "task_results": task_results,
        "task_statuses": task_statuses,
        "execution_error": state.execution_error,
        "logger": active_logger
    }

    if not tfu_instance:
        error_msg = "TaskFetchingUnit instance (for executor service) not found in agent state."
        active_logger.error(error_msg, event_type="NODE_CONFIG_ERROR", metadata={"node_name": "executor_module"})
        return_update["execution_error"] = return_update["execution_error"] + "; " + error_msg if return_update["execution_error"] else error_msg
        return return_update

    if not task_id_to_execute:
        active_logger.info("Executor module: No current_task_id_to_execute specified in state. Nothing to execute here.", event_type="EXECUTOR_SKIP", metadata={"node_name": "executor_module"})
        return return_update

    if not current_dag or not current_dag.tasks:
        error_msg = f"Executor module: DAG not found or empty, cannot execute task '{task_id_to_execute}'."
        active_logger.error(error_msg, event_type="NODE_INPUT_ERROR", metadata={"node_name": "executor_module"})
        return_update["execution_error"] = return_update["execution_error"] + "; " + error_msg if return_update["execution_error"] else error_msg
        return return_update

    task_to_run: Optional[TaskNode] = None
    for task_node in current_dag.tasks:
        if task_node.id == task_id_to_execute:
            task_to_run = task_node
            break
    
    if not task_to_run:
        error_msg = f"Executor module: Task with ID '{task_id_to_execute}' not found in the current DAG."
        active_logger.error(error_msg, event_type="NODE_INPUT_ERROR", metadata={"node_name": "executor_module"})
        return_update["execution_error"] = return_update["execution_error"] + "; " + error_msg if return_update["execution_error"] else error_msg
        return return_update

    executor_service = tfu_instance.executor_service 

    try:
        active_logger.info(f"Executor module: Executing task '{task_id_to_execute}' (Tool: {task_to_run.server_id}/{task_to_run.tool_name}) using executor service.", event_type="EXECUTOR_INVOKE", metadata={"node_name": "executor_module"}) # type: ignore
        
        task_statuses[task_id_to_execute] = "in_progress"
        return_update["task_statuses"] = task_statuses

        result_model: ToolExecutionResult = await executor_service.execute_task(
            task=task_to_run, 
            task_results=task_results
        )

        active_logger.info(f"Executor module: Task '{task_id_to_execute}' execution completed. Status: {result_model.status}", event_type="EXECUTOR_RESULT", metadata={"node_name": "executor_module"}) # type: ignore

        task_results[task_id_to_execute] = result_model
        if result_model.status == "success":
            task_statuses[task_id_to_execute] = "completed"
        else:
            task_statuses[task_id_to_execute] = "failed"
            active_logger.error(f"Executor module: Task '{task_id_to_execute}' failed. Error: {result_model.error}", event_type="EXECUTOR_TASK_FAILURE", metadata={"node_name": "executor_module"})
            
        return_update["task_results"] = task_results
        return_update["task_statuses"] = task_statuses
        return_update["current_task_id_to_execute"] = None 

    except Exception as e:
        error_msg = f"Executor module: Unexpected exception during task '{task_id_to_execute}' execution: {e}"
        active_logger.error(error_msg, exc_info=True, event_type="EXECUTOR_UNEXPECTED_ERROR", metadata={"node_name": "executor_module"})
        task_statuses[task_id_to_execute] = "failed"
        if task_id_to_execute not in task_results:
            task_results[task_id_to_execute] = ToolExecutionResult(task_id=task_id_to_execute, status="failure", error=str(e))
        elif task_results[task_id_to_execute].status != "failure":
            task_results[task_id_to_execute].status = "failure"
            task_results[task_id_to_execute].error = str(e)
            
        return_update["task_results"] = task_results
        return_update["task_statuses"] = task_statuses
        return_update["execution_error"] = return_update["execution_error"] + "; " + error_msg if return_update["execution_error"] else error_msg
        return_update["current_task_id_to_execute"] = None

    active_logger.info(f"Executor module: Returning - task_id: {task_id_to_execute}, status: {task_statuses.get(task_id_to_execute)}, error: '{return_update['execution_error']}'", event_type="NODE_EXEC_COMPLETE", metadata={"node_name": "executor_module"}) # type: ignore
    return return_update


async def joiner_node(state: AgentState) -> AgentState:
    """Runs the Joiner module to synthesize results and decide on replanning."""
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("Joiner_node: MochiLogger not found in input state. Using default module_logger.")
        
    joiner_instance = state.joiner_instance 
    
    if not joiner_instance:
        # Attempt to get from services if not directly on state (legacy or alternative setup)
        # This part can be removed if joiner_instance is always expected directly on AgentState
        # For now, keeping for some backward compatibility or flexibility.
        if hasattr(state, 'services') and isinstance(state.services, dict):
             joiner_instance = state.services.get("joiner")

    if not joiner_instance:
        log_message = "Joiner service instance not found in state (checked 'joiner_instance' and 'services.joiner')."
        active_logger.error(log_message, event_type="NODE_ERROR", metadata={"node_name": "joiner_node"})

        # Create a new state for returning, ensuring service instances are by reference
        new_state_data_on_error = {}
        if state: # Check if state is not None
            new_state_data_on_error = state.model_dump(exclude_none=False) # Dump all fields
            # Restore non-serializable/problematic fields by reference
            new_state_data_on_error['mcp_clients'] = state.mcp_clients
            new_state_data_on_error['planner_instance'] = state.planner_instance
            new_state_data_on_error['task_fetching_unit_instance'] = state.task_fetching_unit_instance
            new_state_data_on_error['joiner_instance'] = state.joiner_instance # Will be None here
            new_state_data_on_error['dag_editor_instance'] = state.dag_editor_instance
            new_state_data_on_error['config'] = state.config
        
        new_state_data_on_error["logger"] = active_logger # Set active logger
        new_state_data_on_error["joiner_error"] = "Joiner service instance not found"
        new_state_data_on_error["needs_replanning"] = False
        # Ensure original_query is present if state was None and AgentState requires it
        if 'original_query' not in new_state_data_on_error and state:
             new_state_data_on_error['original_query'] = state.original_query
        elif 'original_query' not in new_state_data_on_error:
             new_state_data_on_error['original_query'] = "Unknown query due to missing state"


        return AgentState(**new_state_data_on_error)


    query = state.original_query
    dag = state.task_dag
    task_results = state.task_results
    task_statuses = state.task_statuses
    planner_output = state.planner_output
    conversation_context = state.conversation_context
    mcp_clients = state.mcp_clients

    if query is None:
        log_message = "Joiner Node: Input query is missing from state."
        active_logger.error(log_message, event_type="NODE_ERROR", metadata={"node_name": "joiner_node"})
        return {**state, "joiner_error": "Input query missing", "needs_replanning": False}

    try:
        active_logger.info("Joiner node: Invoking joiner_instance.process_results", event_type="JOINER_INVOKE_START", metadata={"node_name": "joiner_node"})
        
        joiner_result = await joiner_instance.process_results(
            query=query,
            dag=dag,
            task_results=task_results,
            task_status=task_statuses,
            planner_output=planner_output,
            conversation_context=conversation_context,
            mcp_clients=mcp_clients,
            stream_callback=state.stream_callback
        )
        
        log_message = f"Joiner node: joiner_instance.process_results call COMPLETED. Needs Replan: {joiner_result.get('needs_replanning')}"
        active_logger.info(log_message, event_type="JOINER_INVOKE_SUCCESS", metadata={"node_name": "joiner_node", "needs_replanning": joiner_result.get('needs_replanning'), "has_error": joiner_result.get("error") is not None}) # type: ignore
            
        # MODIFIED: Safe state copy and update
        new_state_data = state.model_dump(exclude_none=False)

        # Update with joiner results
        new_state_data['needs_replanning'] = joiner_result.get("needs_replanning", False)
        new_state_data['final_response'] = joiner_result.get("response")
        new_state_data['joiner_error'] = joiner_result.get("error")
        
        if joiner_result.get("repair_instructions_available", False):
            new_state_data['repair_instructions_available'] = True
            new_state_data['dag_repair_instructions'] = joiner_result.get("dag_repair_instructions")
        else:
            new_state_data['repair_instructions_available'] = False
            new_state_data['dag_repair_instructions'] = None
        
        # Restore/set non-serializable fields by reference
        new_state_data['logger'] = active_logger
        new_state_data['mcp_clients'] = state.mcp_clients
        new_state_data['planner_instance'] = state.planner_instance
        new_state_data['task_fetching_unit_instance'] = state.task_fetching_unit_instance
        new_state_data['joiner_instance'] = state.joiner_instance
        new_state_data['dag_editor_instance'] = state.dag_editor_instance
        new_state_data['config'] = state.config
        
        updated_state = AgentState(**new_state_data)
        return updated_state
    except Exception as e:
        log_message = f"Joiner node: Unhandled exception during joiner processing: {e}"
        active_logger.error(log_message, event_type="NODE_ERROR", exc_info=True, metadata={"node_name": "joiner_node"})

        # MODIFIED: Safe state copy for error path
        new_state_data_on_error = {}
        if state: # Check if state is not None
            new_state_data_on_error = state.model_dump(exclude_none=False)
             # Restore non-serializable/problematic fields by reference
            new_state_data_on_error['mcp_clients'] = state.mcp_clients
            new_state_data_on_error['planner_instance'] = state.planner_instance
            new_state_data_on_error['task_fetching_unit_instance'] = state.task_fetching_unit_instance
            new_state_data_on_error['joiner_instance'] = state.joiner_instance
            new_state_data_on_error['dag_editor_instance'] = state.dag_editor_instance
            new_state_data_on_error['config'] = state.config
            if 'original_query' not in new_state_data_on_error: # Ensure required field
                 new_state_data_on_error['original_query'] = state.original_query if hasattr(state, 'original_query') else "Unknown"
        else: # If state is None, provide minimal required fields
            new_state_data_on_error['original_query'] = "Unknown query due to missing state in error path"


        new_state_data_on_error['logger'] = active_logger
        new_state_data_on_error['joiner_error'] = f"Unhandled exception: {e}"
        new_state_data_on_error['needs_replanning'] = False
        new_state_data_on_error['final_response'] = "An internal error occurred while finalizing the response."
        
        return_state_on_error = AgentState(**new_state_data_on_error)
        return return_state_on_error

def format_response(state: AgentState) -> Dict[str, Any]:
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("FormatResponse: MochiLogger not found in input state. Using default module_logger.")
    
    node_metadata = {"node_name": "format_response"}
    active_logger.info("--- Formatting Final Response ---", event_type="NODE_EXEC_START", metadata=node_metadata)

    final_response_from_state = state.final_response
    error_message_detail = state.error_message

    if error_message_detail:
        active_logger.error(f"FormatResponse: An error occurred. Detail: {error_message_detail}")
        calculated_final_response = f"An error occurred. Details: {error_message_detail}".strip()
    elif final_response_from_state is not None:
        calculated_final_response = final_response_from_state
    else:
        calculated_final_response = "No response generated and no error reported."

    active_logger.info(f"FormatResponse: Final response to be returned: {calculated_final_response[:200]}...")
    
    # Apply final updates to the state object directly
    state.final_response = calculated_final_response
    state.overall_status = "completed_successfully"
    state.error_message = None # Clear any previous error
    state.logger = active_logger # Ensure the active logger is on the state being returned

    active_logger.info(f"FormatResponse: Node returning. Final Response: '{state.final_response}'. Overall Status: '{state.overall_status}'.", event_type="NODE_EXEC_COMPLETE", metadata={"node_name": "format_response"})
    
    return state.model_dump(exclude_none=False) # Return the modified state as a dict

def get_node_description(node_name: str) -> str:
    """Returns a human-readable description for a given graph node name."""
    return NODE_DESCRIPTIONS.get(node_name, f"Description for node '{node_name}' not defined.")

async def handle_no_plan_query(state: AgentState) -> AgentState:
    """Handles queries initially classified as NO_PLAN by the planner,
    by setting them up for a direct conversational response via the Joiner.
    """
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("handle_no_plan_query: MochiLogger not found in input state. Using module_logger.")

    user_query = state.original_query

    active_logger.info(f"Node: handle_no_plan_query for query: '{user_query}'. Setting type to NO_PLAN_CONVERSE for Joiner.", event_type="NODE_EXECUTION")

    # MODIFIED: Safe state copy and update
    new_state_data = state.model_dump(exclude_none=False)

    # Restore/set non-serializable fields by reference
    new_state_data['logger'] = active_logger
    new_state_data['mcp_clients'] = state.mcp_clients
    new_state_data['planner_instance'] = state.planner_instance
    new_state_data['task_fetching_unit_instance'] = state.task_fetching_unit_instance
    new_state_data['joiner_instance'] = state.joiner_instance
    new_state_data['dag_editor_instance'] = state.dag_editor_instance
    new_state_data['config'] = state.config
    
    # Ensure planner_output is a dict before item assignment or spreading
    if new_state_data.get('planner_output') is None or not isinstance(new_state_data.get('planner_output'), dict):
        new_state_data['planner_output'] = {}
    new_state_data['planner_output'] = {**new_state_data['planner_output'], "query_type": "NO_PLAN_CONVERSE"}

    
    new_state_data['task_dag'] = None
    new_state_data['generated_dag'] = None
    new_state_data['task_statuses'] = {}
    new_state_data['task_results'] = {}
    new_state_data['planner_error'] = None
    new_state_data['execution_error'] = None 
    # new_state_data['tfu_error'] = None # tfu_error is not a field on AgentState
    new_state_data['needs_replanning'] = False
    new_state_data['final_response'] = None
    # new_state_data['joiner_output'] = None # joiner_output is not a field on AgentState
    new_state_data['current_task_id_to_execute'] = None
    new_state_data['has_ready_tasks'] = False
    new_state_data['all_completed'] = True # For NO_PLAN, graph effectively ends execution part.

    active_logger.info(f"handle_no_plan_query: State prepared for Joiner (NO_PLAN_CONVERSE).", event_type="NODE_EXECUTION_RESULT")

    updated_state = AgentState(**new_state_data)
    return updated_state 

async def dag_repair_node(state: AgentState) -> Dict[str, Any]:
    """Applies repair instructions to the current DAG if available."""
    active_logger: MochiLogger = state.logger or module_logger
    if state.logger is None:
        module_logger.warning("DAGRepairNode: MochiLogger not found in input state. Using module_logger.")

    active_logger.info("--- DAG Repair Node: Executing ---", event_type="NODE_EXEC_START", metadata={"node_name": "dag_repair_node"})

    dag_editor_instance: Optional[DAGEditor] = state.dag_editor_instance
    current_dag: Optional[TaskDAG] = state.task_dag # task_dag instead of dag
    repair_instructions: Optional[Dict[str, Any]] = state.dag_repair_instructions

    # Return type is Dict, so this part is okay. Fields being set are 'task_dag', etc.
    return_update: Dict[str, Any] = { # Explicitly define return type
        "logger": active_logger,
        "task_dag": current_dag, 
        "dag_repair_error": None,
        "planner_error": state.planner_error,
        "repair_instructions_available": state.repair_instructions_available, # Carry over
        "dag_repair_instructions": state.dag_repair_instructions # Carry over
    }

    if not state.repair_instructions_available or not repair_instructions:
        active_logger.info("DAGRepairNode: No repair instructions available or provided. Skipping repair.", event_type="DAG_REPAIR_SKIP")
        return_update["repair_instructions_available"] = False
        return_update["dag_repair_instructions"] = None
        return return_update

    if not dag_editor_instance:
        error_msg = "DAGEditor instance not found in agent state. Cannot apply repairs."
        active_logger.error(error_msg, event_type="NODE_CONFIG_ERROR", metadata={"node_name": "dag_repair_node"})
        return_update["dag_repair_error"] = error_msg
        return_update["repair_instructions_available"] = False # Clear flags as repair cannot proceed
        return_update["dag_repair_instructions"] = None
        return return_update

    if not current_dag:
        error_msg = "DAGRepairNode: Current DAG not found in state. Cannot apply repairs."
        active_logger.error(error_msg, event_type="NODE_INPUT_ERROR", metadata={"node_name": "dag_repair_node"})
        return_update["dag_repair_error"] = error_msg
        return_update["repair_instructions_available"] = False
        return_update["dag_repair_instructions"] = None
        return return_update

    try:
        active_logger.info(f"DAGRepairNode: Applying repair instructions: {json.dumps(repair_instructions)[:200]}...", event_type="DAG_REPAIR_APPLY_START")
        repaired_dag = dag_editor_instance.apply_repair(
            current_dag=current_dag,
            repair_instructions=repair_instructions
        )
        return_update["task_dag"] = repaired_dag # Changed from "dag" to "task_dag"
        active_logger.info("DAGRepairNode: Successfully applied DAG repair.", event_type="DAG_REPAIR_APPLY_SUCCESS")
        # If repair is successful, it might make sense to clear any previous planner_error
        # that led to the failed execution, as the plan has now been altered.
        # However, this depends on the desired error propagation logic.
        # For now, let's clear planner_error if a repair is made, assuming it addresses the planner's output problem.
        return_update["planner_error"] = None 

    except ValueError as ve:
        error_msg = f"DAGRepairNode: ValueError during DAG repair: {ve}"
        active_logger.error(error_msg, exc_info=True, event_type="DAG_REPAIR_APPLY_VAL_ERROR")
        return_update["dag_repair_error"] = str(ve)
        # Keep the original DAG if repair fails
    except Exception as e:
        error_msg = f"DAGRepairNode: Unexpected exception during DAG repair: {e}"
        active_logger.error(error_msg, exc_info=True, event_type="DAG_REPAIR_APPLY_UNEXPECTED_ERROR")
        return_update["dag_repair_error"] = str(e)
        # Keep the original DAG if repair fails
    finally:
        # Always clear repair instructions after attempting to apply them
        return_update["repair_instructions_available"] = False
        return_update["dag_repair_instructions"] = None

    active_logger.info(f"DAGRepairNode: Returning. Repaired DAG: {return_update['task_dag'] is not current_dag if return_update['task_dag'] else False}, Error: '{return_update['dag_repair_error']}'", event_type="NODE_EXEC_COMPLETE", metadata={"node_name": "dag_repair_node"})
    return return_update 