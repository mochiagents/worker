from typing import Dict, Any, List, Optional
import json # For potential debugging or logging if needed.
from worker.core.models import TaskDAG, TaskNode # Core Pydantic models
from worker.core.logging import MochiLogger # For logging within the editor

class DAGEditor:
    """
    Responsible for applying repair instructions to a TaskDAG.
    """
    def __init__(self, logger: Optional[MochiLogger] = None):
        self.logger = logger or MochiLogger() # Basic default logger

    def apply_repair(self, current_dag: TaskDAG, repair_instructions: Dict[str, Any]) -> TaskDAG:
        """
        Applies a list of repair actions to the current DAG.

        Args:
            current_dag: The TaskDAG Pydantic model to be modified.
            repair_instructions: A dictionary containing a list of 'repair_actions'.
                                 Each action is a dict specifying the modification.

        Returns:
            A new TaskDAG instance with the repairs applied.

        Raises:
            ValueError: If repair_instructions are malformed or a repair action is invalid.
        """
        if not repair_instructions or "repair_actions" not in repair_instructions or not isinstance(repair_instructions["repair_actions"], list):
            self.logger.error("DAGEditor: Invalid or missing 'repair_actions' list in repair_instructions.", event_type="DAG_REPAIR_INVALID_INPUT")
            raise ValueError("Invalid or missing 'repair_actions' list.")

        # Create a deep copy of the DAG to modify. Pydantic models can be copied.
        # This ensures we don't modify the original DAG in place if it's from a shared state.
        # TaskDAG.model_copy(deep=True) is the Pydantic v2 way.
        # If using Pydantic v1, it would be current_dag.copy(deep=True)
        # Assuming TaskDAG is a Pydantic BaseModel.
        try:
            # Pydantic v2
            modified_dag_model = current_dag.model_copy(deep=True)
        except AttributeError:
            # Pydantic v1 fallback (or if model_copy is not on your Pydantic version)
            self.logger.warning("DAGEditor: model_copy(deep=True) failed, attempting copy(deep=True). Check Pydantic version.")
            modified_dag_model = current_dag.copy(deep=True)


        # Convert tasks list to a dictionary for easier lookup and modification by ID
        tasks_dict: Dict[str, TaskNode] = {task.id: task for task in modified_dag_model.tasks}

        for action in repair_instructions["repair_actions"]:
            action_type = action.get("action_type")
            self.logger.info(f"DAGEditor: Processing repair action: {action_type}", event_type="DAG_REPAIR_ACTION_PROCESS")

            if action_type == "MODIFY_TASK_INPUTS":
                task_id = action.get("task_id")
                updated_inputs = action.get("updated_inputs")
                if not task_id or updated_inputs is None or task_id not in tasks_dict:
                    self.logger.error(f"DAGEditor: Invalid MODIFY_TASK_INPUTS action: {action}", event_type="DAG_REPAIR_ACTION_INVALID")
                    raise ValueError(f"Invalid MODIFY_TASK_INPUTS action for task_id: {task_id}")
                tasks_dict[task_id].inputs = updated_inputs
                self.logger.info(f"DAGEditor: Modified inputs for task {task_id}", event_type="DAG_REPAIR_ACTION_SUCCESS")

            elif action_type == "ADD_TASK":
                new_task_def_dict = action.get("new_task_definition")
                if not new_task_def_dict:
                    self.logger.error(f"DAGEditor: Invalid ADD_TASK action, missing 'new_task_definition': {action}", event_type="DAG_REPAIR_ACTION_INVALID")
                    raise ValueError("Invalid ADD_TASK action, missing 'new_task_definition'")
                try:
                    new_task = TaskNode(**new_task_def_dict) # Validate and create TaskNode
                    if new_task.id in tasks_dict:
                        self.logger.error(f"DAGEditor: ADD_TASK action failed, task ID '{new_task.id}' already exists.", event_type="DAG_REPAIR_ACTION_INVALID")
                        raise ValueError(f"ADD_TASK action failed, task ID '{new_task.id}' already exists.")
                    tasks_dict[new_task.id] = new_task
                    self.logger.info(f"DAGEditor: Added new task {new_task.id}", event_type="DAG_REPAIR_ACTION_SUCCESS")
                except Exception as e: # Catch Pydantic validation errors or others
                    self.logger.error(f"DAGEditor: Error creating new task from definition: {new_task_def_dict}. Error: {e}", event_type="DAG_REPAIR_ACTION_INVALID")
                    raise ValueError(f"Error creating new task from definition: {e}")

            elif action_type == "MODIFY_TASK_DEPENDENCIES":
                task_id = action.get("task_id")
                updated_dependencies = action.get("updated_dependencies")
                if not task_id or updated_dependencies is None or task_id not in tasks_dict:
                    self.logger.error(f"DAGEditor: Invalid MODIFY_TASK_DEPENDENCIES action: {action}", event_type="DAG_REPAIR_ACTION_INVALID")
                    raise ValueError(f"Invalid MODIFY_TASK_DEPENDENCIES action for task_id: {task_id}")
                if not all(dep_id in tasks_dict for dep_id in updated_dependencies):
                    self.logger.error(f"DAGEditor: MODIFY_TASK_DEPENDENCIES action for task {task_id} includes non-existent dependency IDs.", event_type="DAG_REPAIR_ACTION_INVALID")
                    raise ValueError(f"MODIFY_TASK_DEPENDENCIES for task {task_id} includes non-existent dependency IDs.")
                tasks_dict[task_id].dependencies = updated_dependencies
                self.logger.info(f"DAGEditor: Modified dependencies for task {task_id}", event_type="DAG_REPAIR_ACTION_SUCCESS")

            elif action_type == "MODIFY_TASK_TOOL":
                task_id = action.get("task_id")
                if not task_id or task_id not in tasks_dict:
                    self.logger.error(f"DAGEditor: Invalid MODIFY_TASK_TOOL action, task_id missing or invalid: {action}", event_type="DAG_REPAIR_ACTION_INVALID")
                    raise ValueError(f"Invalid MODIFY_TASK_TOOL action, task_id missing or invalid: {task_id}")
                
                task_to_modify = tasks_dict[task_id]
                if "new_server_id" in action:
                    task_to_modify.server_id = action["new_server_id"]
                if "new_tool_name" in action:
                    task_to_modify.tool_name = action["new_tool_name"]
                self.logger.info(f"DAGEditor: Modified tool for task {task_id}", event_type="DAG_REPAIR_ACTION_SUCCESS")
            
            elif action_type == "NO_REPAIR_POSSIBLE":
                # This action type should ideally be handled before calling apply_repair,
                # but if received, it means no changes to the DAG.
                self.logger.info(f"DAGEditor: Received NO_REPAIR_POSSIBLE action. Reason: {action.get('reason')}. No changes applied to DAG.", event_type="DAG_REPAIR_NO_ACTION")
                # We can simply return the original (copied) DAG as no modification is done.
                # Or, if this means failure, an exception could be raised earlier.
                # For now, assume it means the Joiner already decided not to proceed with this repair.
                break # Stop processing further actions if one indicates no repair.

            else:
                self.logger.warning(f"DAGEditor: Unknown repair action_type: {action_type}", event_type="DAG_REPAIR_ACTION_UNKNOWN")
                # Optionally raise ValueError or just skip unknown actions
                # raise ValueError(f"Unknown repair action_type: {action_type}")

        # Reconstruct the tasks list in the modified_dag_model
        modified_dag_model.tasks = list(tasks_dict.values())
        
        # Optional: Add a validation step here to ensure the modified DAG is still structurally sound
        # e.g., check for circular dependencies, ensure all referenced dependencies exist.
        # For now, this is omitted for brevity.

        self.logger.info(f"DAGEditor: DAG repair application complete. Original task count: {len(current_dag.tasks)}, New task count: {len(modified_dag_model.tasks)}", event_type="DAG_REPAIR_COMPLETE")
        return modified_dag_model 