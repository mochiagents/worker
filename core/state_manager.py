"""In-memory state manager for Mochi agents."""

import os
import json
import threading
import warnings
from typing import Dict, Any, Optional, List, Tuple

try:
    from worker.agent.state import AgentState
    from worker.core.models import TaskDAG
except ImportError:
    from typing import TypedDict
    try:
        from worker.core.models import TaskDAG as PlaceholderTaskDAG
    except ImportError:
        PlaceholderTaskDAG = Any # Fallback if even Pydantic model can't be imported

    class AgentState(TypedDict):
        input_query: str
        final_response: Optional[str]
        needs_replanning: bool
        error_message: Optional[str]
        task_id: Optional[str]
        conversation_context: Optional[str]
        llm_api_key: Optional[str]
        config: Optional[Dict]
        dag: Optional[PlaceholderTaskDAG]
        planner_error: Optional[str]
        task_results: Dict[str, Any]
        task_status: Dict[str, str]
        current_task_id_to_execute: Optional[str]
        has_ready_tasks: bool
        all_completed: bool
        error: Optional[str]

def get_default_agent_state(input_query: str) -> AgentState:
    """Returns a default AgentState dictionary."""
    return AgentState(
        input_query=input_query,
        final_response=None,
        needs_replanning=False,
        error_message=None,
        task_id=None,
        conversation_context=None,
        llm_api_key=None,
        config=None,
        dag=None,
        planner_error=None,
        task_results={},
        task_status={},
        current_task_id_to_execute=None,
        has_ready_tasks=False,
        all_completed=False,
        error=None,
    )

class StateManager:
    """
    Manages the in-memory state for Mochi worker agents.
    Each state is an instance of AgentState, keyed by a unique state_id.
    Can optionally persist states to a directory as JSON files.
    This class is designed to be thread-safe.
    """
    def __init__(self, persistence_dir: Optional[str] = None):
        self._states: Dict[str, AgentState] = {}
        self.persistence_dir: Optional[str] = persistence_dir
        self._lock = threading.RLock()

        if self.persistence_dir:
            try:
                if not os.path.exists(self.persistence_dir):
                    os.makedirs(self.persistence_dir)
            except OSError as e:
                warnings.warn(f"StateManager: Error creating persistence directory {self.persistence_dir}: {e}. Persistence will be disabled.", UserWarning)
                self.persistence_dir = None

    def _get_state_filepath(self, state_id: str) -> Optional[str]:
        if not self.persistence_dir:
            return None
        return os.path.join(self.persistence_dir, f"{state_id}.json")

    def _persist_state(self, state_id: str) -> bool:
        if not self.persistence_dir or state_id not in self._states:
            return False
        
        filepath = self._get_state_filepath(state_id)
        if not filepath:
            return False

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._states[state_id], f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"StateManager: Error persisting state '{state_id}' to {filepath}: {e}")
            return False
        except TypeError as e:
            print(f"StateManager: TypeError while serializing state '{state_id}': {e}")
            return False


    def _load_state_from_disk(self, state_id: str) -> Optional[AgentState]:
        if not self.persistence_dir:
            return None

        filepath = self._get_state_filepath(state_id)
        if not filepath or not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict) and 'input_query' in loaded_data:
                    self._states[state_id] = loaded_data
                    return self._states[state_id]
                else:
                    print(f"StateManager: Loaded data from {filepath} for state '{state_id}' does not seem to be a valid AgentState.")
                    return None
        except (IOError, json.JSONDecodeError) as e:
            print(f"StateManager: Error loading state '{state_id}' from {filepath}: {e}")
            return None

    def create_state(self, state_id: str, initial_input_query: Optional[str] = "Default initial query") -> AgentState:
        """
        Creates a new agent state in memory and persists it if configured.
        This method is thread-safe.

        Args:
            state_id: The unique identifier for the state.
            initial_input_query: The initial query to set for the new state.

        Returns:
            The newly created AgentState.

        Raises:
            ValueError: If a state with the given state_id already exists.
        """
        with self._lock:
            if state_id in self._states:
                if self.persistence_dir:
                    loaded_state = self._load_state_from_disk(state_id)
                    if loaded_state:
                        return loaded_state
                raise ValueError(f"State with ID '{state_id}' already exists in memory or could not be loaded from potential disk state.")
            
            new_state = get_default_agent_state(initial_input_query if initial_input_query is not None else "Default initial query")
            self._states[state_id] = new_state
            self._persist_state(state_id)
            return new_state

    def get_state(self, state_id: str) -> Optional[AgentState]:
        """
        Retrieves an agent state, trying memory first, then disk if persistence is enabled.
        This method is thread-safe.

        Args:
            state_id: The identifier of the state to retrieve.

        Returns:
            The AgentState if found, otherwise None.
        """
        with self._lock:
            if state_id in self._states:
                return self._states[state_id]
            
            loaded_state = self._load_state_from_disk(state_id)
            if loaded_state:
                return loaded_state
            
            return None

    def update_state(self, state_id: str, updates: Dict[str, Any]) -> AgentState:
        """
        Updates an existing agent state in memory and persists it.
        Only keys present in the AgentState TypedDict will be updated.
        This method is thread-safe.

        Args:
            state_id: The identifier of the state to update.
            updates: A dictionary containing the fields to update and their new values.

        Returns:
            The updated AgentState.

        Raises:
            KeyError: If the state with the given state_id is not found (neither in memory nor on disk).
        """
        with self._lock:
            if state_id not in self._states:
                if not self._load_state_from_disk(state_id):
                    raise KeyError(f"State with ID '{state_id}' not found.")

            state_to_update = self._states[state_id]
            
            for key, value in updates.items():
                if key in AgentState.__annotations__:
                    state_to_update[key] = value

            self._states[state_id] = state_to_update 
            self._persist_state(state_id)
            return state_to_update

    def delete_state(self, state_id: str) -> bool:
        """
        Deletes an agent state from memory and its persisted file if it exists.
        This method is thread-safe.

        Args:
            state_id: The identifier of the state to delete.

        Returns:
            True if the state was deleted, False if the state was not found.
        """
        with self._lock:
            deleted_from_memory = False
            if state_id in self._states:
                del self._states[state_id]
                deleted_from_memory = True
            
            deleted_from_disk = False
            if self.persistence_dir:
                filepath = self._get_state_filepath(state_id)
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        deleted_from_disk = True
                    except OSError as e:
                        print(f"StateManager: Error deleting persisted state file {filepath}: {e}")
            
            return deleted_from_memory or deleted_from_disk

    def has_state(self, state_id: str) -> bool:
        """
        Checks if a state with the given ID exists in memory or on disk if persistence is enabled.
        This method is thread-safe.

        Args:
            state_id: The identifier of the state to check.

        Returns:
            True if the state exists, False otherwise.
        """
        with self._lock:
            if state_id in self._states:
                return True
            if self.persistence_dir:
                filepath = self._get_state_filepath(state_id)
                if filepath and os.path.exists(filepath):
                    return True
            return False

    def list_states(self) -> List[str]:
        """
        Lists the IDs of all states currently managed in memory and/or on disk.
        This method is thread-safe.

        Returns:
            A list of unique state_ids.
        """
        with self._lock:
            memory_keys = set(self._states.keys())
            disk_keys = set()
            if self.persistence_dir:
                try:
                    for filename in os.listdir(self.persistence_dir):
                        if filename.endswith(".json"):
                            disk_keys.add(filename[:-5])
                except OSError as e:
                    print(f"StateManager: Error listing states from disk directory {self.persistence_dir}: {e}")

            return list(memory_keys.union(disk_keys)) 