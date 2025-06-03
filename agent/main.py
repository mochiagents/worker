"""
Main entry point for the Mochi worker agent.
Initializes and orchestrates the core components of the agent.
"""
import logging
import sys 
from typing import Dict, List, Optional, Any
import json
import asyncio
import uuid 
import signal
import time
from datetime import datetime, timezone

from worker.config.manager import ConfigurationManager
from worker.config.models import MochiWorkerConfig
from worker.core.logging import MochiLogger
from worker.core.state_manager import StateManager
from worker.core.llm_service import LLMService, LLMServiceError
from worker.core.heartbeat import HeartbeatManager, HealthMetrics, HealthAlert
from worker.agent.planner import Planner
from worker.agent.execution import TaskFetchingUnit
from worker.agent.joiner import Joiner
from worker.agent.mcp_client import McpClient, McpError
from worker.agent.graph import app as lang_graph_app 
from worker.agent.state import AgentState 
from worker.core.models import TaskDAG, ToolExecutionResult
from worker.prompts.planner_prompts import PlannerPromptBuilder
from worker.prompts.joiner_prompts import JoinerPromptBuilder
from worker.agent.dag_editor import DAGEditor
from worker.config.exceptions import MochiConfigError 

class MochiAgent:
    """
    The main Mochi worker agent class.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes all components of the Mochi agent.
        """
        self.config_manager = ConfigurationManager(default_config_path=config_path)
        try:
            self.config: MochiWorkerConfig = self.config_manager.get_config()
        except Exception as e:
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"CRITICAL: Failed to load MochiWorkerConfig: {e}", exc_info=True)
            raise
        
        try:
            self.logger = MochiLogger(config=self.config.logging)
        except Exception as e:
            logging.error(f"CRITICAL: Failed to initialize MochiLogger: {e}. Falling back to basic logging.", exc_info=True)
            self.logger = logging.getLogger("MochiAgent_fallback")
            self.logger.error(f"MochiLogger init failed, using basic logger. Error: {e}")

        self.managed_conversation_histories: Dict[str, List[Dict[str, str]]] = {}

        self.logger.info(f"Mochi Agent initializing with agent_id: {self.config.agent_id}...", event_type="AGENT_INIT_START")

        try:
            self.llm_service = LLMService(config=self.config, logger_instance=self.logger)
            self.logger.info("LLMService initialized.", event_type="COMPONENT_INIT")
        except LLMServiceError as e:
            self.logger.error(f"Failed to initialize LLMService: {e}", event_type="LLM_SERVICE_INIT_FAILURE")
            raise
        except Exception as e: 
            self.logger.error(f"Unexpected error initializing LLMService: {e}", exc_info=True, event_type="LLM_SERVICE_INIT_FAILURE_UNEXPECTED")
            raise

        persistence_dir: Optional[str] = self.config.agent_settings.state_persistence_dir
        self.state_manager = StateManager(persistence_dir=persistence_dir)
        self.logger.info(f"StateManager initialized. Persistence: {'Enabled - path: ' + persistence_dir if persistence_dir else 'Disabled (in-memory)'}", event_type="COMPONENT_INIT") # type: ignore

        self.mcp_clients: Dict[str, McpClient] = {}
        if self.config.mcp_tool_servers:
            for server_conf in self.config.mcp_tool_servers:
                self.logger.info(f"Initializing MCPClient for server: {server_conf.name} at endpoint: {server_conf.endpoint_url}", event_type="MCP_CLIENT_INIT") # type: ignore
                
                client = McpClient(
                    server_name=server_conf.name,
                    server_address=server_conf.endpoint_url, 
                    logger_instance=self.logger,
                    server_config_override=server_conf 
                )
                self.mcp_clients[server_conf.name] = client
        else:
            self.logger.warning("No MCP tool servers configured.", event_type="CONFIG_WARNING")
        
        try:
            planner_llm_profile_name = self.config.planner.llm_profile_name
            joiner_llm_profile_name = self.config.joiner.llm_profile_name

            if not planner_llm_profile_name:
                error_msg = "Planner LLM profile name (planner.llm_profile_name) not configured! Ensure it is set in your configuration file and points to a valid profile in 'llm_profiles'."
                self.logger.error(error_msg, event_type="CONFIG_ERROR_LLM_PROFILE")
                raise MochiConfigError(error_msg)
            if not joiner_llm_profile_name:
                error_msg = "Joiner LLM profile name (joiner.llm_profile_name) not configured! Ensure it is set in your configuration file and points to a valid profile in 'llm_profiles'."
                self.logger.error(error_msg, event_type="CONFIG_ERROR_LLM_PROFILE")
                raise MochiConfigError(error_msg)

            self.planner_llm = self.llm_service.get_llm(planner_llm_profile_name)
            self.logger.info(f"Planner LLM from profile '{planner_llm_profile_name}' successfully initialized: {type(self.planner_llm)}", event_type="LLM_INIT_SUCCESS")
            
            self.joiner_llm = self.llm_service.get_llm(joiner_llm_profile_name)
            self.logger.info(f"Joiner LLM from profile '{joiner_llm_profile_name}' successfully initialized: {type(self.joiner_llm)}", event_type="LLM_INIT_SUCCESS")

        except LLMServiceError as e_service:
            self.logger.error(f"LLMServiceError initializing LLMs: {e_service}", event_type="LLM_INIT_FAILURE")
            raise
        except MochiConfigError:
            raise
        except ValueError as e_value:
            error_msg = f"ValueError related to LLM profile configuration (e.g., profile name specified but not found in 'llm_profiles'): {e_value}"
            self.logger.error(error_msg, event_type="CONFIG_ERROR_LLM_PROFILE")
            raise MochiConfigError(error_msg) from e_value 
        except Exception as e_llm_init_general:
            self.logger.error(f"Unexpected error initializing LLMs via LLMService: {e_llm_init_general}", exc_info=True, event_type="LLM_INIT_FAILURE_UNEXPECTED")
            raise
        
        planner_prompt_builder = PlannerPromptBuilder()
        self.planner = Planner(
            llm=self.planner_llm,
            settings=self.config.planner,
            mcp_clients=self.mcp_clients,
            prompt_builder=planner_prompt_builder,
            logger_instance=self.logger
        )
        self.logger.info("Planner initialized.", event_type="COMPONENT_INIT")

        self.task_fetcher = TaskFetchingUnit(
            mcp_clients=self.mcp_clients,
            synthesis_llm=self.joiner_llm,
            settings=self.config,
            logger_instance=self.logger
        )
        self.logger.info("TaskFetchingUnit (and internal TaskExecutor) initialized.", event_type="COMPONENT_INIT")

        joiner_prompt_builder = JoinerPromptBuilder()
        self.joiner = Joiner(
            llm=self.joiner_llm,
            settings=self.config.joiner,
            prompt_builder=joiner_prompt_builder,
            logger=self.logger
        )
        self.logger.info("Joiner initialized.", event_type="COMPONENT_INIT")

        self.dag_editor = DAGEditor(logger=self.logger)
        self.logger.info("DAGEditor initialized.", event_type="COMPONENT_INIT")

        self.lang_graph = lang_graph_app
        self.logger.info("LangGraph application loaded.", event_type="COMPONENT_INIT")
        
        self.agent_id = self.config.agent_id
        self.shutdown_event = asyncio.Event()
        
        # Initialize the enhanced heartbeat system
        self.heartbeat_manager = HeartbeatManager(
            agent_id=self.agent_id,
            interval_seconds=self.config.agent_settings.heartbeat_interval_seconds,
            logger=self.logger,
            enable_system_metrics=True,
            enable_alerts=True,
            metrics_history_size=100
        )
        self.logger.info("HeartbeatManager initialized.", event_type="COMPONENT_INIT")
        
        self.logger.info(f"[AGENT_INIT_COMPLETE] Mochi Agent initialized successfully.", event_type="AGENT_INIT_COMPLETE")

    def _update_heartbeat_task_counters(self, active: Optional[int] = None, completed_delta: int = 0, failed_delta: int = 0):
        """Update task counters in the heartbeat manager."""
        if hasattr(self, 'heartbeat_manager'):
            self.heartbeat_manager.update_task_counters(
                active=active,
                completed_delta=completed_delta,
                failed_delta=failed_delta,
                last_successful=time.time() if completed_delta > 0 else None
            )

    def _update_heartbeat_mcp_counters(self):
        """Update MCP connection counters in the heartbeat manager."""
        if hasattr(self, 'heartbeat_manager') and hasattr(self, 'mcp_clients'):
            active_connections = len([c for c in self.mcp_clients.values() if getattr(c, '_is_initialized', False)])
            failed_connections = len([c for c in self.mcp_clients.values() if not getattr(c, '_is_initialized', False)])
            self.heartbeat_manager.update_mcp_counters(active_connections, failed_connections)

    def add_health_alert(self, alert: HealthAlert):
        """Add a custom health alert to the heartbeat system."""
        if hasattr(self, 'heartbeat_manager'):
            self.heartbeat_manager.add_alert(alert)

    def add_health_callback(self, callback):
        """Add a health monitoring callback to the heartbeat system."""
        if hasattr(self, 'heartbeat_manager'):
            self.heartbeat_manager.add_health_callback(callback)

    def _format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """Formats a list of message dictionaries into a single string for the prompt."""
        if not history:
            return ""
        
        formatted_lines = []
        for message in history:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    async def _generate_clarification_request(self, original_query: str, last_joiner_explanation: Optional[str]) -> str:
        """
        Generates a message to the user asking for clarification when the agent
        hits the maximum replanning cycles.
        """
        if not self.joiner_llm:
            self.logger.error("Joiner LLM not available for generating clarification request.", event_type="CLARIFICATION_ERROR_NO_LLM")
            return "I've tried a few times but I'm still having trouble with your request. Could you please try rephrasing it or provide more details?"

        prompt_template = """You are a helpful AI assistant. The system you are part of has tried multiple times to answer the user's query but has reached its processing limits without full success.
Your goal is to explain this to the user and ask for clarification.

User's Original Query:
{original_query}

The system's last internal assessment before stopping was:
{last_joiner_explanation}

Based on this, please:
1. Briefly inform the user that the request could not be fully completed after multiple attempts.
2. Using the "system's last internal assessment" above, explain in simple terms what the system was struggling with or what information it might still need.
3. Ask a clear and concise question to the user that would help clarify the request or provide the missing information.
Avoid technical jargon. Be polite and helpful.
"""
        
        explanation_text = last_joiner_explanation if last_joiner_explanation else "The system could not pinpoint a specific reason but was unable to proceed effectively."

        prompt = prompt_template.format(original_query=original_query, last_joiner_explanation=explanation_text)
        
        self.logger.info("Generating clarification request to user.", event_type="CLARIFICATION_REQUEST_GEN_START")
        try:
            llm_response = await self.joiner_llm.ainvoke(prompt) # Assuming joiner_llm supports ainvoke
            clarification_message = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            self.logger.info("Successfully generated clarification request.", event_type="CLARIFICATION_REQUEST_GEN_SUCCESS")
            return clarification_message.strip()
        except Exception as e:
            self.logger.error(f"Error generating clarification request using LLM: {e}", exc_info=True, event_type="CLARIFICATION_REQUEST_LLM_ERROR")
            return "I've tried a few times but I'm having trouble with your request due to an internal error. Could you please try rephrasing or provide more details?"

    async def run_managed_query(
        self, 
        query: str, 
        conversation_id: str, 
        stream_callback: Optional[Any] = None,
        user_info: Optional[Dict[str, Any]] = None,
        session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs a query for a given conversation_id, managing its history automatically.
        Also incorporates additional contextual information into the prompt if provided.
        """
        self.logger.info(f"Running managed query for conversation_id: {conversation_id}", event_type="MANAGED_QUERY_START")

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.logger.warning(f"No conversation_id provided for managed query. Generated new one: {conversation_id}", event_type="MANAGED_QUERY_WARN")

        if conversation_id not in self.managed_conversation_histories:
            self.managed_conversation_histories[conversation_id] = []
            self.logger.info(f"New conversation started with id: {conversation_id}", event_type="CONVERSATION_START")

        self.managed_conversation_histories[conversation_id].append({"role": "user", "content": query})

        formatted_history_string = self._format_history_for_prompt(self.managed_conversation_histories[conversation_id])
        
        additional_context_parts = []
        if user_info:
            additional_context_parts.append(f"User Information: {json.dumps(user_info)}")
        if session_data:
            additional_context_parts.append(f"Session Data: {json.dumps(session_data)}")
        
        additional_context_str = ""
        if additional_context_parts:
            additional_context_str = "\n\n--- Additional Context ---\n" + "\n".join(additional_context_parts)
            
        full_conversation_context = formatted_history_string + additional_context_str

        current_run_id = str(uuid.uuid4())

        self.logger.debug(f"Conversation ID: {conversation_id}, Current Run ID: {current_run_id}, Query: '{query}'", event_type="MANAGED_QUERY_DETAIL")
        self.logger.debug(f"Formatted conversation context for run {current_run_id}:{full_conversation_context}", event_type="MANAGED_QUERY_CONTEXT")

        result_dict = await self.run_query(
            query=query, 
            conversation_context=full_conversation_context, 
            run_id=current_run_id,
            stream_callback=stream_callback
        )

        final_response = result_dict.get("answer")
        if final_response:
            self.managed_conversation_histories[conversation_id].append({"role": "agent", "content": final_response})
            self.logger.info(f"Agent response for conversation_id {conversation_id} (run {current_run_id}) added to history.", event_type="MANAGED_QUERY_RESPONSE_STORED")
        elif result_dict.get("error_message"):
            error_msg = result_dict.get("error_message", "Unknown error")
            self.managed_conversation_histories[conversation_id].append({"role": "agent", "content": f"[Agent Error: {error_msg}]"})
            self.logger.warning(f"Agent encountered an error for conversation_id {conversation_id} (run {current_run_id}). Error stored in history: {error_msg}", event_type="MANAGED_QUERY_ERROR_STORED")
        else:
            self.managed_conversation_histories[conversation_id].append({"role": "agent", "content": "[Agent did not provide a final response]"})
            self.logger.info(f"No final response from agent for conversation_id {conversation_id} (run {current_run_id}). Placeholder stored.", event_type="MANAGED_QUERY_NO_RESPONSE_STORED")
            
        self.logger.info(f"Managed query for conversation_id: {conversation_id} (run {current_run_id}) finished.", event_type="MANAGED_QUERY_END")
        return result_dict

    async def _initialize_mcp_clients(self):
        """Asynchronously initializes all configured MCP clients."""
        if not self.mcp_clients:
            self.logger.info("No MCP clients to initialize.")
            return
        
        self.logger.info("Asynchronously initializing MCP clients...")
        init_tasks = []
        for server_id, client in self.mcp_clients.items():
            if hasattr(client, 'initialize') and asyncio.iscoroutinefunction(client.initialize):
                self.logger.info(f"Queueing initialization for MCP client: {server_id}")
                init_tasks.append(client.initialize())
            else:
                self.logger.info(f"MCP client {server_id} does not have an async initialize method or is already handled.")

        if init_tasks:
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            for i, server_id in enumerate(self.mcp_clients.keys()):
                if isinstance(results[i], Exception):
                    self.logger.error(f"Failed to initialize MCP client {server_id}: {results[i]}", event_type="MCP_CLIENT_INIT_FAILURE")
                else:
                    self.logger.info(f"MCP client {server_id} initialized successfully via async call.", event_type="MCP_CLIENT_INIT_SUCCESS")
            
            # Update MCP connection counters after initialization attempts
            self._update_heartbeat_mcp_counters()
        self.logger.info("Async MCP client initialization complete.", event_type="MCP_CLIENT_INIT_BATCH_COMPLETE")

    async def start(self):
        """
        Asynchronously starts the Mochi agent and its long-running services.
        Ensures all components, especially async ones like MCP clients, are initialized.
        """
        self.logger.info("MochiAgent starting asynchronously...")
        await self._initialize_mcp_clients()
        
        # Update MCP connection counters after initialization
        self._update_heartbeat_mcp_counters()
        
        # Start enhanced heartbeat system if enabled
        if self.config.agent_settings.enable_heartbeat:
            if not self.heartbeat_manager.is_running:
                self.logger.info("Starting enhanced heartbeat system.")
                await self.heartbeat_manager.start()
            else:
                self.logger.warning("Heartbeat system already running.")
        else:
            self.logger.info("Heartbeat is disabled by configuration.")

        self.logger.info("MochiAgent async start sequence complete.", event_type="AGENT_START_COMPLETE")

    async def shutdown(self, signum=None, frame=None):
        """
        Asynchronously shuts down the Mochi agent and its long-running services gracefully.
        """
        if self.shutdown_event.is_set():
            self.logger.info("Shutdown already in progress.")
            return
            
        signal_name = signal.Signals(signum).name if signum else "programmatically"
        self.logger.info(f"Shutdown initiated by signal {signal_name}...", event_type="AGENT_SHUTDOWN_START")
        self.shutdown_event.set()

        # Stop enhanced heartbeat system
        if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager.is_running:
            self.logger.info("Stopping enhanced heartbeat system...")
            try:
                await self.heartbeat_manager.stop()
                self.logger.info("Heartbeat system successfully stopped.")
            except Exception as e:
                self.logger.error(f"Error during heartbeat system shutdown: {e}", exc_info=True)
        
        self.logger.info("[AGENT_SHUTDOWN_START] MochiAgent shutting down asynchronously...", event_type="AGENT_SHUTDOWN_START")
        tasks = []
        if hasattr(self, 'mcp_clients') and self.mcp_clients:
            for name, client in self.mcp_clients.items():
                if hasattr(client, 'closeGracefully') and asyncio.iscoroutinefunction(client.closeGracefully):
                    self.logger.info(f"Queueing graceful shutdown for MCP client: {name}")
                    tasks.append(client.closeGracefully())
                else:
                    self.logger.warning(f"MCP client {name} does not have a closeGracefully method, skipping graceful shutdown.", event_type="MCP_CLIENT_SHUTDOWN_SKIP")
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error during graceful shutdown of an MCP client: {result}", event_type="MCP_CLIENT_SHUTDOWN_ERROR")
                self.logger.info("All queued MCP clients processed for shutdown.", event_type="MCP_CLIENT_SHUTDOWN_BATCH_COMPLETE")
            else:
                self.logger.info("No MCP clients required shutdown.", event_type="MCP_CLIENT_SHUTDOWN_BATCH_COMPLETE")
        else:
            self.logger.info("No MCP clients dictionary found during shutdown.", event_type="AGENT_SHUTDOWN_SKIP_NO_CLIENTS")
            
        # TODO: Add cancellation logic for any ongoing agent runs (e.g., graph invocations) if possible/needed.
        self.logger.info("Shutdown: Cancellation logic for ongoing agent runs is not yet fully implemented and is a future enhancement.", event_type="AGENT_SHUTDOWN_NOTE")
        
        self.logger.info("[AGENT_SHUTDOWN_COMPLETE] MochiAgent shutdown sequence complete.", event_type="AGENT_SHUTDOWN_COMPLETE")

    async def _execute_graph_for_state(self, agent_state_dict: dict, stream_callback: Optional[Any] = None) -> dict:
        """Executes the LangGraph application loop for a given agent state (single DAG processing)."""
        current_iteration_state_dict = agent_state_dict.copy() 
        
        # Add stream_callback to the state dict so nodes can access it
        # This field is excluded from serialization in AgentState, so it's safe to add here
        current_iteration_state_dict['stream_callback'] = stream_callback
        
        # max_dag_replanning_cycles refers to how many times we can replan for *this specific DAG*
        max_dag_replanning_cycles = self.config.agent_settings.max_replanning_cycles 
        
        # 'replanning_cycles' should be initialized to 0 in the input agent_state_dict 
        # before this method is called for a new DAG.
        # This loop_count tracks attempts for the current DAG.
        loop_count = current_iteration_state_dict.get('replanning_cycles', 0)

        while loop_count < max_dag_replanning_cycles:
            self.logger.info(f"Starting graph invocation for current DAG. Attempt {loop_count + 1}/{max_dag_replanning_cycles}. Run_id: {current_iteration_state_dict.get('run_id')}, Phase: {current_iteration_state_dict.get('current_phase_id', 'N/A')}", event_type="AGENT_GRAPH_LOOP_START")
            if stream_callback:
                stream_callback({"event_type": "graph_loop_start", "loop_count": loop_count + 1, "max_loops": max_dag_replanning_cycles, "current_state": current_iteration_state_dict.copy()})
            
            # Update active task count for heartbeat monitoring
            dag_model = current_iteration_state_dict.get('task_dag')
            if dag_model and isinstance(dag_model, dict) and 'tasks' in dag_model:
                active_task_count = len(dag_model['tasks'])
                self._update_heartbeat_task_counters(active=active_task_count)
            
            # Invoke the graph. The input current_iteration_state_dict is a dictionary.
            # The output from lang_graph.ainvoke is an AddableValuesDict (a LangGraph internal type).
            final_graph_output_addable_dict = await self.lang_graph.ainvoke(
                current_iteration_state_dict
            )
            
            # Convert the AddableValuesDict back to a standard dictionary to be safe
            # and then to our AgentState Pydantic model for consistent handling.
            # This also ensures that any new fields added by LangGraph nodes (if any) are captured if they match AgentState fields.
            final_graph_output_dict = dict(final_graph_output_addable_dict)
            current_iteration_state_model = AgentState(**final_graph_output_dict) # Convert to Pydantic model
            current_iteration_state_dict = current_iteration_state_model.model_dump(exclude_none=True) # Back to dict for the loop

            self.logger.debug(f"[DEBUG_GRAPH_LOOP_END_STATE] State after graph invoke. run_id: {current_iteration_state_dict.get('run_id')}, phase: {current_iteration_state_dict.get('current_phase_id', 'N/A')}, loop: {loop_count + 1}", 
                              metadata={
                                  "needs_replanning": current_iteration_state_dict.get("needs_replanning"),
                                  "final_response_type": str(type(current_iteration_state_dict.get("final_response"))),
                                  "final_response_preview": str(current_iteration_state_dict.get("final_response"))[:200] if current_iteration_state_dict.get("final_response") else None,
                                  "replan_context_type": str(type(current_iteration_state_dict.get("replan_context"))),
                                  "replan_context_preview": str(current_iteration_state_dict.get("replan_context"))[:200] if current_iteration_state_dict.get("replan_context") else None,
                                  "overall_status": current_iteration_state_dict.get("overall_status")
                              })

            if stream_callback:
                # It's better to stream the dict representation for serialization safety.
                stream_callback({"event_type": "graph_loop_end", "loop_count": loop_count + 1, "output_state": current_iteration_state_dict.copy()})

            # current_iteration_state_dict is now the dict representation of the state after graph run
            current_iteration_state_dict['replanning_cycles'] = loop_count # Persist current attempt count back

            dag_model = current_iteration_state_model.task_dag # Access task_dag from the model
            if dag_model and hasattr(dag_model, "tasks") and isinstance(dag_model.tasks, list):
                if (
                    len(dag_model.tasks) == 1 and
                    getattr(dag_model.tasks[0], "tool_name", None) == "cannot_answer_without_tools"
                ):
                    self.logger.info("Planner returned 'cannot_answer_without_tools'. Exiting graph execution loop for this DAG.", event_type="AGENT_GRAPH_CANNOT_ANSWER")
                    if stream_callback:
                        stream_callback({"event_type": "cannot_answer_without_tools", "final_state": current_iteration_state_dict.copy()})
                    break

            needs_replanning_value = current_iteration_state_dict.get("needs_replanning") # Get the value, could be True, False, or None
            if needs_replanning_value is None: # Explicitly check for None
                needs_replanning_value = False # Treat None as False for replanning decisions
            
            self.logger.debug(f"Graph attempt {loop_count + 1}: 'needs_replanning' evaluated to {needs_replanning_value} (original value from dict: {current_iteration_state_dict.get('needs_replanning')})", event_type="AGENT_GRAPH_REPLAN_CHECK")

            if not needs_replanning_value:
                self.logger.info(f"Graph attempt {loop_count + 1}: Replanning not needed for current DAG. Exiting graph execution loop.", event_type="AGENT_GRAPH_LOOP_EXIT_NO_REPLAN")
                if stream_callback:
                    stream_callback({"event_type": "graph_loop_exit_no_replan", "final_state": current_iteration_state_dict.copy()})
                
                # Update task completion metrics in heartbeat system
                task_results = current_iteration_state_dict.get("task_results", {})
                completed_count = len([r for r in task_results.values() if r and not str(r).startswith("Error")])
                failed_count = len([r for r in task_results.values() if r and str(r).startswith("Error")])
                if completed_count > 0:
                    self._update_heartbeat_task_counters(completed_delta=completed_count)
                if failed_count > 0:
                    self._update_heartbeat_task_counters(failed_delta=failed_count)
                
                break
            else:
                loop_count += 1 # Increment before next iteration
                current_iteration_state_dict['replanning_cycles'] = loop_count # Update for next graph input

                if loop_count >= max_dag_replanning_cycles:
                    self.logger.warning(f"Reached max_replanning_cycles ({max_dag_replanning_cycles}) for the current DAG processing (run_id: {current_iteration_state_dict.get('run_id')}, Phase: {current_iteration_state_dict.get('current_phase_id', 'N/A')}). Forcing graph loop exit.", event_type="AGENT_GRAPH_MAX_CYCLES_REACHED")
                    current_iteration_state_dict["needs_replanning"] = False # Stop further replanning attempts for this DAG
                    
                    # Preserve the explanation why replanning was initially requested before overriding final_response
                    last_joiner_explanation = current_iteration_state_dict.get("final_response")
                    if 'replan_context' not in current_iteration_state_dict or current_iteration_state_dict['replan_context'] is None:
                        current_iteration_state_dict['replan_context'] = {}
                    current_iteration_state_dict['replan_context']['last_attempt_explanation'] = last_joiner_explanation
                    current_iteration_state_dict['replan_context']['reason_for_clarification'] = "Max replanning cycles reached."

                    original_query_for_clarification = current_iteration_state_dict.get("original_query", "")
                    
                    clarification_message = await self._generate_clarification_request(original_query_for_clarification, last_joiner_explanation)
                    
                    current_iteration_state_dict["clarification_message"] = clarification_message
                    current_iteration_state_dict["status"] = "NEEDS_CLARIFICATION" # LangGraph does not have a 'status' field, this is for Mochi's overall_status
                    current_iteration_state_dict["overall_status"] = "needs_clarification" # Matching AgentState field
                    current_iteration_state_dict["error_message"] = f"Processing for the current plan/DAG reached maximum replanning cycles ({max_dag_replanning_cycles}) and requires clarification."
                    current_iteration_state_dict["final_response"] = clarification_message # User-facing message is the clarification
                    # Keep planner_error and other errors as they might be relevant context for clarification

                    if stream_callback:
                        stream_callback({"event_type": "max_dag_replanning_cycles_reached_needs_clarification", "final_state": current_iteration_state_dict.copy()})
                    break 
                else:
                    self.logger.info(f"Graph attempt {loop_count}: Replanning needed for current DAG (Next Attempt {loop_count + 1}/{max_dag_replanning_cycles}). Continuing graph execution loop.", event_type="AGENT_GRAPH_LOOP_CONTINUE_REPLAN")
                    
                    # Store the reason for replanning (from joiner, currently in final_response) into replan_context
                    if 'replan_context' not in current_iteration_state_dict or current_iteration_state_dict['replan_context'] is None:
                        current_iteration_state_dict['replan_context'] = {}
                    current_iteration_state_dict['replan_context']['last_attempt_explanation'] = current_iteration_state_dict.get("final_response")
                    current_iteration_state_dict['replan_context']['previous_attempt_failed'] = True


                    # Reset fields for the new planning attempt
                    current_iteration_state_dict['final_response'] = None # Clear final_response so it doesn't become the answer
                    current_iteration_state_dict['task_dag'] = None # Planner will generate a new one
                    current_iteration_state_dict['generated_dag'] = None # Old alias for task_dag
                    current_iteration_state_dict['planner_error'] = None
                    current_iteration_state_dict['task_statuses'] = {} # Reset task statuses
                    current_iteration_state_dict['task_results'] = {} # Reset task results
                    current_iteration_state_dict['joiner_error'] = None
                    current_iteration_state_dict['execution_error'] = None 
                    # accumulated_global_task_outputs is per-phase, should not be reset here if we are in a multi-phase plan's later phase.
                    # For a single DAG replan (SIMPLE query or one phase of COMPLEX), this is fine.
                    # If COMPLEX phase replanning occurs, this might need more nuanced handling for accumulated_global_task_outputs.
                    # For now, assume replan_context gives planner enough info if it needs prior phase results.


                    if stream_callback:
                        # Stream the state *before* it goes into the next replan attempt
                        # This state now includes the populated replan_context
                        stream_callback({"event_type": "graph_replan_triggered", "state_for_replan": current_iteration_state_dict.copy(), "next_loop_count": loop_count +1, "replanning_cycle": loop_count})
                    
                    # 'needs_replanning' is True, so graph will go back to planner.
        else: # Corresponds to while loop finishing because loop_count >= max_dag_replanning_cycles
            if loop_count >= max_dag_replanning_cycles: # This condition is met if the loop terminated due to cycle limit
                # This case should now be handled by the NEEDS_CLARIFICATION block above.
                # If it somehow gets here without that, log it.
                if not current_iteration_state_dict.get("status") == "NEEDS_CLARIFICATION":
                    self.logger.warning(f"Exited DAG execution loop due to max_dag_replanning_cycles ({max_dag_replanning_cycles}) but status was not set to NEEDS_CLARIFICATION. Run_id: {current_iteration_state_dict.get('run_id')}, Phase: {current_iteration_state_dict.get('current_phase_id', 'N/A')}", event_type="AGENT_GRAPH_MAX_LOOPS_UNHANDLED_EXIT")
                    current_iteration_state_dict["error_message"] = f"Agent stopped processing current DAG: Maximum replanning cycles ({max_dag_replanning_cycles}) reached."
                    current_iteration_state_dict["error"] = "MAX_DAG_REPLANNING_CYCLES_REACHED_UNHANDLED"
                    if stream_callback:
                        stream_callback({"event_type": "max_dag_replanning_cycles_reached_unhandled", "final_state": current_iteration_state_dict.copy()})
        
        return current_iteration_state_dict

    async def run_query(self, query: str, conversation_context: Optional[str] = None, run_id: Optional[str] = None, stream_callback: Optional[Any] = None) -> Dict[str, Any]:
        """Runs the main agent logic with a given query, adopting hierarchical planning strategy."""
        current_run_id = run_id if run_id else str(uuid.uuid4())
        self.logger.info(f"Starting run_query for query: '{query}'. Run ID: {current_run_id}", event_type="RUN_QUERY_START", metadata={"run_id": current_run_id, "query": query})

        # Initialize AgentState Pydantic model
        agent_state_model = AgentState(
            original_query=query,
            conversation_context=conversation_context,
            run_id=current_run_id,
            logger=self.logger,
            config=self.config.model_dump(exclude_none=True) if self.config else None,
            planner_instance=self.planner,
            task_fetching_unit_instance=self.task_fetcher,
            joiner_instance=self.joiner,
            mcp_clients=self.mcp_clients,
            dag_editor_instance=self.dag_editor,
            hierarchical_plan=None, 
            task_dag=None,
            task_results={},
            task_statuses={},
            final_response=None,
            error_message=None,
            all_completed=False,
            has_ready_tasks=False,
            execution_error=None,
            current_iteration_log=[],
            replanning_cycles=0,
            replan_context={},
            current_phase_id=None,
            current_phase_description=None,
            dag_repair_error=None,
            dag_repair_instructions=None,
            current_task_id_to_execute=None,
            planner_output={},
            joiner_error=None,
            repair_instructions_available=False,
            planner_error=None,
            accumulated_global_task_outputs={},
            overall_status="starting"
        )

        if stream_callback:
            stream_callback({"event_type": "agent_init_complete", "run_id": current_run_id, "agent_id": self.agent_id})

        try:
            # 1. Classify query planning strategy
            self.logger.info("Classifying query planning strategy...", event_type="QUERY_CLASSIFICATION_START", metadata={"run_id": current_run_id})
            agent_state_model.overall_status = "classifying_query"
            if stream_callback: stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
            
            strategy = await self.planner._classify_query_planning_strategy(query, conversation_context)
            self.logger.info(f"Query classification result: {strategy}", event_type="QUERY_CLASSIFICATION_END", metadata={"run_id": current_run_id, "strategy": strategy})

            if strategy == "NO_PLAN":
                agent_state_model.overall_status = "completed_no_plan"
                agent_state_model.final_response = "This query type does not require a plan. I can answer directly or it's outside my capabilities for planning."
                self.logger.info("Query classified as NO_PLAN. Returning direct response.", event_type="NO_PLAN_RESPONSE", metadata={"run_id": current_run_id})
                if stream_callback: 
                    stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
                    stream_callback({"event_type": "final_result", "run_id": current_run_id, "answer": agent_state_model.final_response, "error": None})
                return {"answer": agent_state_model.final_response, "error": None, "run_id": current_run_id, "agent_state": agent_state_model.model_dump(exclude_none=True)}

            # This will hold the AgentState as a dictionary to be passed to _execute_graph_for_state
            # and updated between phases. It includes 'accumulated_global_task_outputs'.
            # For the first phase/simple plan, accumulated_global_task_outputs starts empty.
            # agent_state_model is the Pydantic model instance tracking the overall state.
            # phase_specific_agent_state_dict is the dict representation passed to graph execution.
            phase_specific_agent_state_dict = agent_state_model.model_dump(exclude_none=True)
            
            # Ensure live instances are in the dict for graph execution, as they are excluded from serialization
            phase_specific_agent_state_dict["logger"] = self.logger
            phase_specific_agent_state_dict["config"] = self.config.model_dump(exclude_none=True) if self.config else None
            phase_specific_agent_state_dict["planner_instance"] = self.planner
            phase_specific_agent_state_dict["task_fetching_unit_instance"] = self.task_fetcher
            phase_specific_agent_state_dict["joiner_instance"] = self.joiner
            phase_specific_agent_state_dict["mcp_clients"] = self.mcp_clients
            phase_specific_agent_state_dict["dag_editor_instance"] = self.dag_editor
            
            if strategy == "COMPLEX":
                self.logger.info("Query classified as COMPLEX. Generating hierarchical plan...", event_type="HIERARCHICAL_PLAN_START", metadata={"run_id": current_run_id})
                agent_state_model.overall_status = "generating_hierarchical_plan"
                if stream_callback: stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})

                hierarchical_plan_model = await self.planner._generate_high_level_phases(query, conversation_context)
                agent_state_model.hierarchical_plan = hierarchical_plan_model
                self.logger.info(f"Generated hierarchical plan with {len(hierarchical_plan_model.phases)} phases.", event_type="HIERARCHICAL_PLAN_END", metadata={"run_id": current_run_id, "num_phases": len(hierarchical_plan_model.phases)})
                if stream_callback:
                    stream_callback({"event_type": "hierarchical_plan_generated", "plan": hierarchical_plan_model.model_dump(exclude_none=True), "run_id": current_run_id})

                for phase_index, phase_obj in enumerate(hierarchical_plan_model.phases):
                    self.logger.info(f"Starting Phase {phase_index + 1}/{len(hierarchical_plan_model.phases)}: '{phase_obj.phase_id}' - '{phase_obj.description}'", event_type="PHASE_START", metadata={"run_id": current_run_id, "phase_id": phase_obj.phase_id, "phase_index": phase_index, "total_phases": len(hierarchical_plan_model.phases)})
                    phase_obj.status = "in_progress"
                    agent_state_model.current_phase_id = phase_obj.phase_id
                    agent_state_model.current_phase_description = phase_obj.description
                    agent_state_model.overall_status = f"executing_phase_{phase_obj.phase_id}"
                    agent_state_model.replanning_cycles = 0 # Reset replanning cycles for the new phase DAG
                    # Clear task-specific states for the new phase from the *model* 
                    # The phase_specific_agent_state_dict will get a fresh DAG from planner
                    agent_state_model.task_dag = None 
                    agent_state_model.task_results = {} 
                    agent_state_model.task_statuses = {} 
                    agent_state_model.error_message = None
                    agent_state_model.execution_error = None

                    if stream_callback:
                        stream_callback({
                            "event_type": "phase_start", 
                            "phase_id": phase_obj.phase_id, 
                            "description": phase_obj.description,
                            "phase_index": phase_index + 1,
                            "total_phases": len(hierarchical_plan_model.phases),
                            "status": phase_obj.status,
                            "run_id": current_run_id
                        })
                        stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})

                    # Prepare agent_state for this specific phase. 
                    # It should inherit accumulated_global_task_outputs from the previous phase's execution.
                    # phase_specific_agent_state_dict is already carrying this from the previous loop or initial state.
                    phase_specific_agent_state_dict["original_query"] = query # Ensure original query is there
                    phase_specific_agent_state_dict["hierarchical_plan"] = agent_state_model.hierarchical_plan.model_dump(exclude_none=True) if agent_state_model.hierarchical_plan else None
                    phase_specific_agent_state_dict["current_phase_id"] = phase_obj.phase_id
                    phase_specific_agent_state_dict["current_phase_description"] = phase_obj.description
                    phase_specific_agent_state_dict["task_dag"] = None # Planner will generate this
                    phase_specific_agent_state_dict["task_results"] = {} # Reset for the current phase's tasks
                    phase_specific_agent_state_dict["task_statuses"] = {}
                    phase_specific_agent_state_dict["replanning_cycles"] = 0 # Reset for this phase
                    phase_specific_agent_state_dict["overall_status"] = agent_state_model.overall_status
                    # accumulated_global_task_outputs is already in phase_specific_agent_state_dict and accumulates

                    # Planner generates DAG for THIS phase, using accumulated_global_task_outputs from prior phases
                    try:
                        phase_dag, planner_status_message = await self.planner.generate_dag(
                            query=query, 
                            agent_state=AgentState(**phase_specific_agent_state_dict), # Pass Pydantic model for type safety
                            conversation_context=conversation_context,
                            # TODO: Pass phase context if generate_dag needs it, e.g. phase.description
                        )
                        planner_out_dict = {"status": planner_status_message, "query_type": "COMPLEX_PHASE", "error": None} # MODIFIED
                        phase_specific_agent_state_dict["planner_output"] = planner_out_dict 
                        agent_state_model.planner_output = planner_out_dict 

                        if not phase_dag or not phase_dag.tasks:
                            self.logger.error(f"Planner returned no DAG or empty DAG for phase '{phase_obj.phase_id}'.", event_type="PLANNER_EMPTY_DAG_ERROR", metadata={"run_id": current_run_id, "phase_id": phase_obj.phase_id})
                            # Update planner_output with error for this specific case
                            planner_out_dict["error"] = f"Planner failed to produce a valid task DAG for phase '{phase_obj.phase_id}'."
                            planner_out_dict["status"] = "Error: Empty DAG"
                            phase_specific_agent_state_dict["planner_output"] = planner_out_dict
                            agent_state_model.planner_output = planner_out_dict
                            raise McpError(planner_out_dict["error"])
                        
                        phase_specific_agent_state_dict["task_dag"] = phase_dag.model_dump()
                        agent_state_model.task_dag = phase_dag # Update main model as well

                    except Exception as e_plan:
                        self.logger.error(f"Error generating DAG for phase '{phase_obj.phase_id}': {e_plan}", exc_info=True, event_type="PHASE_PLANNING_ERROR", metadata={"run_id": current_run_id, "phase_id": phase_obj.phase_id})
                        phase_obj.status = "failed_planning"
                        agent_state_model.overall_status = "failed"
                        agent_state_model.error_message = f"Failed to plan for phase '{phase_obj.phase_id}': {e_plan}"
                        if stream_callback: 
                            stream_callback({"event_type": "phase_end", "phase_id": phase_obj.phase_id, "status": phase_obj.status, "error": str(e_plan), "run_id": current_run_id})
                            stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
                        break # Stop processing further phases

                    # Execute the DAG for the current phase
                    # _execute_graph_for_state expects a dict and returns a dict
                    completed_phase_state_dict = await self._execute_graph_for_state(phase_specific_agent_state_dict, stream_callback)
                    
                    # Update the main agent_state_model fields based on the completed phase execution
                    agent_state_model.task_dag = TaskDAG(**completed_phase_state_dict["task_dag"]) if completed_phase_state_dict.get("task_dag") else None
                    
                    # === BEGIN MODIFIED task_results UPDATE FOR COMPLEX ===
                    raw_phase_task_results = completed_phase_state_dict.get("task_results", {})
                    agent_state_model.task_results = {
                        k: ToolExecutionResult(**v) if isinstance(v, dict) else v
                        for k, v in raw_phase_task_results.items()
                    }
                    # === END MODIFIED task_results UPDATE FOR COMPLEX ===
                    
                    agent_state_model.task_statuses = completed_phase_state_dict.get("task_statuses", {})
                    agent_state_model.final_response = completed_phase_state_dict.get("final_response") 
                    agent_state_model.error_message = completed_phase_state_dict.get("error_message")
                    agent_state_model.replanning_cycles = completed_phase_state_dict.get("replanning_cycles", agent_state_model.replanning_cycles)
                    agent_state_model.overall_status = completed_phase_state_dict.get("overall_status", agent_state_model.overall_status) # Get status from graph
                    
                    # Accumulate outputs from this completed phase into phase_specific_agent_state_dict["accumulated_global_task_outputs"]
                    # This ensures the NEXT phase gets these outputs via the Planner.
                    # Ensure 'accumulated_global_task_outputs' exists and is a dict in phase_specific_agent_state_dict
                    if "accumulated_global_task_outputs" not in phase_specific_agent_state_dict or \
                       not isinstance(phase_specific_agent_state_dict["accumulated_global_task_outputs"], dict):
                        phase_specific_agent_state_dict["accumulated_global_task_outputs"] = {}

                    current_phase_task_results = completed_phase_state_dict.get("task_results", {})
                    if isinstance(current_phase_task_results, dict):
                        for local_task_id, tool_exec_result_data in current_phase_task_results.items():
                            output_val = None
                            # tool_exec_result_data could be a ToolExecutionResult model instance or its dict representation
                            if hasattr(tool_exec_result_data, 'output'): # Covers Pydantic model case
                                output_val = tool_exec_result_data.output
                            elif isinstance(tool_exec_result_data, dict) and 'output' in tool_exec_result_data: # Covers dict case
                                output_val = tool_exec_result_data.get('output')
                            else:
                                self.logger.warning(f"Could not extract output from task result for '{local_task_id}' in phase '{phase_obj.phase_id}'. Data: {tool_exec_result_data}", metadata={"run_id": current_run_id})
                            
                            if output_val is not None:
                                global_task_id = f"{phase_obj.phase_id}_{local_task_id}"
                                phase_specific_agent_state_dict["accumulated_global_task_outputs"][global_task_id] = output_val
                                self.logger.debug(f"Accumulated output for global task ID '{global_task_id}' from phase '{phase_obj.phase_id}'", metadata={"run_id": current_run_id})
                    
                    # Update the main Pydantic agent_state_model with the latest accumulated outputs as well
                    if isinstance(phase_specific_agent_state_dict.get("accumulated_global_task_outputs"), dict):
                        agent_state_model.accumulated_global_task_outputs = phase_specific_agent_state_dict["accumulated_global_task_outputs"].copy()
                    else: # Should not happen if initialized correctly
                        agent_state_model.accumulated_global_task_outputs = {}

                    phase_obj.status = "completed_successfully" # Default for the phase object itself
                    if agent_state_model.overall_status == "needs_clarification":
                        phase_obj.status = "requires_clarification"
                    elif agent_state_model.error_message or completed_phase_state_dict.get("execution_error"): # Check if graph execution indicated failure for the phase
                        phase_obj.status = "failed_execution"
                    
                    # phase_obj.status is for the current phase in a hierarchical plan
                    # agent_state_model.overall_status reflects the status of THIS _execute_graph_for_state call

                    if stream_callback:
                        stream_callback({
                            "event_type": "phase_end", 
                            "phase_id": phase_obj.phase_id, 
                            "status": phase_obj.status, # Status of this specific phase
                            "phase_index": phase_index + 1,
                            "error": agent_state_model.error_message or completed_phase_state_dict.get("execution_error"),
                            "final_response_for_phase": agent_state_model.final_response, 
                            "run_id": current_run_id
                        })
                    
                    if phase_obj.status != "completed_successfully":
                        self.logger.warning(f"Phase '{phase_obj.phase_id}' did not complete successfully (status: {phase_obj.status}). Halting hierarchical plan.", event_type="PHASE_HALTED", metadata={"run_id": current_run_id, "phase_id": phase_obj.phase_id, "status": phase_obj.status})
                        # The overall_status of the agent_state_model is already set from completed_phase_state_dict
                        # No need to override it here unless phase_obj.status implies a more severe overall failure
                        if stream_callback: stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
                        break # Stop processing further phases

                # After loop: Final assessment of hierarchical plan
                # This logic determines the *final overall_status* of the multi-phase plan
                if all(p.status == "completed_successfully" for p in agent_state_model.hierarchical_plan.phases):
                    agent_state_model.overall_status = "completed_successfully"
                    # If all phases are good, final_response is likely from the last phase's joiner, which is already in agent_state_model.final_response
                    self.logger.info("Hierarchical plan completed successfully.", event_type="HIERARCHICAL_PLAN_SUCCESS", metadata={"run_id": current_run_id})
                elif agent_state_model.overall_status == "needs_clarification":
                    self.logger.warning("Hierarchical plan requires clarification.", event_type="HIERARCHICAL_PLAN_CLARIFICATION", metadata={"run_id": current_run_id})
                    # final_response is already the clarification message
                elif any(p.status == "failed_execution" for p in agent_state_model.hierarchical_plan.phases):
                    agent_state_model.overall_status = "failed"
                    agent_state_model.error_message = agent_state_model.error_message or "One or more phases failed during execution."
                    self.logger.warning("Hierarchical plan failed due to phase execution failure.", event_type="HIERARCHICAL_PLAN_FAILURE", metadata={"run_id": current_run_id})
                else: # Some other non-successful completion
                    agent_state_model.overall_status = "failed_incomplete_phases"
                    agent_state_model.error_message = agent_state_model.error_message or "Hierarchical plan did not complete successfully."
                    self.logger.warning(f"Hierarchical plan finished with status: {agent_state_model.overall_status}.", event_type="HIERARCHICAL_PLAN_INCOMPLETE", metadata={"run_id": current_run_id})
                
                if not agent_state_model.final_response and agent_state_model.overall_status not in ["completed_successfully", "requires_clarification", "needs_clarification"]:
                    agent_state_model.final_response = agent_state_model.error_message or "The request could not be completed successfully."

                if stream_callback: # This callback is for the end of the COMPLEX block
                    stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})

            elif strategy == "SIMPLE":
                self.logger.debug(f"[DEBUG_STRATEGY_CHECK_INSIDE] strategy='{strategy}', type={type(strategy)}", metadata={"run_id": current_run_id})
                self.logger.info("Query classified as SIMPLE. Executing single DAG.", event_type="SIMPLE_PLAN_START", metadata={"run_id": current_run_id})
                agent_state_model.overall_status = "executing_simple_plan"
                if stream_callback: stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
                
                # Prepare dict for graph execution, including live instances
                phase_specific_agent_state_dict = agent_state_model.model_dump(exclude_none=True)
                phase_specific_agent_state_dict["logger"] = self.logger
                phase_specific_agent_state_dict["config"] = self.config.model_dump(exclude_none=True) if self.config else None
                phase_specific_agent_state_dict["planner_instance"] = self.planner
                phase_specific_agent_state_dict["task_fetching_unit_instance"] = self.task_fetcher
                phase_specific_agent_state_dict["joiner_instance"] = self.joiner
                phase_specific_agent_state_dict["mcp_clients"] = self.mcp_clients
                phase_specific_agent_state_dict["dag_editor_instance"] = self.dag_editor

                try:
                    simple_dag, planner_status_message = await self.planner.generate_dag(
                        query=query, 
                        agent_state=AgentState(**phase_specific_agent_state_dict), 
                        conversation_context=conversation_context
                    )
                    planner_out_dict = {"status": planner_status_message, "query_type": "SIMPLE", "error": None} # MODIFIED
                    phase_specific_agent_state_dict["planner_output"] = planner_out_dict
                    agent_state_model.planner_output = planner_out_dict

                    if not simple_dag or not simple_dag.tasks:
                        self.logger.error("Planner returned no DAG or empty DAG for simple query.", event_type="PLANNER_EMPTY_DAG_ERROR", metadata={"run_id": current_run_id})
                        # Update planner_output with error
                        planner_out_dict["error"] = "Planner failed to produce a valid task DAG for the query."
                        planner_out_dict["status"] = "Error: Empty DAG"
                        phase_specific_agent_state_dict["planner_output"] = planner_out_dict
                        agent_state_model.planner_output = planner_out_dict
                        raise McpError(planner_out_dict["error"])
                    
                    phase_specific_agent_state_dict["task_dag"] = simple_dag.model_dump()
                    agent_state_model.task_dag = simple_dag

                except Exception as e_plan: # Catching broader exceptions here to populate planner_output.error
                    self.logger.error(f"Error generating DAG for simple query: {e_plan}", exc_info=True, event_type="SIMPLE_PLANNING_ERROR", metadata={"run_id": current_run_id})
                    planner_out_dict = {"status": "Error during DAG generation", "query_type": "SIMPLE", "error": str(e_plan)}
                    phase_specific_agent_state_dict["planner_output"] = planner_out_dict
                    agent_state_model.planner_output = planner_out_dict
                    agent_state_model.overall_status = "failed"
                    agent_state_model.error_message = f"Failed to plan for the query: {e_plan}"
                    if stream_callback: stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
                    # Fall through to finally block for returning state

                if agent_state_model.task_dag: # Proceed only if planning was successful (no exception AND dag exists)
                    # Execute the single DAG
                    completed_run_state_dict = await self._execute_graph_for_state(phase_specific_agent_state_dict, stream_callback)

                    # Update agent_state_model from the result of the single execution run
                    agent_state_model.task_dag = TaskDAG(**completed_run_state_dict["task_dag"]) if completed_run_state_dict.get("task_dag") else None

                    # === BEGIN MODIFIED task_results UPDATE FOR SIMPLE ===
                    raw_simple_task_results = completed_run_state_dict.get("task_results", {})
                    agent_state_model.task_results = {
                        k: ToolExecutionResult(**v) if isinstance(v, dict) else v
                        for k, v in raw_simple_task_results.items()
                    }
                    # === END MODIFIED task_results UPDATE FOR SIMPLE ===
                    
                    agent_state_model.task_statuses = completed_run_state_dict.get("task_statuses", {})
                    agent_state_model.final_response = completed_run_state_dict.get("final_response")
                    agent_state_model.error_message = completed_run_state_dict.get("error_message")
                    agent_state_model.replanning_cycles = completed_run_state_dict.get("replanning_cycles", agent_state_model.replanning_cycles)
                    agent_state_model.accumulated_global_task_outputs = completed_run_state_dict.get("accumulated_global_task_outputs", {})
                    
                    # === BEGIN MODIFIED all_completed UPDATE ===
                    graph_all_completed = completed_run_state_dict.get('all_completed', False) # Default to False
                    self.logger.debug(f"[RUN_QUERY_ALL_COMPLETED_FROM_GRAPH] graph_output_dict['all_completed']: {graph_all_completed}, current agent_state_model.all_completed: {agent_state_model.all_completed}")
                    agent_state_model.all_completed = graph_all_completed
                    self.logger.debug(f"[RUN_QUERY_ALL_COMPLETED_POST_GRAPH_UPDATE] Updated agent_state_model.all_completed: {agent_state_model.all_completed}")
                    # === END MODIFIED all_completed UPDATE ===

                    # Determine overall status based on the graph execution result
                    final_graph_status = completed_run_state_dict.get("overall_status")
                    if final_graph_status:
                        agent_state_model.overall_status = final_graph_status
                    elif agent_state_model.final_response and not agent_state_model.error_message and not completed_run_state_dict.get("execution_error"):
                        agent_state_model.overall_status = "completed_successfully"
                    elif agent_state_model.error_message or completed_run_state_dict.get("execution_error"):
                         agent_state_model.overall_status = "failed"
                    else:
                        # If no explicit status, no error, but also no response, it's ambiguous
                        agent_state_model.overall_status = "unknown_completion_state" 
                        agent_state_model.error_message = agent_state_model.error_message or "The agent finished but the outcome is unclear."

                    # If failed and no specific final_response (like a clarification), use error_message
                    if agent_state_model.overall_status not in ["completed_successfully", "needs_clarification", "requires_clarification"] and not agent_state_model.final_response:
                        agent_state_model.final_response = agent_state_model.error_message or "The request could not be completed."

                    if stream_callback:
                        stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
            else: # This 'else' pairs with 'if strategy == "SIMPLE"' (handles unknown strategy)
                self.logger.error(f"Unknown planning strategy: {strategy}", event_type="UNKNOWN_STRATEGY_ERROR", metadata={"run_id": current_run_id, "strategy": strategy})
                agent_state_model.overall_status = "failed"
                agent_state_model.error_message = f"Internal error: Unknown planning strategy '{strategy}' encountered."
                if stream_callback: stream_callback({"event_type": "agent_status_update", "status": agent_state_model.overall_status, "run_id": current_run_id})
        
        except McpError as e_mcp:
            self.logger.error(f"McpError during agent run '{current_run_id}': {e_mcp}", exc_info=True, event_type="AGENT_MCP_ERROR")
            agent_state_model.error_message = agent_state_model.error_message or f"MCP operation failed: {e_mcp}"
            agent_state_model.overall_status = "failed"
        except Exception as e_agent:
            self.logger.error(f"Unhandled exception during agent run '{current_run_id}': {e_agent}", exc_info=True, event_type="AGENT_UNHANDLED_EXCEPTION")
            agent_state_model.error_message = agent_state_model.error_message or f"An unexpected error occurred: {e_agent}"
            agent_state_model.overall_status = "failed"
        finally:
            self.logger.info(f"run_query finished for Run ID: {current_run_id}. Overall status: {agent_state_model.overall_status}. Final response: {agent_state_model.final_response}", 
                             event_type="RUN_QUERY_END", 
                             metadata={
                                 "run_id": current_run_id, 
                                 "overall_status": agent_state_model.overall_status, 
                                 "final_response_length": len(agent_state_model.final_response) if agent_state_model.final_response else 0,
                                 "error_present": bool(agent_state_model.error_message)
                                 })
            
            # Update heartbeat metrics based on query completion
            if agent_state_model.overall_status == "completed_successfully":
                self._update_heartbeat_task_counters(completed_delta=1)
            elif agent_state_model.overall_status in ["failed", "needs_clarification"]:
                self._update_heartbeat_task_counters(failed_delta=1)
            
            # === BEGIN ADDED CORRECTIVE LOGIC FOR all_completed IN FINALLY BLOCK ===
            terminal_statuses = ["completed_successfully", "failed", "completed_with_clarification_failed"]
            if agent_state_model.overall_status in terminal_statuses and not agent_state_model.all_completed:
                self.logger.warning(f"[RUN_QUERY_FORCE_ALL_COMPLETED_FINALLY] Overall status is '{agent_state_model.overall_status}' but all_completed is {agent_state_model.all_completed}. Forcing to True before final serialization.", metadata={"run_id": current_run_id})
                agent_state_model.all_completed = True
            elif agent_state_model.overall_status in terminal_statuses and agent_state_model.all_completed:
                self.logger.info(f"[RUN_QUERY_ALL_COMPLETED_OK_FINALLY] Overall status is '{agent_state_model.overall_status}' and all_completed is True. No override needed.", metadata={"run_id": current_run_id})
            # === END ADDED CORRECTIVE LOGIC FOR all_completed IN FINALLY BLOCK ===

            final_agent_state_dump_tuple = agent_state_model.model_dump(exclude_none=False, warnings=True) # model_dump with warnings=True returns (dict, set_of_warnings)
            
            final_agent_state_dict: Dict[str, Any]
            if isinstance(final_agent_state_dump_tuple, tuple):
                final_agent_state_dict = final_agent_state_dump_tuple[0]
                pydantic_warnings = final_agent_state_dump_tuple[1]
                if pydantic_warnings:
                    self.logger.warning(f"[PYDANTIC_FINAL_DUMP_WARNINGS] Warnings during final agent state dump for run_id {current_run_id}: {pydantic_warnings}")
            else: # Should not happen with warnings=True, but handle defensively
                final_agent_state_dict = final_agent_state_dump_tuple # type: ignore 

            if stream_callback: 
                stream_callback({
                    "event_type": "final_result", 
                    "run_id": current_run_id, 
                    "answer": agent_state_model.final_response, # Use the model's current attribute
                    "error": agent_state_model.error_message,   # Use the model's current attribute
                    "agent_state": final_agent_state_dict 
                })
        
        return {
            "answer": agent_state_model.final_response,
            "error": agent_state_model.error_message,
            "run_id": current_run_id,
            "agent_state": final_agent_state_dict # Use the (potentially) corrected dict
        }

    async def run_direct_query(self, query_text: str, initial_state: Optional[Dict[str, Any]] = None, stream_callback: Optional[Any] = None) -> Dict[str, Any]:
        """
        Runs a query directly against the LangGraph state machine.
        This is a primary entry point for agent operations.
        Ensures MCP clients are initialized before processing the query.
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"Received direct query (request_id: {request_id}): {query_text}", event_type="DIRECT_QUERY_START")

        await self._initialize_mcp_clients()

        if initial_state is None:
            initial_state_dict = {
                "input_query": query_text,
                "conversation_context": None,
                "logger": self.logger,
                "config": self.config,
                "run_id": request_id,
                "planner_instance": self.planner,
                "task_fetching_unit_instance": self.task_fetcher,
                "joiner_instance": self.joiner,
                "mcp_clients": self.mcp_clients,
                "dag_editor_instance": self.dag_editor,
                "dag": None,
                "generated_dag": None,
                "planner_error": None,
                "task_statuses": {},
                "task_results": {},
                "error_message": None,
                "error": None,
                "final_response": None,
                "needs_replanning": False,
                "current_task_id_to_execute": None,
                "has_ready_tasks": False,
                "all_completed": False,
            }
        else:
            initial_state_dict = initial_state

        final_state = await self.run_query(initial_state_dict["input_query"], initial_state_dict["conversation_context"], initial_state_dict["run_id"])

        if stream_callback:
            stream_callback(final_state)

        return final_state

    def start_external_api(self, host="0.0.0.0", port=8000):
        """Starts the FastAPI external API server."""
        self.logger.info("Attempting to start external API...", event_type="API_INIT")
        try:
            from worker.api.main import app as fastapi_app
        except ImportError as e:
            self.logger.error(f"Failed to import FastAPI app from mochi.api.main: {e}. Ensure API components are correctly structured.", exc_info=True, event_type="API_IMPORT_ERROR")
            return

        fastapi_app.state.mochi_agent = self
        self.logger.info("Mochi Agent instance attached to FastAPI app state.", event_type="API_SETUP")

        try:
            import uvicorn
            self.logger.info(f"Starting Uvicorn server on {host}:{port}...", event_type="API_START")
            uvicorn.run(fastapi_app, host=host, port=port, log_config=None)
        except ImportError:
            self.logger.error("Uvicorn is not installed. Cannot start the external API. Please install it: pip install uvicorn[standard]", event_type="API_ERROR_NO_UVICORN")
        except Exception as e:
            self.logger.error(f"Failed to start Uvicorn server: {e}", exc_info=True, event_type="API_START_FAILURE")

    def start_cli(self, cli_args: Optional[List[str]] = None):
        """
        Starts the CLI interface.
        The MochiCLI instance will need access to this MochiAgent instance.
        """
        try:
            from worker.interfaces.dev_cli import MochiCLI
            
            cli_runner = MochiCLI(agent=self)
            self.logger.info("Starting CLI interface...", event_type="CLI_START")
            cli_runner.run_cli(cli_args if cli_args is not None else sys.argv[1:])
        except ImportError:
            self.logger.error("MochiCLI class not found or mochi.cli.interface could not be imported.", event_type="CLI_ERROR")
        except Exception as e:
            self.logger.error(f"Failed to start CLI: {e}", exc_info=True, event_type="CLI_ERROR")

    def get_agent_status(self) -> Dict[str, Any]:
        """Returns the current status and health of the agent."""
        base_status = {
            "agent_id": self.agent_id,
            "status": "running" if not self.shutdown_event.is_set() else "shutting_down",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "mcp_servers_configured": len(self.config.mcp_tool_servers) if self.config.mcp_tool_servers else 0,
                "mcp_clients_active": len([c for c in self.mcp_clients.values() if getattr(c, '_is_initialized', False)]),
                "heartbeat_enabled": self.config.agent_settings.enable_heartbeat
            },
            "conversations": {
                "active_conversations": len(self.managed_conversation_histories),
                "total_messages": sum(len(history) for history in self.managed_conversation_histories.values())
            }
        }
        
        # Add health information from heartbeat system
        if hasattr(self, 'heartbeat_manager') and self.heartbeat_manager.is_running:
            health_summary = self.heartbeat_manager.get_health_summary()
            base_status.update({"health": health_summary})
        
        return base_status

    def get_health_metrics(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed health metrics from the heartbeat system."""
        if not hasattr(self, 'heartbeat_manager') or not self.heartbeat_manager.is_running:
            return {"error": "Heartbeat system not running"}
        
        current_metrics = self.heartbeat_manager.get_current_metrics()
        metrics_history = self.heartbeat_manager.get_metrics_history(limit)
        
        return {
            "current": current_metrics.to_dict() if current_metrics else None,
            "history": [m.to_dict() for m in metrics_history],
            "summary": self.heartbeat_manager.get_health_summary()
        }

if __name__ == "__main__":
    async def execute_agent_actions():
        agent = MochiAgent()
        await agent.start()
        agent.logger.info("MochiAgent successfully started via agent.start().")

        async def local_direct_query_runner():
            if len(sys.argv) > 1 and sys.argv[1].lower() not in ["--cli", "--api"]:
                query = " ".join(sys.argv[1:])
                agent.logger.info(f"Executing direct query from __main__: {query}")
                final_state = await agent.run_query(query)
                agent.logger.info(f"Direct Query Final State from __main__: {final_state.get('final_response')}", metadata=final_state)
            elif len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].lower() not in ["--cli", "--api"]):
                print("Usage: python -m mochi.agent.main <query_text>")
                print("   or: python -m mochi.agent.main --cli [cli_options]")
                print("   or: python -m mochi.agent.main --api")
    
        if "--cli" in sys.argv:
            agent.logger.info("Starting agent in CLI mode from __main__...")
            cli_arguments = [arg for arg in sys.argv[1:] if arg != "--cli"]
            agent.start_cli(cli_args=cli_arguments if cli_arguments else None)
        elif "--api" in sys.argv:
            agent.logger.info("Starting agent in API mode from __main__...")
            agent.start_external_api()
        else:
            await local_direct_query_runner()

    asyncio.run(execute_agent_actions())