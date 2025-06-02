from typing import Optional, Dict, Any, Union
import json
import re # Added for regex-based JSON extraction
import datetime # Added for current_date
from pydantic import ValidationError # Added for Pydantic validation
from ..core.models import TaskDAG, TaskNode, StructuredError, DAGRepairActions # Added DAGRepairActions
from ..prompts.joiner_prompts import JoinerPromptBuilder
from ..core.logging import MochiLogger
from ..config import LoggingConfig, JoinerSettings

class Joiner:
    """
    The Joiner module is responsible for synthesizing the results from executed tasks
    in a DAG, determining if the overall user request has been successfully addressed,
    formulating a final response, and deciding if replanning is necessary.
    """

    def __init__(self, 
                 llm=None, 
                 settings: Optional[JoinerSettings] = None,
                 logger: Optional[MochiLogger] = None, 
                 prompt_builder: Optional[JoinerPromptBuilder] = None):
        """
        Initializes the Joiner.

        Args:
            llm: An optional pre-configured LangChain LLM instance.
                 If None, it's expected to be configured or set later.
            settings: Optional JoinerSettings instance.
            logger: An optional MochiLogger instance. If None, a default MochiLogger is created.
            prompt_builder: An optional JoinerPromptBuilder instance.
        """
        self.llm = llm
        self.settings = settings or JoinerSettings()
        if logger is not None:
            self.logger = logger
        else:
            default_logging_config = LoggingConfig()
            self.logger = MochiLogger(config=default_logging_config)
        self.prompt_builder = prompt_builder or JoinerPromptBuilder()

    def _extract_json_from_string(self, text: str) -> Optional[str]:
        """
        Extracts a JSON block from a string, accommodating markdown code fences.
        Returns the JSON string if found, otherwise None.
        """
        # Regex to find JSON within ```json ... ``` or ``` ... ```
        # It also tries to find a valid JSON object if no fences are present.
        json_match = re.search(r"```(?:json)?\\s*(\{.*?\})\\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Fallback: try to find a JSON object directly if no fences
        # This is a bit more risky but can catch cases where only JSON is returned.
        try:
            # Attempt to find the start of a JSON object or array
            json_start_match = re.search(r"\s*(\{|\[)", text)
            if json_start_match:
                substring_from_json_start = text[json_start_match.start():]
                # Try to parse it to see if it's valid JSON
                json.loads(substring_from_json_start)
                return substring_from_json_start # Return the valid JSON part
        except json.JSONDecodeError:
            pass # Not a valid JSON object from the start

        self.logger.warning(f"Could not extract a clear JSON block from text: {text[:200]}...")
        return None

    def create_joining_prompt(self, query: str, dag: Optional[TaskDAG], task_results: dict, agent_state: Optional[Dict[str, Any]] = None, current_date: Optional[str] = None, planner_output: Optional[Dict[str, Any]] = None, conversation_context: Optional[str] = None) -> str:
        """
        Creates a prompt for the LLM. 
        If planner_output indicates a NO_PLAN_CONVERSE type, it uses a direct conversational prompt.
        Otherwise, it synthesizes results and determines next steps.

        Args:
            query: The original user query.
            dag: The TaskDAG Pydantic model (source of truth for task definitions and their latest status/attempts/errors).
            task_results: A dictionary mapping task IDs to their execution results (ToolExecutionResult models often, for raw output).
            agent_state: The full agent state dictionary. Used here to potentially access task_statuses if dag is not available, though dag is preferred.
            current_date: The current date as a string (YYYY-MM-DD).
            planner_output: An optional dictionary containing planner output information, including 'query_type'.
            conversation_context: An optional string containing conversation context.

        Returns:
            A formatted prompt string for the LLM.
        """
        
        custom_instructions = self.settings.joiner_response_format_instructions
        current_date_str = current_date if current_date else datetime.date.today().isoformat()
        self.logger.debug(f"Joiner: Current date being used for prompt: {current_date_str}")

        if planner_output and planner_output.get("query_type") == "NO_PLAN_CONVERSE":
            self.logger.info("Joiner: Using direct conversational prompt for NO_PLAN_CONVERSE query type.")
            formatted_prompt = self.prompt_builder.get_direct_conversation_prompt_string(
                query=query, 
                conversation_context=conversation_context
            )
            return formatted_prompt

        self.logger.info("Joiner: Using standard joining prompt for synthesizing task results.")
        results_str_parts = []
        if dag and dag.tasks:
            for task_node in dag.tasks: # task_node is a TaskNode Pydantic model instance
                task_id = task_node.id
                tool_display_name = f"{task_node.server_id}/{task_node.tool_name}" if task_node.server_id else task_node.tool_name
                
                # Primary source of status, error, attempts is the TaskNode model itself
                status = task_node.status
                error_message_from_node = task_node.error
                attempts = task_node.execution_attempts

                # Get raw output from task_results if available
                # task_results often contains ToolExecutionResult models or similar dicts
                result_data_obj = task_results.get(task_id)
                raw_output = None
                if isinstance(result_data_obj, dict):
                    raw_output = result_data_obj.get('output') # Standard place in ToolExecutionResult
                elif hasattr(result_data_obj, 'output'): # If it's a ToolExecutionResult model
                    raw_output = result_data_obj.output
                elif result_data_obj is not None: # Fallback for other types, though less ideal
                    raw_output = result_data_obj 

                current_task_info = f"Task {task_id} (Tool: {tool_display_name}):\n"
                current_task_info += f"  Status: {status}\n"
                current_task_info += f"  Execution Attempts: {attempts}\n"
                
                if status == 'completed_success':
                    current_task_info += f"  Result: {json.dumps(raw_output) if raw_output is not None else 'No output recorded'}\n" # Serialize complex outputs
                elif status == 'completed_failure':
                    error_to_display = error_message_from_node if error_message_from_node else "Unknown error"
                    current_task_info += f"  Error: {error_to_display}\n"
                elif status == 'in_progress' or status == 'ready_to_run' or status == 'pending':
                    current_task_info += f"  Details: Task is currently {status}.\n"
                else: # Unknown or other statuses
                    current_task_info += f"  Details: {raw_output if raw_output is not None else 'N/A'}\n"
                results_str_parts.append(current_task_info)
        elif agent_state and agent_state.get('task_statuses'): # Fallback if dag is not directly passed but agent_state is
            self.logger.warning("DAG not provided to create_joining_prompt, falling back to agent_state.task_statuses. Status details will be limited.")
            # This fallback is less ideal as it won't have execution_attempts or canonical error from TaskNode.
            task_statuses_dict = agent_state.get('task_statuses', {})
            for task_id, status_val in task_statuses_dict.items():
                result_data = task_results.get(task_id, {})
                tool_name_placeholder = task_id # Cannot get server_id/tool_name without DAG
                current_task_info = f"Task {task_id} (Tool: {tool_name_placeholder}):\n"
                current_task_info += f"  Status: {status_val}\n"
                if status_val == 'completed': # older status, map to completed_success for display consistency if possible
                     current_task_info += f"  Result: {result_data.get('output') if isinstance(result_data,dict) else result_data}\n"
                elif status_val == 'failed':
                     current_task_info += f"  Error: {result_data.get('error') if isinstance(result_data,dict) else result_data}\n"
                results_str_parts.append(current_task_info)
        
        results_str = "\n".join(results_str_parts)
        if not results_str:
            results_str = "No tasks were executed or no results are available for synthesis."
        
        formatted_prompt = self.prompt_builder.get_joining_prompt_string(
            query=query, 
            results_str=results_str,
            current_date=current_date_str,
            custom_format_instructions=custom_instructions
        )
        return formatted_prompt

    def _get_task_schema_for_prompt(self, task: TaskNode, mcp_clients: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Helper to get a schema for a task for the repair prompt."""
        if not task.server_id or not task.tool_name:
            self.logger.warning(f"Task {task.id} is missing server_id or tool_name, cannot fetch schema.")
            return {"name": task.tool_name or "UnknownTool", "error": "Missing server_id or tool_name"}

        if not mcp_clients:
            self.logger.warning(f"MCP clients dictionary not provided to Joiner, cannot fetch schema for task {task.id}.")
            return {"name": task.tool_name, "error": "MCP clients not available"}

        mcp_client = mcp_clients.get(task.server_id)
        if not mcp_client:
            self.logger.warning(f"MCP client for server_id '{task.server_id}' not found, cannot fetch schema for task {task.id}.")
            return {"name": task.tool_name, "server_id": task.server_id, "error": f"MCP client for {task.server_id} not found"}

        tool_schema_data = None
        if hasattr(mcp_client, 'tool_schemas') and isinstance(mcp_client.tool_schemas, list):
            # Ensure tool_schemas is not None and is a list before trying to list its contents for the log
            tool_names_in_schemas = ["<empty_or_malformed_schema>"] * len(mcp_client.tool_schemas)
            if mcp_client.tool_schemas: # Check if list is not empty
                tool_names_in_schemas = [s.get('tool_name', '<name_missing>') if isinstance(s, dict) else '<not_a_dict>' for s in mcp_client.tool_schemas]

            self.logger.debug(f"Searching for tool '{task.tool_name}' in mcp_client.tool_schemas for server '{task.server_id}'. Schemas available by tool_name: {tool_names_in_schemas}")
            for schema in mcp_client.tool_schemas:
                if not isinstance(schema, dict): # Ensure schema is a dict before using .get
                    self.logger.warning(f"Encountered a non-dictionary item in mcp_client.tool_schemas for server '{task.server_id}': {type(schema)}")
                    continue

                schema_name_candidates = [schema.get('tool_name'), schema.get('name'), schema.get('id')]
                # self.logger.debug(f"Comparing '{task.tool_name}' with candidates: {schema_name_candidates} from schema: {schema.get('tool_name')}") # Optional: more verbose
                if task.tool_name in schema_name_candidates:
                    tool_schema_data = schema
                    self.logger.debug(f"Found schema for '{task.tool_name}' in mcp_client.tool_schemas for server '{task.server_id}'.")
                    break
            if not tool_schema_data: # Log if loop finished without finding
                 self.logger.warning(f"Tool '{task.tool_name}' not found directly in mcp_client.tool_schemas for server '{task.server_id}' after iterating {len(mcp_client.tool_schemas)} schemas.")
        else:
            has_tool_schemas_attr = hasattr(mcp_client, 'tool_schemas')
            is_list_instance = isinstance(mcp_client.tool_schemas, list) if has_tool_schemas_attr else False
            schemas_content = mcp_client.tool_schemas if has_tool_schemas_attr else "Attribute 'tool_schemas' not found"
            self.logger.warning(f"mcp_client.tool_schemas not available or not a list for server '{task.server_id}'. Has 'tool_schemas' attribute: {has_tool_schemas_attr}. Is 'tool_schemas' a list: {is_list_instance}. Content: {schemas_content}")

        if not tool_schema_data and hasattr(mcp_client, 'openapi_spec') and mcp_client.openapi_spec:
            # Fallback: Try to extract basic info if a raw OpenAPI spec is available (simplified)
            self.logger.info(f"Tool schema for '{task.tool_name}' not found directly in mcp_client.tool_schemas. Attempting to find in OpenAPI spec for {task.server_id}.")
            try:
                # This is a very simplified extraction. Real OpenAPI parsing is complex.
                # Paths are usually like /tool_name or /tools/tool_name
                path_key_candidates = [f"/{task.tool_name}", f"/tools/{task.tool_name}"]
                path_info = None
                for pkc in path_key_candidates:
                    if pkc in mcp_client.openapi_spec.get("paths", {}):
                        path_info = mcp_client.openapi_spec["paths"][pkc]
                        break
                
                if path_info:
                    # Try to find a POST or GET operation for typical tool calls
                    op_info = path_info.get("post") or path_info.get("get")
                    if op_info:
                        tool_schema_data = {
                            "name": task.tool_name,
                            "description": op_info.get("summary", op_info.get("description", "No description in OpenAPI for this path operation.")),
                            "inputs": op_info.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {"info": "Input schema structure varies based on OpenAPI spec."}),
                            "outputs": op_info.get("responses", {}).get("200", {}).get("content", {}).get("application/json", {}).get("schema", {"info": "Output schema structure varies based on OpenAPI spec."})
                        }
                        self.logger.info(f"Extracted placeholder schema for '{task.tool_name}' from OpenAPI spec of {task.server_id}.")
            except Exception as e_openapi:
                self.logger.warning(f"Error trying to parse OpenAPI spec for '{task.tool_name}' from {task.server_id}: {e_openapi}")

        if tool_schema_data:
            # We want a distilled version for the prompt, not necessarily the full verbose schema object.
            # Select key fields that are useful for the repair LLM.
            distilled_schema = {
                "name": tool_schema_data.get("tool_name", tool_schema_data.get("name", tool_schema_data.get("id", task.tool_name))),
                "description": tool_schema_data.get("description", "No description provided."),
                "input_schema": tool_schema_data.get("input_schema", {}), # Expects a JSON schema dict typically
                "output_schema": tool_schema_data.get("output_schema", {}) # Expects a JSON schema dict
            }
            self.logger.debug(f"Found and distilled schema for task {task.id} ({task.tool_name}) from server {task.server_id}: {distilled_schema}")
            return distilled_schema
        else:
            self.logger.warning(f"Could not find or extract schema for tool '{task.tool_name}' on server '{task.server_id}'. Using placeholder.")
            return {
                "name": task.tool_name,
                "server_id": task.server_id,
                "error": f"Schema not found for tool '{task.tool_name}' on server '{task.server_id}'",
                "inputs": "Unknown (schema not found)",
                "outputs": "Unknown (schema not found)"
            }

    async def _attempt_dag_repair(
        self, 
        query: str, 
        dag: TaskDAG, 
        failed_task_id: str, 
        error_message: str, 
        conversation_context: Optional[str],
        mcp_clients: Optional[Dict[str, Any]] # Added to fetch schemas
    ) -> Optional[Dict[str, Any]]:
        """
        Attempts to get targeted repair instructions from an LLM.
        Returns repair instructions dict if successful, None otherwise.
        """
        self.logger.info(f"Attempting targeted DAG repair for failed task: {failed_task_id}", event_type="JOINER_REPAIR_ATTEMPT")

        failed_task_schema_json = "{}"
        connected_tasks_schemas_list = []
        
        # Find the failed task and its connections to get schemas
        # This assumes mcp_clients is accessible to the Joiner, e.g., passed during init or to process_results
        # and then to this method.
        # For a production system, ensure mcp_clients is available and schema fetching is robust.
        
        failed_task_instance = next((t for t in dag.tasks if t.id == failed_task_id), None)
        if failed_task_instance:
            failed_task_schema_json = json.dumps(self._get_task_schema_for_prompt(failed_task_instance, mcp_clients))

            # Get schemas for directly connected tasks (dependencies and dependents)
            # This is a simplified example; actual connected tasks might need more complex graph traversal.
            dependent_task_ids = [t.id for t in dag.tasks if failed_task_id in t.dependencies]
            dependency_task_ids = failed_task_instance.dependencies
            
            connected_task_ids = set(dependent_task_ids + dependency_task_ids)

            for task_id in connected_task_ids:
                task = next((t for t in dag.tasks if t.id == task_id), None)
                if task:
                    connected_tasks_schemas_list.append(self._get_task_schema_for_prompt(task, mcp_clients))
        
        connected_tasks_schemas_json = json.dumps(connected_tasks_schemas_list)
        dag_json = dag.model_dump_json()

        repair_prompt = self.prompt_builder.get_dag_repair_prompt_string(
            query=query,
            dag_json=dag_json,
            failed_task_id=failed_task_id,
            error_message=error_message,
            failed_task_schema_json=failed_task_schema_json,
            connected_tasks_schemas_json=connected_tasks_schemas_json,
            conversation_context=conversation_context
        )

        try:
            self.logger.info(f"Joiner: Invoking LLM for DAG repair. Prompt (first 200 chars): {repair_prompt[:200]}...", event_type="JOINER_REPAIR_LLM_INVOKE")
            llm_response_object = await self.llm.ainvoke(repair_prompt) # Assuming async LLM client
            repair_content = llm_response_object.content if hasattr(llm_response_object, 'content') else str(llm_response_object)
            self.logger.info(f"Joiner: Received LLM response for DAG repair. Length: {len(repair_content)}", event_type="JOINER_REPAIR_LLM_SUCCESS")

            # Extract JSON from the response string
            json_to_parse = self._extract_json_from_string(repair_content)
            if not json_to_parse:
                self.logger.error(f"Joiner: Could not extract JSON from DAG repair LLM response. Raw response: {repair_content}", event_type="JOINER_REPAIR_JSON_EXTRACTION_FAILED")
                return None

            # Use Pydantic model for parsing and validation
            validated_repair_actions = DAGRepairActions.model_validate_json(json_to_parse)

            # Check for NO_REPAIR_POSSIBLE action explicitly, even though it's part of the model
            # This is to maintain similar logic flow as before for logging and early exit.
            if validated_repair_actions.repair_actions: # Ensure there's at least one action
                first_action = validated_repair_actions.repair_actions[0]
                if first_action.action_type == "NO_REPAIR_POSSIBLE":
                    self.logger.info("DAG Repair LLM indicated no repair possible. Falling back to full replan.", event_type="JOINER_REPAIR_NO_REPAIR")
                    return None
            
            self.logger.info(f"Successfully parsed and validated DAG repair actions: {validated_repair_actions.model_dump()}", event_type="JOINER_REPAIR_ACTIONS_PARSED_VALIDATED")
            return validated_repair_actions.model_dump() # Return as dict as previously expected

        except ValidationError as pydantic_e:
            self.logger.error(f"Joiner: Pydantic validation failed for DAG repair LLM response. Error: {pydantic_e}. Extracted text: {json_to_parse}", exc_info=True, event_type="JOINER_REPAIR_PYDANTIC_VALIDATION_ERROR")
            return None
        except json.JSONDecodeError as json_e:
            # This might still occur if _extract_json_from_string returns something that passes initial json.loads but isn't what Pydantic expects, 
            # or if json_to_parse was None and we somehow bypassed the check (though unlikely with current logic).
            # However, Pydantic's model_validate_json should ideally catch most malformed JSON issues if json_to_parse is not None.
            self.logger.error(f"Joiner: Failed to parse extracted JSON (pre-Pydantic) from DAG repair LLM response. Error: {json_e}. Extracted text was: {json_to_parse if json_to_parse else 'None'}", exc_info=True, event_type="JOINER_REPAIR_JSON_ERROR_PRE_PYDANTIC")
            return None
        except Exception as e:
            self.logger.error(f"Joiner: DAG repair LLM invocation failed. Error: {e}", exc_info=True, event_type="JOINER_REPAIR_LLM_FAILURE")
            return None

    async def process_results(self, query: str, dag: Optional[TaskDAG], task_results: Optional[dict], task_status: Optional[dict], planner_output: Optional[Dict[str, Any]] = None, conversation_context: Optional[str] = None, mcp_clients: Optional[Dict[str, Any]] = None) -> dict:
        """
        Processes the task results by invoking the LLM.
        If it's a NO_PLAN_CONVERSE query type, it gets a direct conversational response.
        Otherwise, it uses a synthesized prompt and parses the LLM's response 
        to determine if replanning is needed and to extract the final response or explanation.

        Args:
            query: The original user query.
            dag: The TaskDAG Pydantic model (Optional, can be None for NO_PLAN_CONVERSE).
            task_results: A dictionary mapping task IDs to their results (Optional).
            task_status: A dictionary mapping task IDs to their execution status (Optional).
            planner_output: An optional dictionary containing planner output information, including 'query_type'.
            conversation_context: An optional string containing conversation context.
            mcp_clients: An optional dictionary mapping server IDs to McpClient instances for schema fetching.

        Returns:
            A dictionary containing:
                - 'needs_replanning': boolean
                - 'response': string, the textual response/explanation from the LLM.
                - 'explanation_for_replan': string, the explanation for replanning if needed.
                - 'error': Optional string if an error occurred during joiner processing.
        
        Raises:
            ValueError: If self.llm is not configured/initialized.
            TypeError: If input arguments are not of the expected type (for mandatory fields).
        """
        if not isinstance(query, str):
            raise TypeError("'query' must be a string.")
        if dag is not None and not isinstance(dag, TaskDAG):
            self.logger.error(f"Joiner process_results: 'dag' must be a TaskDAG model if provided. Got: {type(dag)}")
            raise TypeError("'dag' must be a TaskDAG Pydantic model if provided.")
        if task_results is not None and not isinstance(task_results, dict):
            raise TypeError("'task_results' must be a dict if provided.")
        if task_status is not None and not isinstance(task_status, dict):
            raise TypeError("'task_status' must be a dict if provided.")

        self.logger.info(f"Joiner starting to process results for query: '{query[:50]}...'")
        query_type_from_planner = planner_output.get("query_type") if planner_output else None
        self.logger.info(f"Joiner: Received query_type from planner_output: {query_type_from_planner}")

        if not self.llm:
            self.logger.error("Joiner: LLM for Joiner is not configured.")
            raise ValueError("LLM for Joiner is not configured. Cannot process results.")

        # Get current date
        today_iso = datetime.date.today().isoformat()

        prompt_string = self.create_joining_prompt(
            query=query, 
            dag=dag, 
            task_results=task_results or {}, 
            agent_state=None, 
            current_date=today_iso,
            planner_output=planner_output, 
            conversation_context=conversation_context
        )
        
        content = ""
        try:
            self.logger.info(f"Joiner: Invoking LLM for result synthesis. Prompt (first 200 chars): {prompt_string[:200]}...")
            llm_response_object = await self.llm.ainvoke(prompt_string)
            content = llm_response_object.content if hasattr(llm_response_object, 'content') else str(llm_response_object)
            self.logger.info(f"Joiner: Received LLM response. Length: {len(content)}")
        except Exception as e:
            self.logger.error(f"Joiner: LLM invocation failed. Error: {e}", exc_info=True)
            return {
                "error": "LLM_INVOCATION_FAILURE",
                "details": str(e),
                "needs_replanning": True,
                "response": "The language model call failed. Replanning is recommended.",
            }

        if query_type_from_planner == "NO_PLAN_CONVERSE":
            self.logger.info("Joiner: NO_PLAN_CONVERSE type, using LLM output directly as response, no replanning.")
            return {
                "needs_replanning": False,
                "response": content.strip(),
                "error": None
            }

        initial_needs_replanning = False
        response_for_user_or_explanation = content.strip()
        content_lines = content.strip().splitlines()

        if not content_lines or not content_lines[0].strip():
            self.logger.warning("Joiner: LLM response for planned query was empty or first line was all whitespace. Defaulting to replan.")
            initial_needs_replanning = True
            response_for_user_or_explanation = f"LLM output format error (empty or malformed first line). Original output (first 200 chars): '{content.strip()[:200]}...'. Replanning is recommended."
        else:
            first_line_upper = content_lines[0].upper().strip()
            if first_line_upper == "REPLAN: YES":
                initial_needs_replanning = True
                response_for_user_or_explanation = "\n".join(content_lines[1:]).strip()
                self.logger.info("Joiner: REPLAN: YES directive found.")
            elif first_line_upper == "REPLAN: NO":
                initial_needs_replanning = False
                response_for_user_or_explanation = "\n".join(content_lines[1:]).strip()
                self.logger.info("Joiner: REPLAN: NO directive found.")
            else:
                self.logger.warning(f"Joiner: REPLAN directive missing or malformed in LLM output for planned query. First line: '{content_lines[0][:100]}'. Defaulting to replan.")
                initial_needs_replanning = True
                response_for_user_or_explanation = f"LLM output format error (REPLAN directive missing/malformed). Original output (first 200 chars): '{content.strip()[:200]}...'. Replanning is recommended."
        
        # --- START TARGETED REPAIR LOGIC ---
        if initial_needs_replanning and dag and task_status:
            repairable_failure_found = False
            failed_task_id_for_repair = None
            error_object_for_repair: Optional[Union[str, StructuredError]] = None

            for task_id, status_val in task_status.items():
                if status_val == 'failed': 
                    task_result_info = (task_results or {}).get(task_id, {})
                    error_detail = task_result_info.get('error') if isinstance(task_result_info, dict) else str(task_result_info)
                    
                    current_error_is_repairable = False
                    error_message_for_log = "Unknown error"

                    if isinstance(error_detail, StructuredError):
                        self.logger.info(f"Task {task_id} failed with StructuredError. Type: {error_detail.error_type}, Repairable Hint: {error_detail.is_repairable}", event_type="JOINER_STRUCTURED_ERROR_DETECTED")
                        error_message_for_log = f"Type: {error_detail.error_type}, Msg: {error_detail.message}"
                        if error_detail.is_repairable is not None:
                            current_error_is_repairable = error_detail.is_repairable
                        else:
                            repairable_error_types = ["ToolReportedError", "McpServerError", "InputResolutionError"]
                            if error_detail.error_type in repairable_error_types:
                                current_error_is_repairable = True
                    elif isinstance(error_detail, str):
                        error_message_for_log = error_detail
                        keywords_for_repair = ["input", "keyerror", "field not found", "missing parameter", "valueerror", "toolreportederror", "mcpservererror"]
                        if any(keyword in error_detail.lower() for keyword in keywords_for_repair):
                            current_error_is_repairable = True
                    else:
                        error_message_for_log = str(error_detail)
                        current_error_is_repairable = False

                    if current_error_is_repairable:
                        self.logger.info(f"Identified potentially repairable failure for task {task_id}: {error_message_for_log}", event_type="JOINER_REPAIRABLE_FAILURE_DETECTED")
                        repairable_failure_found = True
                        failed_task_id_for_repair = task_id
                        error_object_for_repair = error_detail
                        break

            if repairable_failure_found and failed_task_id_for_repair and error_object_for_repair:
                simple_error_message_for_prompt = error_object_for_repair.message if isinstance(error_object_for_repair, StructuredError) else str(error_object_for_repair)
                
                repair_instructions = await self._attempt_dag_repair(
                    query=query,
                    dag=dag,
                    failed_task_id=failed_task_id_for_repair,
                    error_message=simple_error_message_for_prompt,
                    conversation_context=conversation_context,
                    mcp_clients=mcp_clients
                )
                if repair_instructions:
                    self.logger.info(f"Targeted DAG repair suggested for task {failed_task_id_for_repair}. Overriding full replan.", event_type="JOINER_REPAIR_SUCCESS")
                    return {
                        "needs_replanning": False,
                        "repair_instructions_available": True,
                        "dag_repair_instructions": repair_instructions,
                        "response": f"Attempting targeted repair for task {failed_task_id_for_repair}. Explanation from joiner: {response_for_user_or_explanation}",
                        "error": None
                    }
                else:
                    self.logger.info(f"Targeted DAG repair attempt failed or not suggested for {failed_task_id_for_repair}. Proceeding with full replan as per initial assessment.", event_type="JOINER_REPAIR_FALLBACK_TO_FULL_REPLAN")
            # If no repairable failure or repair attempt fails, proceed with initial_needs_replanning decision
        # --- END TARGETED REPAIR LOGIC ---
        
        extracted_block = ""
        if "USER_RESPONSE_START" in response_for_user_or_explanation:
            try:
                start_marker = "USER_RESPONSE_START"
                end_marker = "USER_RESPONSE_END"
                start_index = response_for_user_or_explanation.find(start_marker) + len(start_marker)
                end_index = response_for_user_or_explanation.find(end_marker, start_index)
                if end_index == -1: end_index = len(response_for_user_or_explanation)
                extracted_block = response_for_user_or_explanation[start_index:end_index].strip()
                self.logger.info(f"Joiner: Extracted USER_RESPONSE block. Length: {len(extracted_block)}")
            except Exception as e_parse_user:
                self.logger.warning(f"Joiner: Error parsing USER_RESPONSE block: {e_parse_user}. Using full response after REPLAN directive.")
                extracted_block = response_for_user_or_explanation
        elif "EXPLANATION_START" in response_for_user_or_explanation:
            try:
                start_marker = "EXPLANATION_START"
                end_marker = "EXPLANATION_END"
                start_index = response_for_user_or_explanation.find(start_marker) + len(start_marker)
                end_index = response_for_user_or_explanation.find(end_marker, start_index)
                if end_index == -1: end_index = len(response_for_user_or_explanation)
                extracted_block = response_for_user_or_explanation[start_index:end_index].strip()
                self.logger.info(f"Joiner: Extracted EXPLANATION block. Length: {len(extracted_block)}")
                self.logger.debug(f"Joiner: Explanation for replan: {extracted_block}")
            except Exception as e_parse_exp:
                self.logger.warning(f"Joiner: Error parsing EXPLANATION block: {e_parse_exp}. Using full response after REPLAN directive.")
                extracted_block = response_for_user_or_explanation
        else:
            self.logger.warning("Joiner: Neither USER_RESPONSE nor EXPLANATION markers found in planned query response. Using content after REPLAN directive as is.")
            extracted_block = response_for_user_or_explanation

        self.logger.info(f"Joiner: Replanning determined as {'needed' if initial_needs_replanning else 'not needed'} for planned query.")
        final_user_response = extracted_block if extracted_block else response_for_user_or_explanation
        final_explanation = response_for_user_or_explanation if initial_needs_replanning else None

        return {
            "needs_replanning": initial_needs_replanning,
            "response": final_user_response,
            "explanation_for_replan": final_explanation,
            "error": None
        }

    def generate_replan_context(self, query: str, dag: Optional[TaskDAG], task_results: Optional[dict], task_status: Optional[dict]) -> dict:
        """
        Generates a context dictionary for the Planner module if replanning is needed.
        This context includes the original query, lists of successful and failed tasks
        with their outcomes, and a dictionary of available results from completed tasks.

        Args:
            query: The original user query.
            dag: The TaskDAG Pydantic model representing the graph (Optional).
            task_results: A dictionary mapping task IDs to their results (Optional).
            task_status: A dictionary mapping task IDs to their execution status (Optional).

        Returns:
            A dictionary structured for providing context to a replanning phase.
        Raises:
            TypeError: If input arguments are not of the expected type (for mandatory fields).
        """
        if not isinstance(query, str):
            raise TypeError("'query' must be a string.")
        if dag is not None and not isinstance(dag, TaskDAG):
            self.logger.error(f"Joiner generate_replan_context: 'dag' must be a TaskDAG model if provided. Got: {type(dag)}")
            raise TypeError("'dag' must be a TaskDAG Pydantic model if provided.")
        if task_results is not None and not isinstance(task_results, dict):
            raise TypeError("'task_results' must be a dict if provided.")
        if task_status is not None and not isinstance(task_status, dict):
            raise TypeError("'task_status' must be a dict if provided.")

        self.logger.info(f"Joiner generating replan context for query: '{query[:50]}...'")

        successful_tasks = []
        failed_tasks = []
        available_results = {}
        
        current_task_results = task_results or {}
        current_task_status = task_status or {}

        if dag and dag.tasks:
            for task_node in dag.tasks:
                task_id = task_node.id
                tool_display_name = f"{task_node.server_id}/{task_node.tool_name}"
                
                status = current_task_status.get(task_id)
                result_payload = current_task_results.get(task_id)
                
                actual_result = None
                error_detail = None

                if isinstance(result_payload, dict):
                    if status == 'completed':
                        actual_result = result_payload.get('output', result_payload) 
                    else:
                        error_detail = result_payload.get('error', str(result_payload))
                elif hasattr(result_payload, 'status') and hasattr(result_payload, 'output') and hasattr(result_payload, 'error'):
                    if result_payload.status == 'success':
                        actual_result = result_payload.output
                    else:
                        error_detail = result_payload.error
                else:
                    if status == 'completed':
                        actual_result = result_payload
                    else:
                        error_detail = str(result_payload)

                if status == 'completed':
                    successful_tasks.append({
                        "id": task_id,
                        "tool_name": tool_display_name,
                        "result_summary": str(actual_result)[:200] + "..." if len(str(actual_result)) > 200 else str(actual_result)
                    })
                    available_results[task_id] = actual_result 
                elif status == 'failed':
                    failed_tasks.append({
                        "id": task_id,
                        "tool_name": tool_display_name,
                        "error": str(error_detail)
                    })
        else:
            self.logger.info("Joiner: No DAG or no tasks in DAG for generating replan context.")

        return {
            "original_query": query,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "available_results": available_results 
        }
