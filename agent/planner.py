import json
from typing import List, Dict, Optional, Any, Set, Union, Tuple
import logging
import re
import asyncio
import os
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.exceptions import OutputParserException
import torch
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


from pydantic import ValidationError
from .mcp_client import McpClient, McpError
from ..prompts.planner_prompts import PlannerPromptBuilder, SYSTEM_MESSAGE_CONTENT, REPLANNING_SYSTEM_PROMPT_TEMPLATE
from ..core.models import TaskNode, TaskDAG, ToolSchema, ServerToolSchemaGroup, HierarchicalPlan, Phase, ToolParameter, AgentState
from ..config import get_settings
from worker.core.logging import MochiLogger
from ..config.models import DEFAULT_SEMANTIC_STOP_WORDS, PlannerSettings

logger = logging.getLogger(__name__)

class Planner:
    '''Planner module to generate task DAGs from user queries.'''

    def __init__(
        self,
        llm: BaseLanguageModel,
        mcp_clients: Dict[str, McpClient],
        settings: PlannerSettings,
        prompt_builder: Optional[PlannerPromptBuilder] = None,
        logger_instance: Optional[MochiLogger] = None,
    ):
        '''Initializes the Planner.'''
        self.logger: Union[MochiLogger, logging.Logger] = logger_instance if logger_instance else logging.getLogger(f"mochi.{self.__class__.__name__}")

        if not isinstance(llm, BaseLanguageModel):
            raise TypeError('llm must be an instance of BaseLanguageModel')
        if not isinstance(mcp_clients, dict):
            raise TypeError('mcp_clients must be a dictionary.')
        if not all(isinstance(client, McpClient) for client in mcp_clients.values()):
            raise TypeError('All values in mcp_clients dictionary must be instances of McpClient.')
        if not mcp_clients:
            self.logger.warning("Planner initialized with no MCP clients. Tool-related functionality will be limited.", event_type="PLANNER_INIT_WARN")
        if not isinstance(settings, PlannerSettings):
             raise TypeError('settings must be an instance of PlannerSettings')

        self.llm = llm
        self.mcp_clients = mcp_clients
        self.prompt_builder = prompt_builder or PlannerPromptBuilder()
        self.settings = settings
        self.stop_words: Set[str] = set(self.settings.semantic_stop_words) if self.settings.semantic_stop_words else DEFAULT_SEMANTIC_STOP_WORDS
        
        # --- Initialize Embedding Model --- 
        self.embedding_model = None
        model_to_load = self.settings.embedding_model_name
        
        try:
            self.logger.info(f"Loading sentence transformer model: {model_to_load}")
            self.embedding_model = SentenceTransformer(model_to_load)
            self.logger.info(f"Sentence transformer model {model_to_load} loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model '{model_to_load}': {e}", exc_info=True)
            self.embedding_model = None

    def _rewrite_input_references_recursive(self, data: Any, id_mapping: Dict[str, str]) -> Any:
        """
        Recursively traverses an input structure (dict, list, or literal) and rewrites
        $result.LOCAL_ID.path references to $result.GLOBAL_ID.path if LOCAL_ID is in id_mapping.
        """
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                new_dict[key] = self._rewrite_input_references_recursive(value, id_mapping)
            return new_dict
        elif isinstance(data, list):
            new_list = []
            for item in data:
                new_list.append(self._rewrite_input_references_recursive(item, id_mapping))
            return new_list
        elif isinstance(data, str) and data.startswith("$result."):
            parsed_ref = self._parse_result_reference(data)
            if parsed_ref:
                producer_task_id, path_segments = parsed_ref
                if producer_task_id in id_mapping:
                    new_producer_id = id_mapping[producer_task_id]
                    new_ref_str = f"$result.{new_producer_id}"
                    if path_segments:
                        new_ref_str += "." + ".".join(path_segments)
                    self.logger.debug(f"Rewritten $result reference from '{data}' to '{new_ref_str}'")
                    return new_ref_str
            return data # Return original if not an intra-phase reference or parsing failed
        else:
            return data

    def _parse_result_reference(self, reference_str: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parses a $result reference string into producer task ID and path segments.
        Example: "$result.task_A.data.items[0].name" -> ("task_A", ["data", "items", "0", "name"])
        Returns None if the string is not a valid $result reference.
        """
        if not reference_str.startswith("$result."):
            return None
        
        parts = reference_str[len("$result."):].split('.')
        if not parts or not parts[0]: # Must have at least a task_id
            return None
            
        producer_task_id = parts[0]
        path_segments = []
        
        for part in parts[1:]:
            # Handle array indices like "[0]" by splitting them out
            match = re.match(r"(.+?)\[(\d+)\]$", part)
            if match:
                path_segments.append(match.group(1)) # The part before index
                path_segments.append(match.group(2)) # The index itself
            else:
                path_segments.append(part)
                
        return producer_task_id, path_segments

    def _get_schema_type_from_path(self, schema: Dict[str, Any], path_segments: List[str]) -> Optional[str]:
        """
        Traverses a JSON schema using a list of path segments and returns the JSON schema type 
        of the field at the given path.
        Handles 'properties' for objects and 'items' (for type of array elements).
        Array indices in path_segments are used to navigate into array item schemas.
        """
        current_schema = schema
        for i, segment in enumerate(path_segments):
            segment_type = current_schema.get("type")
            
            if segment_type == "object":
                if "properties" in current_schema and segment in current_schema["properties"]:
                    current_schema = current_schema["properties"][segment]
                else:
                    # Property not found in schema
                    self.logger.debug(f"_get_schema_type_from_path: Segment '{segment}' not found in properties of object schema: {current_schema}")
                    return None
            elif segment_type == "array":
                # If the next segment is an array index, we expect 'items' to define the schema for elements.
                # The segment itself should be an index if we're trying to access an array element.
                # However, we are interested in the type of items IN the array, which 'items' defines.
                if "items" in current_schema:
                    current_schema = current_schema["items"]
                    # After processing 'items', the current_schema is now the schema of an array element.
                    # If the *current* path segment (segment) is an index, we've effectively "used" it
                    # by moving into the 'items' schema. We continue to the next segment for further path traversal
                    # *within* the item's structure if the item itself is an object or array.
                    # If the current segment is NOT an index (e.g., 'length' on an array), then this path is invalid
                    # for typed data access according to typical JSON schema data patterns.
                    # We assume that if we are pathing *through* an array, the next segment in the original path
                    # would apply to the elements of the array.
                    
                    # Let's check if the current segment is meant to be an index.
                    # If the segment is a digit, it implies we are trying to access a specific element.
                    # The type we care about is the type of *elements* in the array.
                    if segment.isdigit(): 
                        # We've already moved into the 'items' schema.
                        # The next segment in path_segments (if any) will apply to this item schema.
                        pass
                    else:
                        # Accessing a property *of the array itself* (e.g. a hypothetical '.length' if it were in schema)
                        # This is not standard for getting type of array *contents*.
                        # For now, if pathing continues beyond an array without an index,
                        # it implies the 'items' schema should be an object with that property.
                        # This logic branch will be hit if `segment` is not a digit.
                        # Example: $result.task_A.myArray.someProperty (if myArray is array, someProperty is on items)
                        # This means `current_schema` is now the schema for items in `myArray`.
                        # The loop will continue, and on the next iteration, `segment` will be `someProperty`.
                        # If `current_schema` (the item schema) is an object, it will try to find `someProperty`.
                        pass


                else:
                    # 'items' not defined for array schema
                    self.logger.debug(f"_get_schema_type_from_path: 'items' not defined for array schema: {current_schema}")
                    return None
            else:
                # Current segment is not an object or array, but path continues.
                # This means the path is trying to go deeper than the schema allows.
                if i < len(path_segments): # If there are more segments after this non-dict/list type
                    self.logger.debug(f"_get_schema_type_from_path: Path goes deeper than schema allows at segment '{segment}'. Current schema type: {segment_type}")
                    return None
                # If this is the last segment, its type is what we return below.

        # After iterating through all path segments, current_schema is the schema of the final segment.
        final_type = current_schema.get("type")
        if isinstance(final_type, str):
            return final_type
        elif isinstance(final_type, list): # Type can be a list e.g. ["string", "null"]
            # For simplicity, return the first non-"null" type if multiple are allowed.
            # A more sophisticated type comparison might be needed for strict compatibility.
            for t in final_type:
                if t != "null":
                    return t
            return "null" # If only "null" or all are "null"
        
        self.logger.debug(f"_get_schema_type_from_path: Could not determine type for schema {schema} at path {path_segments}. Final schema part: {current_schema}")
        return None

    async def get_tool_schemas(self) -> List[ServerToolSchemaGroup]:
        '''Retrieves and formats MCP Tool Server schemas as Pydantic models from all configured clients.'''
        all_raw_tool_defs: List[Dict[str, Any]] = []
        
        if not self.mcp_clients:
            self.logger.info("No MCP clients configured in Planner. Cannot fetch tool schemas.")
            return []

        for server_id_key, client in self.mcp_clients.items():
            try:
                if not client._is_initialized:
                    self.logger.warning(f'McpClient for server {server_id_key} not initialized. Attempting to initialize.')
                    if hasattr(client, 'initialize') and asyncio.iscoroutinefunction(client.initialize):
                        await client.initialize()
                    else:
                        self.logger.warning(f"McpClient for server {server_id_key} does not have an async initialize method or it's not callable.", event_type="MCP_CLIENT_WARN")

                self.logger.info(f"Fetching tools from MCP server: {server_id_key} via client: {client.server_address}")
                tools_from_server = await client.list_tools()
                for tool_def in tools_from_server:
                    if 'server_id' not in tool_def:
                        self.logger.debug(f"Tool definition from {server_id_key} missing 'server_id', adding it. Tool: {tool_def.get('tool_name', 'Unknown')}")
                        tool_def['server_id'] = server_id_key
                    elif tool_def['server_id'] != server_id_key:
                        self.logger.warning(f"Mismatched server_id in tool definition from {server_id_key}. Expected {server_id_key}, got {tool_def['server_id']}. Using expected: {server_id_key}", event_type="MCP_DATA_MISMATCH")
                        tool_def['server_id'] = server_id_key

                all_raw_tool_defs.extend(tools_from_server)
                
            except McpError as e:
                self.logger.error(f'MCPError while fetching tool schemas from server {server_id_key} ({client.server_address}): {e}', exc_info=True)
                continue 
            except Exception as e:
                self.logger.error(f'Unexpected error fetching tool schemas from server {server_id_key} ({client.server_address}): {e}', exc_info=True)
                continue
        
        if not all_raw_tool_defs:
            self.logger.warning("No tool definitions could be fetched from any MCP client.", event_type="PLANNER_TOOLS_EMPTY")
            return []

        grouped_schemas_dict: Dict[str, ServerToolSchemaGroup] = {}
        for tool_def in all_raw_tool_defs:
            server_id = tool_def.get('server_id')
            if not server_id:
                self.logger.warning(f'Tool definition missing server_id after aggregation: {tool_def.get("tool_name", "Unknown tool")}')
                continue

            try:
                tool_instance = ToolSchema.model_validate(tool_def)
            except ValidationError as ve:
                self.logger.warning(f"Pydantic validation error for tool {tool_def.get('tool_name')} on server {server_id}: {ve}")
                continue

            if server_id not in grouped_schemas_dict:
                server_desc_from_tool = tool_def.get('server_description', f'Tools available on server: {server_id}')
                grouped_schemas_dict[server_id] = ServerToolSchemaGroup(
                    server_id=server_id,
                    description=server_desc_from_tool,
                    tools=[]
                )
            
            grouped_schemas_dict[server_id].tools.append(tool_instance)
        
        return list(grouped_schemas_dict.values())

    def _recommend_tools(
        self,
        query: str,
        all_server_schemas: List[ServerToolSchemaGroup],
    ) -> List[ServerToolSchemaGroup]:
        '''Recommends a list of server schemas based on semantic similarity to the query.'''
        top_n = self.settings.tool_recommendation_top_n
        min_score_threshold = self.settings.semantic_score_min_threshold

        if not all_server_schemas:
            self.logger.info("_recommend_tools: No server schemas provided.")
            return []

        if not util:
            self.logger.warning(f"_recommend_tools: Sentence transformer model specified in config ('{self.settings.embedding_model_name}') not loaded or library unavailable. Cannot perform semantic recommendation. Falling back to returning ALL tools.")
            return all_server_schemas

        try:
            self.logger.info(f"_recommend_tools: Generating embedding for query: '{query[:100]}...'")
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        except Exception as e:
            self.logger.error(f"_recommend_tools: Failed to encode query '{query[:100]}...': {e}", exc_info=True)
            self.logger.warning("_recommend_tools: Query encoding failed. Falling back to returning ALL tools.")
            return all_server_schemas

        scored_servers = []
        for server_schema in all_server_schemas:
            max_tool_score_for_server = 0.0
            server_description_score = 0.0
            tool_texts_to_embed = []
            tool_indices = [] 
            
            for idx, tool in enumerate(server_schema.tools):
                tool_text = f"{tool.tool_name}. {tool.description or ''}".strip()
                if tool_text:
                    tool_texts_to_embed.append(tool_text)
                    tool_indices.append(idx)
            
            if not tool_texts_to_embed:
                self.logger.debug(f"_recommend_tools: Server '{server_schema.server_id}' has no tools with text to embed.")
                pass 
            else:
                try:
                    self.logger.debug(f"_recommend_tools: Embedding {len(tool_texts_to_embed)} tool texts for server '{server_schema.server_id}'.")
                    tool_embeddings = self.embedding_model.encode(tool_texts_to_embed, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(query_embedding, tool_embeddings)[0]
                    if cosine_scores.numel() > 0:
                         max_tool_score_for_server = torch.max(cosine_scores).item()
                    else:
                         max_tool_score_for_server = 0.0 
                except Exception as e:
                    self.logger.error(f"_recommend_tools: Failed to encode or score tools for server '{server_schema.server_id}': {e}", exc_info=True)
                    max_tool_score_for_server = 0.0
            
            if self.settings.semantic_search_include_server_description and server_schema.description:
                try:
                    self.logger.debug(f"_recommend_tools: Embedding server description for '{server_schema.server_id}'")
                    server_desc_embedding = self.embedding_model.encode(server_schema.description, convert_to_tensor=True)
                    server_desc_sim = util.cos_sim(query_embedding, server_desc_embedding)[0]
                    if server_desc_sim.numel() > 0:
                         server_description_score = server_desc_sim.item()
                except Exception as e:
                    self.logger.error(f"_recommend_tools: Failed to encode or score server description for '{server_schema.server_id}': {e}", exc_info=True)
                    server_description_score = 0.0 
            
            final_server_score = max(max_tool_score_for_server, server_description_score)
            
            if final_server_score >= min_score_threshold:
                scored_servers.append((server_schema, final_server_score))
            else:
                self.logger.debug(f"_recommend_tools: Server '{server_schema.server_id}' score {final_server_score:.4f} below threshold {min_score_threshold}. Skipping.")

        scored_servers.sort(key=lambda x: x[1], reverse=True)
        
        effective_top_n = max(top_n, 1) if all_server_schemas else 0
        recommended_schemas = [server_schema for server_schema, score in scored_servers[:effective_top_n]]
        
        if not recommended_schemas and all_server_schemas:
            self.logger.warning(f"_recommend_tools: Semantic tool recommendation yielded no results above threshold {min_score_threshold} (or all scores were low/zero) for query '{query}'. Falling back to providing ALL available tools.")
            return all_server_schemas
        
        self.logger.info(f"_recommend_tools: Recommended {len(recommended_schemas)} server(s) based on semantic similarity (threshold={min_score_threshold}, top_n={top_n}).")
        return recommended_schemas

    def _parse_llm_output_to_dag_model(self, llm_output: str) -> TaskDAG:
        '''Parses LLM string output to a TaskDAG Pydantic model.
        Raises ValueError if parsing or Pydantic validation fails.
        '''
        self.logger.debug(f'Raw LLM output for DAG generation:\n{llm_output}')
        
        json_str = llm_output
        markdown_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL | re.IGNORECASE)
        if markdown_match:
            json_str = markdown_match.group(1)
        else:
            first_brace = llm_output.find('{')
            last_brace = llm_output.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = llm_output[first_brace : last_brace + 1]

        json_str = json_str.strip()
        self.logger.debug(f'Attempting to parse JSON into TaskDAG from (repr): {repr(json_str)}')
        
        try:
            dag_model = TaskDAG.model_validate_json(json_str)
            self.logger.info('Successfully parsed and validated LLM output to TaskDAG model.')
            return dag_model
        except ValidationError as ve:
            self.logger.error(f'Pydantic validation error parsing LLM output to TaskDAG: {ve}', exc_info=True)
            self.logger.error(f'Problematic JSON string part (repr): {repr(json_str[:500])}')
            raise ValueError(f'LLM output failed Pydantic validation for TaskDAG. Error: {ve}. Cleaned JSON (repr): {repr(json_str[:500])}...') from ve
        except json.JSONDecodeError as jde:
            self.logger.error(f'Failed to decode JSON from LLM output (json.JSONDecodeError): {jde}', exc_info=True)
            self.logger.error(f'Problematic JSON string part (repr): {repr(json_str[:500])}')
            raise ValueError(f'LLM output is not valid JSON. Error: {jde}. Cleaned JSON (repr): {repr(json_str[:500])}...') from jde

    def _validate_dag_logic(
        self, 
        dag: TaskDAG,
        is_phase_specific_validation: bool = False
    ) -> None:
        """Validates the logical consistency of the generated TaskDAG,
        including $result reference schema cross-checking.
        """
        if not dag.tasks:
            self.logger.info("DAG has no tasks, validation skipped.")
            return

        task_ids_in_current_dag = {task.id for task in dag.tasks}
        task_id_to_node_map = {task.id: task for task in dag.tasks}
        
        configured_server_ids = set(self.mcp_clients.keys())
        if not configured_server_ids:
            self.logger.warning("No MCP clients are configured. Full $result reference validation will be limited.", event_type="PLANNER_VALIDATION_WARN")

        for task_node in dag.tasks:
            # Server ID and Special Tool Validation
            if task_node.server_id: 
                if configured_server_ids and task_node.server_id not in configured_server_ids:
                    raise ValueError(
                        f"Task '{task_node.id}' (tool: {task_node.tool_name}) references server_id '{task_node.server_id}' which was not found "
                        f"in the configured MCP clients: {list(configured_server_ids)}"
                    )
            
            if task_node.tool_name in ["direct_answer", "cannot_answer_without_tools"]:
                if task_node.server_id is not None: # Should be None (JSON null) for these
                    self.logger.warning(
                        f"Task '{task_node.id}' uses special tool '{task_node.tool_name}' but has server_id '{task_node.server_id}'. "
                        f"Special tools should have server_id set to null or be omitted."
                    )
            
            # Input Validation for $result references
            for input_key, input_value in task_node.inputs.items():
                if isinstance(input_value, str) and input_value.startswith("$result."):
                    parsed_ref = self._parse_result_reference(input_value)
                    if not parsed_ref:
                        raise ValueError(f"Task '{task_node.id}', input '{input_key}': Invalid $result reference format: '{input_value}'")
                    
                    producer_task_id, path_segments = parsed_ref

                    if producer_task_id not in task_id_to_node_map:
                        if is_phase_specific_validation and "_" in producer_task_id:
                            self.logger.debug(f"Task '{task_node.id}', input '{input_key}': Producer task ID '{producer_task_id}' from reference '{input_value}' is not local to the current phase. Assuming it's a global ID for a previous phase task. Skipping schema validation for this reference at this stage.")
                            continue # Skip schema validation for this presumed cross-phase reference
                        # If not phase-specific validation, or if it doesn't look like a global ID, then it's an error.
                        raise ValueError(f"Task '{task_node.id}', input '{input_key}': Producer task ID '{producer_task_id}' from reference '{input_value}' not found in the current DAG task IDs: {list(task_ids_in_current_dag)}")

                    producer_task_node = task_id_to_node_map[producer_task_id]
                    producer_output_schema: Optional[Dict[str, Any]] = None

                    # 1. Get Producer's Output Schema
                    if producer_task_node.tool_name == "direct_answer":
                        producer_output_schema = {"type": "object", "properties": {"answer_text": {"type": "string"}}}
                    elif producer_task_node.tool_name == "cannot_answer_without_tools":
                         producer_output_schema = {"type": "object", "properties": {"reason": {"type": "string"}}}
                    elif producer_task_node.server_id and producer_task_node.server_id in self.mcp_clients:
                        mcp_client = self.mcp_clients[producer_task_node.server_id]
                        producer_tool_schema_obj: Optional[ToolSchema] = None
                        
                        # Access McpClient.tool_schemas (should be populated list of ToolSchema models or compatible dicts)
                        found_raw_schema = None
                        if hasattr(mcp_client, 'tool_schemas') and mcp_client.tool_schemas:
                            for raw_s in mcp_client.tool_schemas: 
                                s_name = getattr(raw_s, 'tool_name', None) if isinstance(raw_s, ToolSchema) else (raw_s.get('tool_name') if isinstance(raw_s, dict) else None)
                                if s_name == producer_task_node.tool_name:
                                    found_raw_schema = raw_s
                                    break
                        
                        if found_raw_schema:
                            if isinstance(found_raw_schema, dict):
                                producer_tool_schema_obj = ToolSchema(**found_raw_schema)
                            elif isinstance(found_raw_schema, ToolSchema):
                                producer_tool_schema_obj = found_raw_schema
                            
                        if not producer_tool_schema_obj:
                            # Attempt to fetch if not found in cache (consider if list_tools should be async here or schemas pre-fetched)
                            # For simplicity of this validation step, we assume schemas are loaded. Production might need async fetch.
                            self.logger.warning(f"Schema for producer {producer_task_node.tool_name} not in mcp_client.tool_schemas. This might indicate schemas not fully pre-loaded for validation.")
                            # Potentially raise error or try a more dynamic fetch if critical for validation flow.
                            # For now, error if not found in pre-loaded:
                            raise ValueError(f"Task '{task_node.id}', input '{input_key}': Could not find schema for producer tool '{producer_task_node.tool_name}' on server '{producer_task_node.server_id}' in McpClient's loaded schemas. Reference: '{input_value}'")
                        producer_output_schema = producer_tool_schema_obj.output_schema
                    elif not producer_task_node.server_id and not producer_task_node.tool_name in ["direct_answer", "cannot_answer_without_tools"]:
                        raise ValueError(
                            f"Task '{task_node.id}', input '{input_key}': Producer task '{producer_task_node.id}' (tool: {producer_task_node.tool_name}) has no server_id and is not a special tool. Cannot determine output schema. Reference: '{input_value}'"
                        )
                    else:
                         raise ValueError(f"Task '{task_node.id}', input '{input_key}': MCP client for producer task '{producer_task_node.id}' server '{producer_task_node.server_id}' not found. Cannot validate schema. Reference: '{input_value}'")

                    if not producer_output_schema: # Should be caught by specific cases above
                        raise ValueError(f"Task '{task_node.id}', input '{input_key}': Output schema for producer task '{producer_task_node.id}' (tool: {producer_task_node.tool_name}) is unexpectedly missing. Reference: '{input_value}'")

                    # 2. Get Producer's Output Field Type from path
                    producer_field_type = self._get_schema_type_from_path(producer_output_schema, path_segments)
                    if producer_field_type is None:
                        raise ValueError(f"Task '{task_node.id}', input '{input_key}': Path '{'.'.join(path_segments)}' not found or invalid in output schema of producer task '{producer_task_node.id}' (tool: {producer_task_node.tool_name}). Reference: '{input_value}'. Producer output schema: {producer_output_schema}")

                    # 3. Get Consumer's Expected Input Type for this specific parameter
                    consumer_expected_type: Optional[str] = None
                    consumer_param_schema_dict: Optional[Dict[str, Any]] = None

                    if task_node.tool_name == "direct_answer":
                        if input_key == "answer_text":
                            consumer_expected_type = "string"
                            consumer_param_schema_dict = {"type": "string"}
                        else:
                            raise ValueError(f"Task '{task_node.id}' (special tool {task_node.tool_name}): Invalid input key '{input_key}'. Expected 'answer_text'.")
                    elif task_node.tool_name == "cannot_answer_without_tools":
                        if input_key == "reason":
                            consumer_expected_type = "string"
                            consumer_param_schema_dict = {"type": "string"}
                        else:
                            raise ValueError(f"Task '{task_node.id}' (special tool {task_node.tool_name}): Invalid input key '{input_key}'. Expected 'reason'.")
                    elif task_node.server_id and task_node.server_id in self.mcp_clients:
                        consumer_mcp_client = self.mcp_clients[task_node.server_id]
                        consumer_tool_schema_obj: Optional[ToolSchema] = None
                        
                        found_consumer_raw_schema = None
                        if hasattr(consumer_mcp_client, 'tool_schemas') and consumer_mcp_client.tool_schemas:
                            for raw_s_consumer in consumer_mcp_client.tool_schemas:
                                s_name_consumer = getattr(raw_s_consumer, 'tool_name', None) if isinstance(raw_s_consumer, ToolSchema) else (raw_s_consumer.get('tool_name') if isinstance(raw_s_consumer, dict) else None)
                                if s_name_consumer == task_node.tool_name:
                                    found_consumer_raw_schema = raw_s_consumer
                                    break
                        
                        if found_consumer_raw_schema:
                            if isinstance(found_consumer_raw_schema, dict):
                                consumer_tool_schema_obj = ToolSchema(**found_consumer_raw_schema)
                            elif isinstance(found_consumer_raw_schema, ToolSchema):
                                consumer_tool_schema_obj = found_consumer_raw_schema
                                
                        if not consumer_tool_schema_obj:
                            raise ValueError(f"Task '{task_node.id}': Could not find schema for its own tool '{task_node.tool_name}' on server '{task_node.server_id}' in McpClient's loaded schemas.")
                        
                        # MODIFICATION START: Simplified logic to find input parameter schema
                        # consumer_tool_schema_obj is an instance of ToolSchema
                        # consumer_tool_schema_obj.input_schema is List[ToolParameter]
                        
                        consumer_param_schema_dict = None 
                        found_param_model: Optional[ToolParameter] = None

                        # Ensure input_schema is a list (it should be, based on ToolSchema definition)
                        if isinstance(consumer_tool_schema_obj.input_schema, list):
                            for param_def_item in consumer_tool_schema_obj.input_schema:
                                # param_def_item could be a ToolParameter model or a dict if parsing was loose
                                current_param_name = None
                                if isinstance(param_def_item, ToolParameter):
                                    current_param_name = param_def_item.name
                                elif isinstance(param_def_item, dict):
                                    current_param_name = param_def_item.get("name")
                                
                                if current_param_name == input_key:
                                    if isinstance(param_def_item, ToolParameter):
                                        found_param_model = param_def_item
                                    elif isinstance(param_def_item, dict): # Convert to model if it was a dict
                                        try:
                                            found_param_model = ToolParameter(**param_def_item)
                                        except ValidationError as ve_param:
                                            self.logger.error(f"Failed to validate list item to ToolParameter for {input_key} in {task_node.tool_name}: {ve_param}")
                                            raise ValueError(f"Task '{task_node.id}', input '{input_key}': Invalid parameter definition in tool '{task_node.tool_name}' schema list.")
                                    break 
                        
                        if found_param_model:
                            consumer_param_schema_dict = found_param_model.model_dump(exclude_none=True) # Get dict from Pydantic model
                        else:
                            # Construct list of available names for error message
                            available_names = []
                            if isinstance(consumer_tool_schema_obj.input_schema, list):
                                for p_item in consumer_tool_schema_obj.input_schema:
                                    if isinstance(p_item, ToolParameter):
                                        available_names.append(p_item.name)
                                    elif isinstance(p_item, dict) and p_item.get("name"):
                                        available_names.append(p_item.get("name"))
                            raise ValueError(
                                f"Task '{task_node.id}', input '{input_key}': This input parameter is not defined in the tool '{task_node.tool_name}'. "
                                f"Available input parameter names: {available_names}. Input schema structure: {type(consumer_tool_schema_obj.input_schema)}"
                            )
                        # MODIFICATION END

                        # consumer_param_schema_dict should now be the schema for the specific input_key
                        if not consumer_param_schema_dict: # Should have been caught by raises above
                             raise ValueError(f"Internal error: consumer_param_schema_dict not set for {input_key}")

                        raw_consumer_type = consumer_param_schema_dict.get("type")
                        if isinstance(raw_consumer_type, list):
                            non_null_types = [t for t in raw_consumer_type if t != "null"]
                            consumer_expected_type = non_null_types[0] if non_null_types else "null"
                        elif isinstance(raw_consumer_type, str):
                            consumer_expected_type = raw_consumer_type
                        # if consumer_expected_type is still None here, it means the type field was missing/invalid in schema

                    elif not task_node.server_id : # and not a special tool (already handled)
                         raise ValueError(
                            f"Task '{task_node.id}' (tool: {task_node.tool_name}) has no server_id and is not a special tool. Cannot determine input schema for parameter '{input_key}'."
                        )
                    else: # server_id not in mcp_clients for consumer task
                        raise ValueError(f"Task '{task_node.id}': MCP client for its server '{task_node.server_id}' not found. Cannot validate input schema for parameter '{input_key}'.")

                    if consumer_expected_type is None:
                         raise ValueError(f"Task '{task_node.id}', input '{input_key}': Could not determine expected type for this input parameter of tool '{task_node.tool_name}'. Parameter schema: {consumer_param_schema_dict}")

                    # 4. Basic Type Compatibility Check
                    compatible = False
                    if producer_field_type == consumer_expected_type:
                        compatible = True
                    elif producer_field_type == "integer" and consumer_expected_type == "number":
                        compatible = True
                    elif consumer_expected_type == "string" and producer_field_type in ["number", "integer", "boolean"]:
                        # Allow conversion of primitive types to string if consumer expects string
                        self.logger.debug(f"Task '{task_node.id}', input '{input_key}': Allowing implicit conversion from producer type '{producer_field_type}' to consumer type 'string'.")
                        compatible = True
                    # TODO: Add more sophisticated compatibility rules (e.g., for 'anyOf', 'oneOf', array item types if consumer is array)
                    # TODO: Handle producer_field_type == "array" and consumer_expected_type == "array" -> check item types

                    if not compatible:
                        raise ValueError(
                            f"Task '{task_node.id}', input '{input_key}': Type mismatch. "
                            f"Producer task '{producer_task_node.id}' (tool: {producer_task_node.tool_name}) output field '{'.'.join(path_segments)}' has type '{producer_field_type}', "
                            f"but consumer tool '{task_node.tool_name}' expects type '{consumer_expected_type}' for input '{input_key}'. Reference: '{input_value}'"
                        )
                    self.logger.debug(f"Task '{task_node.id}', input '{input_key}': $result ref '{input_value}' schema validated. Producer type '{producer_field_type}', Consumer expects '{consumer_expected_type}'. Compatible.")

            # Dependency and Cycle Check Validation (existing logic from original _validate_dag_logic)
            # Moved this section to run *after* all individual task server_id and input checks are done for all tasks.
            # This is because the adj list construction needs all tasks to be initially parsed.

        # Perform Dependency and Cycle Check Validation for the entire DAG once
        adj: Dict[str, List[str]] = {task_id_val: [] for task_id_val in task_ids_in_current_dag}
        in_degree: Dict[str, int] = {task_id_val: 0 for task_id_val in task_ids_in_current_dag}

        for task_node_for_deps in dag.tasks: # Iterate again specifically for dependency structure
            if task_node_for_deps.dependencies:
                for dep_id in task_node_for_deps.dependencies:
                    if dep_id == task_node_for_deps.id: 
                        raise ValueError(f"Task '{task_node_for_deps.id}' cannot depend on itself.")

                    if dep_id not in task_ids_in_current_dag:
                        if is_phase_specific_validation:
                            # In phase-specific validation, a dependency not in the current phase's tasks
                            # might be a (valid or invalid) reference to a task from a PREVIOUS phase.
                            # The final, full DAG validation after all phases are stitched will catch truly undefined inter-phase dependencies.
                            self.logger.debug(f"Phase-specific validation: Dependency '{dep_id}' for task '{task_node_for_deps.id}' is not in current phase's tasks. Assuming inter-phase dependency or placeholder for later stitching.")
                            continue 
                        else:
                            # This is full DAG validation (not phase-specific), so the dependency MUST exist.
                            raise ValueError(
                                f"Task '{task_node_for_deps.id}' has an undefined dependency: '{dep_id}'. "
                                f"'{dep_id}' does not match any task ID in the DAG: {list(task_ids_in_current_dag)}"
                            )
                    
                    # If dep_id is valid and in the current DAG.
                    adj[dep_id].append(task_node_for_deps.id)
                    in_degree[task_node_for_deps.id] += 1
        
        # Call the new pattern validation method here
        try:
            self._validate_search_extract_use_pattern(dag, task_id_to_node_map)
        except ValueError as ve_pattern:
            # Log the pattern validation error and re-raise to halt processing if needed
            self.logger.error(f"Search-Extract-Use pattern validation failed: {ve_pattern}", exc_info=True, event_type="PLANNER_PATTERN_VALIDATION_ERROR")
            raise # Re-raise to indicate validation failure

        visited_during_dfs: Set[str] = set()
        recursion_stack: Set[str] = set()

        def detect_cycle_util(node_id: str) -> bool:
            visited_during_dfs.add(node_id)
            recursion_stack.add(node_id)

            for neighbor_id in adj.get(node_id, []): 
                if neighbor_id not in visited_during_dfs:
                    if detect_cycle_util(neighbor_id):
                        return True
                elif neighbor_id in recursion_stack:
                    self.logger.error(f"Cycle detected involving node {neighbor_id} (part of recursion stack: {recursion_stack})")
                    # To provide a more helpful error, one might try to reconstruct the cycle path here.
                    # For now, this message and raising ValueError is the primary goal.
                    return True
            
            recursion_stack.remove(node_id)
            return False

        for task_id_str_val in task_ids_in_current_dag:
            if task_id_str_val not in visited_during_dfs:
                if detect_cycle_util(task_id_str_val):
                    # More detailed cycle path reconstruction could be added here.
                    raise ValueError("Cyclic dependency detected in the task DAG. Please review task dependencies.")
        
        self.logger.info("DAG logical validation, including $result reference schema checks, passed successfully.")

    def _validate_search_extract_use_pattern(self, dag: TaskDAG, task_id_to_node_map: Dict[str, TaskNode]) -> None:
        """
        Validates adherence to the Search -> Extract (direct_answer) -> Use (Structured Data Tool) pattern.
        This is a heuristic check for common anti-patterns.
        """
        self.logger.debug("Starting Search-Extract-Use pattern validation.")

        SEARCH_TOOL_NAMES = ["tavily_search"] # Add other search tools if any
        STRUCTURED_DATA_TOOL_NAMES = ["google_sheets_append", "google_sheets_update"] # Add other tools that write structured data

        # Build a map of dependents for easier lookup
        dependents_map: Dict[str, List[TaskNode]] = {task_id: [] for task_id in task_id_to_node_map.keys()}
        for task_node_for_dep_build in dag.tasks:
            for dep_id in task_node_for_dep_build.dependencies:
                if dep_id in dependents_map:
                    dependents_map[dep_id].append(task_node_for_dep_build)

        for task_node in dag.tasks:
            # CHECK 1: Is a structured data tool directly consuming a known search tool's complex output? (Existing Check)
            if task_node.tool_name in STRUCTURED_DATA_TOOL_NAMES:
                for input_key, input_value in task_node.inputs.items():
                    if isinstance(input_value, str) and input_value.startswith("$result."):
                        parsed_ref = self._parse_result_reference(input_value)
                        if not parsed_ref: continue 
                        producer_task_id, path_segments = parsed_ref
                        if producer_task_id not in task_id_to_node_map: continue
                        producer_task_node = task_id_to_node_map[producer_task_id]

                        if producer_task_node.tool_name in SEARCH_TOOL_NAMES:
                            # If path segments point to common collection fields from search tools
                            is_complex_search_output_ref = any(ps.lower() in ["results", "hits", "entries", "items"] for ps in path_segments)
                            if not path_segments: # Direct reference to the whole output object of search tool
                                is_complex_search_output_ref = True
                            
                            if is_complex_search_output_ref:
                                consumer_param_schema = None
                                consumer_expected_type = None
                                if task_node.server_id and task_node.server_id in self.mcp_clients:
                                    client = self.mcp_clients[task_node.server_id]
                                    # Correctly get ToolSchema object from client.tool_schemas
                                    consumer_tool_schema_obj = None
                                    if hasattr(client, 'tool_schemas') and client.tool_schemas:
                                        for schema_item in client.tool_schemas:
                                            name_to_check = getattr(schema_item, 'tool_name', None) if isinstance(schema_item, ToolSchema) else (schema_item.get('tool_name') if isinstance(schema_item, dict) else None)
                                            if name_to_check == task_node.tool_name:
                                                consumer_tool_schema_obj = schema_item if isinstance(schema_item, ToolSchema) else ToolSchema(**schema_item)
                                                break
                                    
                                    if consumer_tool_schema_obj and consumer_tool_schema_obj.input_schema.get("properties") and input_key in consumer_tool_schema_obj.input_schema["properties"]:
                                        consumer_param_schema = consumer_tool_schema_obj.input_schema["properties"][input_key]
                                        raw_type = consumer_param_schema.get("type")
                                        if isinstance(raw_type, str): consumer_expected_type = raw_type
                                        elif isinstance(raw_type, list): consumer_expected_type = next((t for t in raw_type if t != "null"), "null")

                                producer_output_details = producer_task_node.output_schema if producer_task_node.output_schema else {}
                                producer_field_type = self._get_schema_type_from_path(producer_output_details, path_segments)

                                if consumer_expected_type in ["string", "number", "integer", "boolean"] and producer_field_type in ["array", "object"]:
                                    raise ValueError(
                                        f"Task '{task_node.id}' (tool: {task_node.tool_name}) input '{input_key}' appears to directly consume a complex output ('{producer_field_type}' from path '{'.'.join(path_segments)}') "
                                        f"from search tool '{producer_task_node.tool_name}' (task: {producer_task_id}). "
                                        f"An intermediate 'direct_answer' extraction task is likely missing as per Search-Extract-Use pattern."
                                    )
            
            # CHECK 2: Forward-looking validation starting from Search Tasks
            if task_node.tool_name in SEARCH_TOOL_NAMES:
                search_task_id = task_node.id
                found_valid_extractor = False
                # Iterate over direct dependents of the search_task
                for dependent_node in dependents_map.get(search_task_id, []):
                    if dependent_node.tool_name == "direct_answer" and dependent_node.server_id is None:
                        # This is a candidate extraction task. Validate its input.
                        is_valid_extraction_input = False
                        for ext_input_key, ext_input_value in dependent_node.inputs.items():
                            if isinstance(ext_input_value, str) and ext_input_value.startswith("$result."):
                                parsed_ext_ref = self._parse_result_reference(ext_input_value)
                                if parsed_ext_ref:
                                    ext_producer_id, ext_path_segments = parsed_ext_ref
                                    # Check if it references the current search task and a plausible collection field
                                    if ext_producer_id == search_task_id and \
                                       (not ext_path_segments or any(ps.lower() in ["results", "hits", "entries", "items"] for ps in ext_path_segments)):
                                        is_valid_extraction_input = True
                                        break 
                        if is_valid_extraction_input:
                            found_valid_extractor = True
                            self.logger.debug(f"Found valid extractor task '{dependent_node.id}' for search task '{search_task_id}'.")
                            # Now check if any structured data tool consumes this extractor
                            found_valid_use_task = False
                            for use_candidate_node in dependents_map.get(dependent_node.id, []):
                                if use_candidate_node.tool_name in STRUCTURED_DATA_TOOL_NAMES:
                                    # Check if the use_candidate_node's input references the extractor's output
                                    consumes_extractor_output = False
                                    for use_input_key, use_input_value in use_candidate_node.inputs.items():
                                        if isinstance(use_input_value, str) and use_input_value.startswith("$result."):
                                            parsed_use_ref = self._parse_result_reference(use_input_value)
                                            if parsed_use_ref and parsed_use_ref[0] == dependent_node.id and \
                                               (not parsed_use_ref[1] or parsed_use_ref[1] == ["answer_text"]): # direct_answer output is answer_text
                                                consumes_extractor_output = True
                                                break
                                    if consumes_extractor_output:
                                        self.logger.debug(f"Found valid use task '{use_candidate_node.id}' for extractor '{dependent_node.id}'. Pattern appears correct.")
                                        found_valid_use_task = True
                                        break # Found a valid use task for this extractor
                            if not found_valid_use_task and dependents_map.get(dependent_node.id):
                                # Extractor exists, but no structured data tool uses its output directly among its dependents.
                                # This might be okay if other tools use it, or if it's the final output.
                                # For stricter pattern, one might warn/error if no STRUCTURED_DATA_TOOL consumes it.
                                self.logger.debug(f"Extractor '{dependent_node.id}' for search task '{search_task_id}' was found, but no subsequent structured data tool directly consumes its 'answer_text' output among its dependents.")
                        # else: This direct_answer task is not a valid extractor for *this* search task based on input reference.
                
                # If search task has dependents but no valid extractor was found among them
                # AND at least one of those dependents is a STRUCTURED_DATA_TOOL, then it's a likely violation of missing extractor.
                if not found_valid_extractor and dependents_map.get(search_task_id):
                    has_structured_data_tool_dependent = False
                    for dep_node in dependents_map.get(search_task_id, []):
                        if dep_node.tool_name in STRUCTURED_DATA_TOOL_NAMES:
                            # Check if this structured data tool is *directly* consuming the search task's complex output
                            # This part overlaps with CHECK 1 but confirms the forward-looking scenario.
                            for _, sdt_input_val in dep_node.inputs.items():
                                if isinstance(sdt_input_val, str) and sdt_input_val.startswith(f"$result.{search_task_id}."):
                                    parsed_sdt_ref = self._parse_result_reference(sdt_input_val)
                                    if parsed_sdt_ref and (not parsed_sdt_ref[1] or any(ps.lower() in ["results", "hits", "entries", "items"] for ps in parsed_sdt_ref[1])):
                                        has_structured_data_tool_dependent = True
                                        break
                            if has_structured_data_tool_dependent: break
                    
                    if has_structured_data_tool_dependent:
                        raise ValueError(
                            f"Search task '{search_task_id}' (tool: {task_node.tool_name}) is directly feeding a structured data tool "
                            f"without a valid intermediate 'direct_answer' extraction task. Pattern violation."
                        )

        self.logger.debug("Search-Extract-Use pattern validation (heuristic checks) completed.")

    def _remap_dependencies(self, task_node: TaskNode, id_mapping: Dict[str, str], current_phase_original_ids: Set[str], all_stitched_task_ids: Set[str], phase_id: str) -> List[str]:
        """Helper to remap dependencies for a task during stitching."""
        new_dependencies = []
        for dep_id in task_node.dependencies:
            if dep_id in id_mapping:
                new_dependencies.append(id_mapping[dep_id])
            elif dep_id in all_stitched_task_ids:
                new_dependencies.append(dep_id)
            elif dep_id in current_phase_original_ids:
                self.logger.warning(f"Task {task_node.id} (new) depends on {dep_id} from phase {phase_id}, but {dep_id} wasn't found in the ID mapping for this phase. Keeping original dependency.")
                anticipated_mapped_id = f"{phase_id}_{dep_id}"
                if anticipated_mapped_id in all_stitched_task_ids:
                    new_dependencies.append(anticipated_mapped_id)
                else:
                    self.logger.error(f"Cannot resolve dependency '{dep_id}' for task {task_node.id} (new). Neither in current phase mapping nor previous phases.")
                    new_dependencies.append(dep_id)
            else:
                self.logger.warning(f"Task {task_node.id} (new) has unresolved dependency '{dep_id}'. It's not in current phase or previous phases. Keeping original.")
                new_dependencies.append(dep_id)
        return new_dependencies

    def _stitch_single_phase_dag(
        self,
        existing_tasks: List[TaskNode],
        phase_dag_to_stitch: TaskDAG,
        phase_obj: Phase,
        tasks_in_immediately_preceding_phase: List[TaskNode]
    ) -> List[TaskNode]:
        """Stitches tasks from a single phase DAG onto an existing list of tasks.
        
        Args:
            existing_tasks: List of TaskNodes already stitched from previous phases.
            phase_dag_to_stitch: The TaskDAG generated for the current phase.
            phase_obj: The Phase object corresponding to phase_dag_to_stitch.
            tasks_in_immediately_preceding_phase: List of TaskNodes from the *prior* phase (after stitching).

        Returns:
            Updated list of TaskNodes including the newly stitched tasks.
        """
        newly_stitched_tasks: List[TaskNode] = []
        all_existing_task_ids: Set[str] = {t.id for t in existing_tasks}
        all_stitched_task_ids_in_this_run: Set[str] = set(all_existing_task_ids)
        current_phase_original_ids: Set[str] = {t.id for t in phase_dag_to_stitch.tasks}
        id_mapping: Dict[str, str] = {}

        for task_node in phase_dag_to_stitch.tasks:
            original_task_id = task_node.id
            new_task_id_base = f"{phase_obj.phase_id}_{original_task_id}"
            new_task_id = new_task_id_base
            counter = 0
            while new_task_id in all_stitched_task_ids_in_this_run:
                counter += 1
                new_task_id = f"{new_task_id_base}_{counter}"
            
            id_mapping[original_task_id] = new_task_id
            all_stitched_task_ids_in_this_run.add(new_task_id)

        for task_node in phase_dag_to_stitch.tasks:
            original_id = task_node.id
            new_id = id_mapping[original_id]
            task_node.id = new_id
            
            task_node.dependencies = self._remap_dependencies(
                task_node, id_mapping, current_phase_original_ids, all_existing_task_ids, phase_obj.phase_id
            )
            # Rewrite $result references in inputs to use the new global IDs for intra-phase tasks
            task_node.inputs = self._rewrite_input_references_recursive(task_node.inputs, id_mapping)

            if tasks_in_immediately_preceding_phase and not task_node.dependencies:
                self.logger.info(f"Auto-linking task {new_id} to all tasks from previous phase ({phase_obj.phase_id} depends on prior).")
                task_node.dependencies.extend([prev_task.id for prev_task in tasks_in_immediately_preceding_phase])
            
            newly_stitched_tasks.append(task_node)

        return existing_tasks + newly_stitched_tasks

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True  # Reraise the last exception if all retries fail
    )
    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        '''Invokes the LLM with the given prompt and system prompt, with retry logic.'''
        self.logger.debug(f"Calling LLM. System prompt (first 100 chars): {system_prompt[:100] if system_prompt else 'None'}. User prompt (first 100 chars): {prompt[:100]}...")
        
        llm_input_for_invoke = []
        if system_prompt:
            llm_input_for_invoke.append(("system", system_prompt))
        llm_input_for_invoke.append(("human", prompt))

        try:
            # Assuming ainvoke can handle a list of tuples (type, content)
            response = await self.llm.ainvoke(llm_input_for_invoke) 
            
            # If response is an AIMessage or similar, extract content
            llm_output = response.content if hasattr(response, 'content') else str(response)

            if not llm_output or llm_output.strip() == "":
                self.logger.warning("_call_llm received empty output. Retrying if attempts left...")
                raise ValueError("LLM returned empty output.") # Force retry for empty string

            self.logger.debug(f"LLM call successful. Output (first 100 chars): {llm_output[:100]}...")
            return llm_output
        except OutputParserException as ope: # Langchain specific exception if output cannot be parsed by a model's default parser
            self.logger.warning(f"_call_llm encountered OutputParserException: {ope}. Retrying if attempts left...")
            raise # Reraise to trigger tenacity retry
        except RetryError: # This is raised by tenacity after all retries are exhausted
            self.logger.error("_call_llm: All retry attempts failed.")
            raise
        except Exception as e:
            self.logger.error(f"_call_llm failed: {e}", exc_info=True)
            raise # Reraise other exceptions

    async def _generate_high_level_phases(self, query: str, conversation_context: Optional[str] = None) -> HierarchicalPlan:
        """Generates a high-level plan (list of phases) from the user query."""
        system_prompt_str = self.prompt_builder.get_high_level_phase_system_prompt()
        main_prompt_str = self.prompt_builder.get_high_level_phase_generation_prompt(
            query=query,
            conversation_context=conversation_context
        )
        
        self.logger.info(f"Generating high-level phases for query: '{query[:100]}...'")
        llm_output = await self._call_llm(prompt=main_prompt_str, system_prompt=system_prompt_str)
        
        self.logger.debug(f"Raw LLM output for HierarchicalPlan generation:\n{llm_output}")

        # Attempt to parse JSON from the response (handles potential markdown code blocks)
        json_str = llm_output
        markdown_match = re.search(r'```json\\s*(\\{.*?\\})\\s*```', llm_output, re.DOTALL | re.IGNORECASE)
        if markdown_match:
            json_str = markdown_match.group(1)
        else:
            # Fallback for plain JSON or if regex fails
            first_brace = llm_output.find('{')
            last_brace = llm_output.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = llm_output[first_brace : last_brace + 1]

        json_str = json_str.strip()
        if not json_str:
            self.logger.error("LLM output for HierarchicalPlan was empty or not valid JSON after stripping.")
            # Return a plan with a single phase indicating failure, or raise an error
            # For now, let's raise an error to be caught by the caller.
            raise ValueError("LLM output for HierarchicalPlan is empty or not valid JSON after stripping.")

        try:
            # Ensure that the HierarchicalPlan.model_validate_json can handle the string directly
            plan = HierarchicalPlan.model_validate_json(json_str)
            if not plan.phases:
                self.logger.warning("HierarchicalPlan generated with no phases. Query: '{query[:100]}...'. LLM output: '{llm_output[:200]}...'")
                # Consider if this should also raise an error or return an empty plan object based on desired strictness
            self.logger.info(f"Successfully parsed LLM output into HierarchicalPlan with {len(plan.phases)} phases.")
            return plan
        except ValidationError as e:
            self.logger.error(f"Pydantic ValidationError for HierarchicalPlan: {e}. JSON tried: '{json_str[:500]}...'", exc_info=True)
            raise ValueError(f"Pydantic validation failed for HierarchicalPlan: {e}. Data: '{json_str[:500]}...'") from e
        except json.JSONDecodeError as e: # Should be caught by Pydantic's model_validate_json ideally, but as a fallback
            self.logger.error(f"JSONDecodeError parsing HierarchicalPlan from LLM output: {e}. Output:\n{json_str[:500]}...", exc_info=True)
            raise ValueError(f"Invalid JSON for HierarchicalPlan: {e}. Content: '{json_str[:500]}...'") from e

    async def _generate_dag_for_phase(
        self,
        phase: Phase,
        original_query: str,
        all_tool_schemas: List[ServerToolSchemaGroup],
        conversation_context: Optional[str] = None,
        previous_phases_outputs: Optional[Dict[str, Any]] = None,
    ) -> TaskDAG:
        """Generates a TaskDAG for a single phase, considering outputs from previous phases."""
        self.logger.info(f"Generating DAG for phase: {phase.phase_id} - '{phase.description}'")

        # 1. Get phase-specific prompt
        # The prompt builder now handles formatting prior outputs correctly with global IDs
        prompt_for_phase = self.prompt_builder.get_phase_specific_dag_prompt(
            original_query=original_query,
            current_phase=phase,
            tool_schemas=all_tool_schemas, # Provide all tools, LLM should select relevant ones
            available_prior_outputs=previous_phases_outputs,
            conversation_context=conversation_context
        )

        # 2. Call LLM
        try:
            self.logger.debug(f"Prompt for phase '{phase.phase_id}':\n{prompt_for_phase[:500]}...") # Log snippet
            llm_response_str = await self._call_llm(prompt=prompt_for_phase) # System prompt is included by get_phase_specific_dag_prompt
            self.logger.debug(f"LLM response for phase '{phase.phase_id}':\n{llm_response_str[:500]}...")
        except Exception as e:
            self.logger.error(f"LLM call failed for phase '{phase.phase_id}' DAG generation: {e}", exc_info=True)
            raise McpError(f"LLM call failed during DAG generation for phase '{phase.phase_id}'.") from e

        # 3. Parse LLM output to TaskDAG model for this phase
        try:
            phase_dag = self._parse_llm_output_to_dag_model(llm_response_str)
            self.logger.info(f"Successfully parsed DAG for phase '{phase.phase_id}'. Tasks: {len(phase_dag.tasks)}")
            
            # Set phase_id for all tasks in this locally generated DAG
            for task_node in phase_dag.tasks:
                task_node.phase_id = phase.phase_id

        except (ValidationError, ValueError, TypeError, OutputParserException) as e:
            self.logger.error(f"Failed to parse or validate LLM output for phase '{phase.phase_id}' into TaskDAG: {e}", exc_info=True)
            self.logger.error(f"LLM Output for phase '{phase.phase_id}' (that caused parsing error):\n{llm_response_str}")
            # Attempt to get a more specific error for the LLM/user
            error_details = str(e)
            if isinstance(e, ValidationError):
                error_details = e.errors()
            raise McpError(
                f"LLM output for phase '{phase.phase_id}' could not be parsed into a valid TaskDAG. "
                f"Please ensure the output is a single JSON object matching the TaskDAG schema. Details: {error_details}"
            ) from e

        # 4. Validate the phase-specific DAG logic (dependencies, inputs, outputs within the phase)
        # This validation primarily checks intra-phase consistency. 
        # Cross-phase reference validation happens later or is more lenient here.
        try:
            self._validate_dag_logic(phase_dag, is_phase_specific_validation=True) # is_phase_specific_validation=True ensures it validates local IDs primarily
            self.logger.info(f"Phase-specific DAG for '{phase.phase_id}' validated successfully.")
        except ValueError as e:
            self.logger.error(f"Validation error in generated DAG for phase '{phase.phase_id}': {e}", exc_info=True)
            raise McpError(f"Generated DAG for phase '{phase.phase_id}' is invalid: {e}") from e
        
        return phase_dag

    def _stitch_dags(self, phase_dags: Dict[str, TaskDAG], hierarchical_plan: HierarchicalPlan) -> TaskDAG:
        """Stitches multiple TaskDAGs (one per phase) into a single TaskDAG.
           Manages unique IDs and attempts to link phases sequentially if not otherwise specified by LLM.
        """
        final_tasks: List[TaskNode] = []
        all_task_ids: Set[str] = set()
        tasks_in_previous_phase: List[TaskNode] = []

        for i, phase_obj in enumerate(hierarchical_plan.phases):
            phase_id = phase_obj.phase_id
            current_dag = phase_dags.get(phase_id)
            if not current_dag:
                self.logger.warning(f"No DAG found for phase {phase_id} during stitching. Skipping.")
                continue

            current_phase_task_ids_mapping: Dict[str, str] = {}

            for task_node in current_dag.tasks:
                original_task_id = task_node.id
                new_task_id = f"{phase_id}_{original_task_id}"
                
                counter = 0
                while new_task_id in all_task_ids:
                    counter += 1
                    new_task_id = f"{phase_id}_{original_task_id}_{counter}"
                current_phase_task_ids_mapping[original_task_id] = new_task_id
                all_task_ids.add(new_task_id)
                
                task_node.id = new_task_id
                
                new_dependencies = []
                for dep_id in task_node.dependencies:
                    if dep_id in current_phase_task_ids_mapping:
                        new_dependencies.append(current_phase_task_ids_mapping[dep_id])
                    else:
                        self.logger.warning(f"Task {new_task_id} has unmapped dependency {dep_id}. It might be a cross-phase dependency not yet handled or an error.")
                        new_dependencies.append(dep_id)
                task_node.dependencies = new_dependencies

                # Rewrite $result references in inputs to use the new global IDs for intra-phase tasks
                task_node.inputs = self._rewrite_input_references_recursive(task_node.inputs, current_phase_task_ids_mapping)

                final_tasks.append(task_node)
            
            # Update tasks_in_previous_phase_after_stitching for the next iteration
            # This should capture all tasks that belong to the *current* phase, now with their global IDs.
            tasks_in_previous_phase_after_stitching = [tn for tn in final_tasks if tn.id.startswith(f"{phase_id}_")]

        return TaskDAG(tasks=final_tasks)

    async def _should_use_hierarchical_planning(self, query: str, conversation_context: Optional[str] = None) -> bool:
        """Determines if hierarchical planning should be used based on query complexity (LLM call)."""
        self.logger.warning("_should_use_hierarchical_planning is deprecated. Use _classify_query_planning_strategy.")
        # Simple heuristic for now, or call _classify_query_planning_strategy and check for COMPLEX
        # Forcing False for now as the new method will be the source of truth.
        return False

    async def _classify_query_planning_strategy(self, query: str, conversation_context: Optional[str] = None) -> str:
        """Determines the query type (COMPLEX, SIMPLE, NO_PLAN) via an LLM call."""
        self.logger.info(f"Classifying query planning strategy for: '{query[:100]}...'")
        
        default_strategy = "SIMPLE" # Default if LLM fails or gives unexpected output

        try:
            system_prompt = self.prompt_builder.get_query_type_classification_system_prompt()
            prompt = self.prompt_builder.get_query_type_classification_prompt(query, conversation_context)
            
            llm_response_raw = await self._call_llm(prompt=prompt, system_prompt=system_prompt)
            
            self.logger.debug(f"Query classification LLM raw response: '{llm_response_raw}'")

            # Attempt to parse JSON from the response (handles potential markdown code blocks)
            json_str = llm_response_raw
            markdown_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response_raw, re.DOTALL | re.IGNORECASE)
            if markdown_match:
                json_str = markdown_match.group(1)
            else:
                # Fallback for plain JSON or if regex fails
                first_brace = llm_response_raw.find('{')
                last_brace = llm_response_raw.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = llm_response_raw[first_brace : last_brace + 1]
            
            json_str = json_str.strip()
            if not json_str:
                self.logger.warning(f"LLM response for query classification was empty after stripping. Defaulting to {default_strategy}.")
                return default_strategy

            try:
                parsed_output = json.loads(json_str)
                query_type = parsed_output.get("query_type")

                if query_type in ["COMPLEX", "SIMPLE", "NO_PLAN"]:
                    self.logger.info(f"Query classified as {query_type}.")
                    return query_type
                else:
                    self.logger.warning(f"LLM returned an unexpected query_type '{query_type}'. Raw response: '{llm_response_raw}'. Defaulting to {default_strategy}.")
                    return default_strategy
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON from query classification LLM response: {e}. Raw response: '{llm_response_raw}'. Defaulting to {default_strategy}.")
                return default_strategy
        except RetryError as e_retry: # Assuming _call_llm might raise this if all retries fail
            self.logger.error(f"All retry attempts failed for query classification LLM call: {e_retry}. Defaulting to {default_strategy}.")
            return default_strategy
        except Exception as e_call: # Catch other exceptions from _call_llm or other operations
            self.logger.error(f"Error during LLM call or processing for query classification: {e_call}. Defaulting to {default_strategy}.", exc_info=True)
            return default_strategy

    async def generate_dag(
        self,
        query: str,
        agent_state: Optional[AgentState] = None,
        conversation_context: Optional[str] = None,
        replan_context: Optional[Dict[str, Any]] = None,
        failed_repair_error: Optional[str] = None,
        failed_repair_instructions: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[TaskDAG], str]:
        """Generates a task DAG based on the query, optionally considering prior context.

        Args:
            query: The user's original natural language query.
            agent_state: The current state of the agent. May contain current_phase_id, 
                         current_phase_description, and existing_task_dag for context.
            conversation_context: String representation of the conversation history.
            replan_context: Dictionary containing context for replanning, if applicable.
            failed_repair_error: Error message from a failed DAG repair attempt.
            failed_repair_instructions: Instructions used in a failed DAG repair attempt.

        Returns:
            A tuple containing the generated TaskDAG (or None if an error occurs) and a status message.
        """
        current_phase_id: Optional[str] = None
        current_phase_description: Optional[str] = None
        effective_query_for_planning = query # Default to original query

        if agent_state:
            current_phase_id = getattr(agent_state, 'current_phase_id', None)
            current_phase_description = getattr(agent_state, 'current_phase_description', None)
            if current_phase_id and current_phase_description:
                effective_query_for_planning = current_phase_description
                self.logger.info(f"Planner: Generating DAG for Phase ID: {current_phase_id} - '{current_phase_description}'. Original query: '{query[:100]}...'")
            else:
                self.logger.info(f"Planner: Generating DAG for main query: '{query[:100]}...' Replan context: {replan_context is not None}")
        else:
            self.logger.info(f"Planner: Generating DAG for main query (no agent_state): '{query[:100]}...' Replan context: {replan_context is not None}")

        try:
            all_tool_schemas = await self.get_tool_schemas()
            if not all_tool_schemas and self.settings.tool_usage_policy == "strict_if_available":
                self.logger.warning("No tool schemas loaded, but policy is strict_if_available. Planning may be limited.", event_type="PLANNER_NO_TOOLS_STRICT")
            elif not all_tool_schemas:
                self.logger.info("No tool schemas loaded. Planner will rely on direct_answer or indicate inability.", event_type="PLANNER_NO_TOOLS_INFO")

            # Use phase description for tool recommendation if available, else main query
            query_for_recommendation = current_phase_description if current_phase_id and current_phase_description else query
            recommended_tools = self._recommend_tools(query_for_recommendation, all_tool_schemas)
            self.logger.debug(f"Recommended tools count: {len(recommended_tools)} for query/phase: '{query_for_recommendation[:50]}...'")

            current_plan_tasks_list_of_dicts = []
            if replan_context is None:
                replan_context = {}

            if agent_state and agent_state.task_dag and agent_state.task_dag.tasks:
                current_plan_tasks_list_of_dicts = [task.model_dump() for task in agent_state.task_dag.tasks]
            elif replan_context.get('original_dag'):
                original_dag_data = replan_context['original_dag']
                if isinstance(original_dag_data, TaskDAG):
                    current_plan_tasks_list_of_dicts = [task.model_dump() for task in original_dag_data.tasks]
                elif isinstance(original_dag_data, dict) and 'tasks' in original_dag_data:
                    current_plan_tasks_list_of_dicts = original_dag_data['tasks']
            
            replan_context['current_plan_tasks'] = current_plan_tasks_list_of_dicts
            
            # --- Prompt Building ---
            main_prompt_str: str
            existing_task_statuses_section_str: str
            final_system_prompt: str

            # Determine if this is a genuine replan scenario for prompt selection
            is_genuine_replan_for_prompt = replan_context and \
                                           (replan_context.get('successful_tasks') is not None or \
                                            replan_context.get('failed_tasks') is not None or \
                                            replan_context.get('original_dag') is not None) # Add original_dag check
            
            effective_replan_context_for_prompt_builder = None
            if is_genuine_replan_for_prompt:
                effective_replan_context_for_prompt_builder = replan_context.copy() if replan_context else {}
                # Ensure current_plan_tasks is populated in the context for the prompt builder
                current_plan_tasks_list_of_dicts_for_builder = []
                if agent_state and agent_state.task_dag and agent_state.task_dag.tasks:
                    current_plan_tasks_list_of_dicts_for_builder = [task.model_dump() for task in agent_state.task_dag.tasks]
                elif effective_replan_context_for_prompt_builder.get('original_dag'):
                    original_dag_data = effective_replan_context_for_prompt_builder['original_dag']
                    if isinstance(original_dag_data, TaskDAG):
                        current_plan_tasks_list_of_dicts_for_builder = [task.model_dump() for task in original_dag_data.tasks]
                    elif isinstance(original_dag_data, dict) and 'tasks' in original_dag_data:
                        current_plan_tasks_list_of_dicts_for_builder = original_dag_data['tasks']
                effective_replan_context_for_prompt_builder['current_plan_tasks'] = current_plan_tasks_list_of_dicts_for_builder
            elif agent_state and agent_state.task_dag and agent_state.task_dag.tasks: # Standard plan but with an existing DAG from agent_state (e.g. after repair)
                effective_replan_context_for_prompt_builder = {'current_plan_tasks': [task.model_dump() for task in agent_state.task_dag.tasks]}


            if current_phase_id and current_phase_description:
                # Generating a brand new DAG for a specific phase.
                # This is not a replan of an *existing* phase DAG in the same sense as main query replanning.
                # Context from previous *completed* phases is handled by `available_prior_outputs`.
                available_prior_outputs = getattr(agent_state, 'accumulated_global_task_outputs', None) if agent_state else None

                current_phase_obj = Phase(phase_id=current_phase_id, description=current_phase_description, status="pending")

                # For generating a new phase DAG, the system prompt is specific to phase planning, and user prompt details phase objectives.
                system_prompt_for_phase, user_prompt_for_phase = self.prompt_builder.build_planning_prompt(
                    query=query, # Original query for overall context
                    tool_schemas=recommended_tools,
                    conversation_context=conversation_context,
                    query_type="COMPLEX_PHASE", # Indicate phase planning
                    current_phase=current_phase_obj,
                    available_prior_outputs=available_prior_outputs,
                    # replan_context here should be None as we are generating a NEW DAG for the phase
                    replan_context=None 
                )
                final_system_prompt = system_prompt_for_phase
                main_prompt_str = user_prompt_for_phase

            else:
                # Standard DAG generation for the main query, or REPLANNING an existing main DAG.
                system_prompt_for_main, user_prompt_for_main = self.prompt_builder.build_planning_prompt(
                    query=query,
                    tool_schemas=recommended_tools,
                    conversation_context=conversation_context,
                    # Pass the carefully constructed effective_replan_context_for_prompt_builder
                    replan_context=effective_replan_context_for_prompt_builder, 
                    failed_repair_error=failed_repair_error,
                    failed_repair_instructions=failed_repair_instructions
                )
                final_system_prompt = system_prompt_for_main
                main_prompt_str = user_prompt_for_main

            self.logger.debug(f"Planner Final System Prompt (first 200 chars): {final_system_prompt[:200]}...")
            self.logger.debug(f"Planner Main Prompt (first 200 chars): {main_prompt_str[:200]}...")

            llm_response_str = await self._call_llm(prompt=main_prompt_str, system_prompt=final_system_prompt)
            self.logger.debug(f"LLM Output for DAG (first 200 chars): {llm_response_str[:200]}...")

            parsed_dag = self._parse_llm_output_to_dag_model(llm_response_str)
            self.logger.info(f"Successfully parsed LLM output into TaskDAG model with {len(parsed_dag.tasks)} tasks.")

            self._validate_dag_logic(parsed_dag, is_phase_specific_validation=bool(current_phase_id))
            self.logger.info("TaskDAG passed structural and logical validation.")

            return parsed_dag, "DAG generated successfully."
            
        except Exception as e:
            self.logger.error(f"Error during DAG generation: {e}", exc_info=True)
            return None, "UNKNOWN"