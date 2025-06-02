import asyncio
from typing import Any, Dict, List, Set, Optional, Union
import logging
from langchain_core.language_models import BaseLanguageModel
from .mcp_client import McpClient, McpError, McpNotInitializedError, McpTimeoutError, McpConnectionError
from ..core.models import TaskNode, ToolExecutionResult, AgentState, TaskDAG
from ..config import get_settings, MochiWorkerConfig
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from ..core.logging import MochiLogger
import re
import json

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
SPECIAL_NON_MCP_TOOLS = ["direct_answer", "cannot_answer_without_tools"]

class TaskExecutor:
    def __init__(self, 
                mcp_clients: Dict[str, McpClient],
                synthesis_llm: Optional[BaseLanguageModel] = None, 
                default_max_retries: int = 2, 
                default_initial_retry_delay: float = 1.0,
                settings: Optional[MochiWorkerConfig] = None,
                logger_instance: Optional[MochiLogger] = None,
                global_prior_phase_outputs: Optional[Dict[str, Any]] = None):
        if not isinstance(mcp_clients, dict) or not all(isinstance(client, McpClient) for client in mcp_clients.values()):
            raise TypeError("mcp_clients must be a dictionary of McpClient instances.")
        self.mcp_clients = mcp_clients
        self.synthesis_llm = synthesis_llm
        self.current_settings: MochiWorkerConfig = settings if settings else get_settings()
        self.logger: MochiLogger = logger_instance or MochiLogger(config=self.current_settings.logging, name=f"mochi.{self.__class__.__name__}")

        self.max_retries = self.current_settings.execution.max_retries if self.current_settings.execution else default_max_retries
        self.initial_retry_delay = default_initial_retry_delay 
        self.global_prior_phase_outputs = global_prior_phase_outputs

    async def execute_task(self, task: TaskNode, task_results: Dict[str, ToolExecutionResult]) -> ToolExecutionResult:
        task_id_str = task.id
        
        if task.tool_name == "direct_answer":
            self.logger.info(f"Handling special non-MCP task: {task_id_str} (Tool: {task.tool_name}) with LLM synthesis.", event_type="TASK_EXEC_SPECIAL_SYNTHESIS")
            if not self.synthesis_llm:
                error_msg = f"Synthesis LLM not available for direct_answer task {task_id_str}."
                self.logger.error(error_msg, event_type="TASK_EXEC_ERROR_NO_SYNTHESIS_LLM")
                return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)

            # Resolve inputs for direct_answer task, specifically the answer_text
            resolved_prompt_text = ""
            try:
                # We need to resolve the entire inputs object to correctly get 'answer_text' if it itself is a reference
                # or contains references.
                # Temporarily create a dictionary for the single input we care about for resolution context.
                resolved_specific_inputs = self._resolve_inputs(TaskNode(id=task.id, tool_name=task.tool_name, server_id=task.server_id, inputs=task.inputs, dependencies=task.dependencies), task_results) # Pass full task for context if _resolve_inputs needs it
                prompt_text_from_resolved = resolved_specific_inputs.get("answer_text")
                
                if not prompt_text_from_resolved or not isinstance(prompt_text_from_resolved, str):
                    error_msg = f"Missing or invalid (after resolution) 'answer_text' input for direct_answer task {task_id_str}. Found: {prompt_text_from_resolved}"
                    self.logger.error(error_msg, event_type="TASK_EXEC_ERROR_INVALID_DA_INPUT")
                    return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)
                resolved_prompt_text = prompt_text_from_resolved

            except ValueError as e_resolve:
                error_msg = f"Input resolution failed for direct_answer task {task_id_str}: {e_resolve}"
                self.logger.error(error_msg, event_type="TASK_EXEC_ERROR_INPUT_RESOLUTION", metadata={'task_id': task_id_str, 'error': str(e_resolve)})
                return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)
            
            try:
                self.logger.info(f"Invoking synthesis LLM for direct_answer task {task_id_str}. Prompt (first 100 chars): '{resolved_prompt_text[:100]}...'", event_type="TASK_EXEC_LLM_SYNTHESIS_START")
                llm_response = await self.synthesis_llm.ainvoke(resolved_prompt_text)
                synthesized_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                output_data = {"answer_text": synthesized_answer}
                self.logger.info(f"LLM synthesis successful for direct_answer task {task_id_str}. Output (first 100 chars): '{synthesized_answer[:100]}...'", event_type="TASK_EXEC_LLM_SYNTHESIS_SUCCESS")
                return ToolExecutionResult(task_id=task_id_str, status="success", output=output_data)
            except Exception as e:
                self.logger.error(f"LLM synthesis failed for direct_answer task {task_id_str}: {e}", exc_info=True, event_type="TASK_EXEC_LLM_SYNTHESIS_ERROR")
                return ToolExecutionResult(task_id=task_id_str, status="failure", error=f"LLM synthesis failed: {e}")

        elif task.tool_name == "cannot_answer_without_tools":
            self.logger.info(f"Handling special non-MCP task: {task_id_str} (Tool: {task.tool_name}) by echoing inputs.", event_type="TASK_EXEC_SPECIAL_ECHO")
            return ToolExecutionResult(
                task_id=task_id_str,
                status="success",
                output=task.inputs
            )
        
        if not task.server_id or not task.tool_name:
            error_msg = f"TaskNode {task_id_str} (an MCP task) is missing a valid server_id or tool_name. ServerID: '{task.server_id}', ToolName: '{task.tool_name}'."
            self.logger.error(error_msg, event_type="TASK_EXEC_ERROR_PREFLIGHT")
            return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)
        
        selected_mcp_client = self.mcp_clients.get(task.server_id)
        if not selected_mcp_client:
            error_msg = f"No McpClient found for server_id: {task.server_id} required by task {task_id_str}. Available clients: {list(self.mcp_clients.keys())}"
            self.logger.error(error_msg, event_type="TASK_EXEC_ERROR_MCP_CLIENT_MISSING")
            return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)

        tool_id_for_mcp_call = task.tool_name 

        resolved_inputs: Dict[str, Any] = {}
        try:
            resolved_inputs = self._resolve_inputs(task, task_results)
        except ValueError as e_resolve:
            error_msg = f"Input resolution failed for task {task_id_str} (tool '{task.server_id}/{task.tool_name}'): {e_resolve}"
            self.logger.error(error_msg, event_type="TASK_EXEC_ERROR_INPUT_RESOLUTION", metadata={'task_id': task_id_str, 'error': str(e_resolve)})
            return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)

        last_exception_in_retry_loop: Exception | None = None
        for attempt_num in range(self.max_retries + 1):
            try:
                title_str = task.title if task.title else 'N/A'
                self.logger.info(f"Attempt {attempt_num + 1}/{self.max_retries + 1} for MCP Tool: '{task.server_id}/{tool_id_for_mcp_call}' for task {task_id_str} (Title: {title_str}) with inputs: {resolved_inputs}", event_type="TASK_EXEC_ATTEMPT")
                
                if hasattr(selected_mcp_client, 'initialize') and asyncio.iscoroutinefunction(selected_mcp_client.initialize) and not selected_mcp_client._is_initialized:
                    self.logger.info(f"Initializing MCP client for {task.server_id} before calling tool.", event_type="MCP_CLIENT_LAZY_INIT")
                    await selected_mcp_client.initialize()

                raw_tool_output = await selected_mcp_client.call_tool(
                    task_id_for_result=task.id,
                    tool_id=str(tool_id_for_mcp_call), 
                    inputs=resolved_inputs
                )
                
                self.logger.info(f"Task {task_id_str} ('{task.server_id}/{tool_id_for_mcp_call}') succeeded on attempt {attempt_num + 1}.", event_type="TASK_EXEC_SUCCESS_ATTEMPT", metadata={'output_type': str(type(raw_tool_output))})
                return raw_tool_output

            except (McpConnectionError, McpTimeoutError) as e_retryable:
                last_exception_in_retry_loop = e_retryable
                self.logger.warning(f"Attempt {attempt_num + 1}/{self.max_retries + 1} failed for task {task_id_str} ('{task.server_id}/{tool_id_for_mcp_call}'): {type(e_retryable).__name__}: {e_retryable}", event_type="TASK_EXEC_RETRYABLE_ERROR")
                if attempt_num < self.max_retries:
                    delay = self.initial_retry_delay * (2 ** attempt_num)
                    self.logger.info(f"Retrying task {task_id_str} in {delay:.2f} seconds...", event_type="TASK_EXEC_RETRY")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed for task {task_id_str} ('{task.server_id}/{tool_id_for_mcp_call}') due to retryable errors. Last error: {last_exception_in_retry_loop}", event_type="TASK_EXEC_MAX_RETRIES_REACHED")
                    break 
            
            except McpNotInitializedError as e_fatal_mcp: 
                error_msg = f"MCP Client for server '{task.server_id}' not initialized before calling tool '{tool_id_for_mcp_call}', task {task_id_str} (Attempt {attempt_num + 1})."
                self.logger.error(f"{error_msg} Details: {e_fatal_mcp}", event_type="TASK_EXEC_ERROR_MCP_NOT_INIT")
                return ToolExecutionResult(task_id=task_id_str, status="failure", error=error_msg)
            
            except McpError as e_other_mcp: 
                last_exception_in_retry_loop = e_other_mcp 
                self.logger.error(f"MCP Error (non-retryable) on attempt {attempt_num + 1} for task {task_id_str} ('{task.server_id}/{tool_id_for_mcp_call}'): {type(e_other_mcp).__name__}: {e_other_mcp}", event_type="TASK_EXEC_ERROR_MCP_OTHER")
                break 

            except Exception as e_unexpected_in_loop: 
                last_exception_in_retry_loop = e_unexpected_in_loop
                self.logger.error(f"Unexpected error on attempt {attempt_num + 1} for task {task_id_str} ('{task.server_id}/{tool_id_for_mcp_call}'): {type(e_unexpected_in_loop).__name__}: {e_unexpected_in_loop}", exc_info=True, event_type="TASK_EXEC_ERROR_UNEXPECTED")
                break 
        
        final_error_message = ""
        if last_exception_in_retry_loop:
            if isinstance(last_exception_in_retry_loop, (McpConnectionError, McpTimeoutError)):
                final_error_message = f"All {self.max_retries + 1} attempts failed for tool '{task.server_id}/{tool_id_for_mcp_call}', task {task_id_str}. Last error: {type(last_exception_in_retry_loop).__name__}: {last_exception_in_retry_loop}"
            else: 
                final_error_message = f"Failed tool '{task.server_id}/{tool_id_for_mcp_call}', task {task_id_str}, due to {type(last_exception_in_retry_loop).__name__}: {last_exception_in_retry_loop}"
        else:
            final_error_message = f"Tool '{task.server_id}/{tool_id_for_mcp_call}', task {task_id_str}, failed after retry attempts without a specific final exception recorded."

        self.logger.error(f"Final error for task {task_id_str}: {final_error_message}", event_type="TASK_EXEC_FAILURE_FINAL")
        return ToolExecutionResult(task_id=task_id_str, status="failure", error=final_error_message)

    def _resolve_inputs(self, task: TaskNode, task_results: Dict[str, ToolExecutionResult]) -> Dict[str, Any]:
        """Resolves task inputs, handling references to other task outputs recursively."""
        self.logger.debug(f"[_resolve_inputs for task {task.id}] Starting resolution. Inputs: {task.inputs}", metadata={'task_id': task.id, 'current_inputs': task.inputs})

        memo: Dict[Any, Any] = {} # Memoization cache for already resolved values

        def _access_value_at_path(data: Any, path_expression: str, task_id_for_logging: str, ref_task_id_for_logging: str) -> Any:
            """Helper to access a value in nested data using a dot-separated path."""
            accessed_value = data
            current_path_segment = ""
            for part_accessor in path_expression.split('.'):
                current_path_segment = f"{current_path_segment}.{part_accessor}" if current_path_segment else part_accessor
                accessor_description = f"part '{part_accessor}' in path '{path_expression}' (full path: {current_path_segment})"
                
                if isinstance(accessed_value, list):
                    try:
                        idx = int(part_accessor)
                        if not (0 <= idx < len(accessed_value)):
                            self.logger.warning(f"Index {idx} out of bounds for list (len {len(accessed_value)}) when accessing {accessor_description} for referenced task '{ref_task_id_for_logging}'", metadata={'task_id': task_id_for_logging, 'referenced_task_id': ref_task_id_for_logging})
                            raise ValueError(f"Index '{part_accessor}' out of bounds for list path '{path_expression}' in output of task '{ref_task_id_for_logging}'. List size: {len(accessed_value)}.")
                        accessed_value = accessed_value[idx]
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid list index '{part_accessor}' when accessing {accessor_description} for referenced task '{ref_task_id_for_logging}'", metadata={'task_id': task_id_for_logging, 'referenced_task_id': ref_task_id_for_logging})
                        raise ValueError(f"Invalid list index '{part_accessor}' for path '{path_expression}' in output of task '{ref_task_id_for_logging}'.")
                elif isinstance(accessed_value, dict):
                    try:
                        accessed_value = accessed_value[part_accessor]
                    except KeyError:
                        self.logger.warning(f"Key '{part_accessor}' not found when accessing {accessor_description} for referenced task '{ref_task_id_for_logging}'", metadata={'task_id': task_id_for_logging, 'referenced_task_id': ref_task_id_for_logging})
                        raise ValueError(f"Key '{part_accessor}' not found for path '{path_expression}' in output of task '{ref_task_id_for_logging}'. Available keys: {list(accessed_value.keys())}")
                else:
                    self.logger.warning(f"Cannot access {accessor_description} for referenced task '{ref_task_id_for_logging}' because current data is not a list or dict (type: {type(accessed_value)})", metadata={'task_id': task_id_for_logging, 'referenced_task_id': ref_task_id_for_logging})
                    raise ValueError(f"Cannot access path '{path_expression}' in non-dict/list data (found type {type(accessed_value)}) from task '{ref_task_id_for_logging}'.")
            return accessed_value

        def _resolve_placeholder_string_reference(placeholder_str: str, current_task_id_for_logging: str) -> Any:
            """Resolves a direct placeholder string like '$result.task_id.path'."""
            self.logger.debug(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Resolving direct placeholder: {placeholder_str}", metadata={'task_id': current_task_id_for_logging, 'placeholder': placeholder_str})
            
            parts = placeholder_str[len("$result."):].split(".", 1)
            ref_task_id_str = parts[0]

            if not ref_task_id_str: 
                self.logger.error(f"Invalid placeholder format: missing task_id in '{placeholder_str}' for task {current_task_id_for_logging}", metadata={'task_id': current_task_id_for_logging})
                raise ValueError(f"Invalid placeholder format: missing task_id in '{placeholder_str}' for task '{current_task_id_for_logging}'.")

            self.logger.debug(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Parsed ref_task_id: '{ref_task_id_str}', remaining_path_parts: '{parts[1:] if len(parts) > 1 else None}'", metadata={'task_id': current_task_id_for_logging, 'ref_task_id': ref_task_id_str})

            data_to_path: Any = None
            source_description: str = "" # For logging

            if ref_task_id_str in task_results:
                source_result_model = task_results[ref_task_id_str]
                source_description = f"local task_results (status: {source_result_model.status})"
                self.logger.debug(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Found source task '{ref_task_id_str}' in local task_results. Status: {source_result_model.status}, Output type: {type(source_result_model.output)}, Output (first 100 chars): {str(source_result_model.output)[:100]}", metadata={'task_id': current_task_id_for_logging, 'source_task_id': ref_task_id_str, 'source_status': source_result_model.status})
                
                if source_result_model.status == "failure":
                    self.logger.error(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Referenced local task '{ref_task_id_str}' failed. Error: {source_result_model.error}", metadata={'task_id': current_task_id_for_logging, 'source_task_id': ref_task_id_str, 'source_error': source_result_model.error})
                    raise ValueError(f"Referenced local task '{ref_task_id_str}' (for task '{current_task_id_for_logging}') failed and cannot be used as input. Error: {source_result_model.error}")
                data_to_path = source_result_model.output
            elif self.global_prior_phase_outputs and ref_task_id_str in self.global_prior_phase_outputs:
                source_description = "global_prior_phase_outputs"
                self.logger.debug(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Found source task '{ref_task_id_str}' in global_prior_phase_outputs. Output type: {type(self.global_prior_phase_outputs[ref_task_id_str])}, Output (first 100 chars): {str(self.global_prior_phase_outputs[ref_task_id_str])[:100]}", metadata={'task_id': current_task_id_for_logging, 'source_task_id': ref_task_id_str, 'source_location': 'global'})
                data_to_path = self.global_prior_phase_outputs[ref_task_id_str]
                # Assuming outputs from prior phases are inherently from successful tasks. No explicit status check here.
            else:
                available_local_keys = list(task_results.keys())
                available_global_keys = list(self.global_prior_phase_outputs.keys()) if self.global_prior_phase_outputs else []
                self.logger.error(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Referenced task '{ref_task_id_str}' not in local task_results or global_prior_phase_outputs. Available local: {available_local_keys}. Available global: {available_global_keys}", metadata={'task_id': current_task_id_for_logging, 'ref_task_id': ref_task_id_str, 'available_local_tasks': available_local_keys, 'available_global_tasks': available_global_keys})
                raise ValueError(f"Referenced task '{ref_task_id_str}' has no result available in local task_results or global_prior_phase_outputs for task '{current_task_id_for_logging}'. Available local: {available_local_keys}, Available global: {available_global_keys}")

            if len(parts) > 1 and parts[1]: # Path expression exists
                path_expression = parts[1]
                try:
                    accessed_value = _access_value_at_path(data_to_path, path_expression, current_task_id_for_logging, ref_task_id_str)
                    self.logger.debug(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Successfully accessed path '{path_expression}' in output of task '{ref_task_id_str}' (from {source_description}). Value: {str(accessed_value)[:100]}", metadata={'task_id': current_task_id_for_logging, 'ref_task_id': ref_task_id_str})
                    return accessed_value
                except Exception as e:
                    self.logger.error(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Failed to resolve path '{path_expression}' in output of task '{ref_task_id_str}' (from {source_description}). Error: {e}", exc_info=True, metadata={'task_id': current_task_id_for_logging, 'ref_task_id': ref_task_id_str, 'path_expression': path_expression})
                    raise ValueError(f"Failed to resolve path '{path_expression}' in output of task '{ref_task_id_str}' (from {source_description}) for task '{current_task_id_for_logging}'. Error: {e}")
            else: # No path, use the entire output
                self.logger.debug(f"[_resolve_placeholder_string_reference for task {current_task_id_for_logging}] Using entire output of task '{ref_task_id_str}' (from {source_description}). Value: {str(data_to_path)[:100]}", metadata={'task_id': current_task_id_for_logging, 'ref_task_id': ref_task_id_str})
                return data_to_path

        def resolve_value(value: Any, current_path_context: str = "") -> Any:
            # Check memoization cache first
            if isinstance(value, (str, int, float, bool)) or value is None: # Simple hashable types
                if value in memo:
                    self.logger.debug(f"[resolve_value for task {task.id}, path_ctx: '{current_path_context}'] Cache hit for simple value: {str(value)[:50]}", metadata={'task_id': task.id})
                    return memo[value]
            # For complex types, could use id(value) if careful about mutability, but might be risky.
            # Sticking to simple types for memo keys for now.

            self.logger.debug(f"[resolve_value for task {task.id}, path_ctx: '{current_path_context}'] Received value: {type(value)} {str(value)[:100]}", metadata={'task_id': task.id})
            
            original_value = value # Keep a reference to the original value for memoization key

            if isinstance(value, str):
                if value.startswith("$result."):
                    try:
                        resolved = _resolve_placeholder_string_reference(value, task.id)
                        # Recursively resolve the result of the placeholder
                        result = resolve_value(resolved, f"{current_path_context} (from direct ${value})")
                        memo[original_value] = result # Cache result for this placeholder string
                        return result
                    except Exception as e:
                        self.logger.error(f"[resolve_value for task {task.id}] Error resolving direct placeholder '{value}': {e}", metadata={'task_id': task.id, 'placeholder': value})
                        raise # Re-raise to allow DAG repair or task failure
                else:
                    # Attempt to find and replace embedded placeholders
                    # Regex to find $result.task_id or $result.task_id.path.attribute etc.
                    # Task ID: alphanumeric, underscore, hyphen. Path: same, plus dot, plus square brackets for indices.
                    pattern = re.compile(r'(\$result\.([a-zA-Z0-9_-]+(?:\[\d+\])?(?:(?:\.(?:[a-zA-Z0-9_-]+(?:\[\d+\])?))+)?))')
                                        
                    processed_string = value
                    made_change_in_iteration = True # Flag to loop if changes are made
                    MAX_EMBEDDED_PASSES = 5 # Safety limit for passes over the string
                    passes = 0

                    while made_change_in_iteration and passes < MAX_EMBEDDED_PASSES:
                        passes += 1
                        made_change_in_iteration = False
                        temp_string_after_pass = ""
                        current_search_offset = 0

                        for match in pattern.finditer(processed_string):
                            placeholder_str = match.group(1)
                            self.logger.debug(f"[resolve_value for task {task.id}] Found embedded placeholder '{placeholder_str}' in string '{processed_string}'", metadata={'task_id': task.id})
                            
                            # Add the part of the string before the match
                            temp_string_after_pass += processed_string[current_search_offset:match.start()]
                            
                            try:
                                resolved_segment = _resolve_placeholder_string_reference(placeholder_str, task.id)
                                
                                if isinstance(resolved_segment, (dict, list)):
                                    # Using str() for simplicity for textual embedding.
                                    # If JSON representation is strictly needed: segment_str_representation = json.dumps(resolved_segment)
                                    # However, if json.dumps is used, ensure resulting string is suitable for text prompt.
                                    segment_str_representation = str(resolved_segment) 
                                else:
                                    segment_str_representation = str(resolved_segment)
                                
                                temp_string_after_pass += segment_str_representation
                                made_change_in_iteration = True
                            except ValueError as e: # Failed to resolve this specific embedded placeholder
                                self.logger.warning(f"[resolve_value for task {task.id}] Failed to resolve embedded placeholder '{placeholder_str}' within string. Error: {e}. Leaving it as is.", metadata={'task_id': task.id, 'placeholder': placeholder_str})
                                temp_string_after_pass += placeholder_str # Keep original placeholder
                            
                            current_search_offset = match.end()
                        
                        temp_string_after_pass += processed_string[current_search_offset:] # Add the remainder of the string
                        processed_string = temp_string_after_pass
                        
                        if not made_change_in_iteration: # No changes in this pass, break
                            break

                    if processed_string != value: # If string was modified by embedded replacements
                        # Recursively call resolve_value on the new string, as it might contain
                        # further placeholders (e.g., if a resolved segment was "$result.another_task")
                        # or if the string itself became a direct placeholder after replacement.
                        result = resolve_value(processed_string, f"{current_path_context} (from embedded in '{value[:50]}...')")
                        memo[original_value] = result # Cache result for the original un-embedded-processed string
                        return result
                    else:
                        memo[original_value] = processed_string # Cache
                        return processed_string # No direct placeholder, and no embedded placeholders found/changed

            elif isinstance(value, dict):
                new_dict = {k: resolve_value(v, f"{current_path_context}.{k}" if current_path_context else k) for k, v in value.items()}
                return new_dict
            elif isinstance(value, list):
                new_list = [resolve_value(v, f"{current_path_context}[{i}]") for i, v in enumerate(value)]
                return new_list
            
            # Base case: value is not a string, dict, or list, or is a string already processed.
            memo[original_value] = value # Cache non-string/dict/list or fully processed string
            return value

        # Initial call to resolve_value for the entire inputs dictionary
        # Ensure task.inputs is a dict before processing. Handle None or other types gracefully.
        current_task_inputs = task.inputs
        if not isinstance(current_task_inputs, dict):
            self.logger.warning(f"[_resolve_inputs for task {task.id}] Task inputs are not a dictionary (type: {type(current_task_inputs)}). Value: {str(current_task_inputs)[:100]}. Proceeding with it as a single value to resolve or returning empty dict if resolution fails or is not applicable.", metadata={'task_id': task.id})
            # If inputs themselves might be a placeholder string or need resolution, try resolving current_task_inputs directly.
            # However, the method signature expects to return Dict[str, Any].
            # For now, if it's not a dict, we might consider it an error or return empty, 
            # unless specific handling for non-dict inputs is defined.
            # Sticking to the expectation that inputs are usually dicts.
            if current_task_inputs is None: 
                current_task_inputs = {} # Treat None as empty dict
            else:
                # If it's a non-dict, non-None type, it's problematic for this function's general structure.
                # For safety, log and return empty dict, or raise an error.
                # Let's raise an error if it's not a dict and not None, as the rest of the logic assumes a dict.
                raise ValueError(f"Task inputs for task {task.id} must be a dictionary or None, but got {type(current_task_inputs)}")

        resolved_inputs = resolve_value(current_task_inputs)
        self.logger.debug(f"[_resolve_inputs for task {task.id}] Finished resolution. Resolved inputs: {resolved_inputs}", metadata={'task_id': task.id, 'resolved_inputs': resolved_inputs})
        return resolved_inputs if isinstance(resolved_inputs, dict) else {}

class TaskFetchingUnit:
    def __init__(self, mcp_clients: Dict[str, McpClient], 
                synthesis_llm: Optional[BaseLanguageModel] = None,
                max_parallel_tasks: int = 3,
                task_max_retries: int = 2, 
                task_initial_retry_delay: float = 1.0, 
                settings: Optional[MochiWorkerConfig] = None,
                logger_instance: Optional[MochiLogger] = None):
        self.mcp_clients = mcp_clients 
        self.synthesis_llm = synthesis_llm
        self.current_settings = settings if settings else get_settings()
        self.logger: MochiLogger = logger_instance or MochiLogger(config=self.current_settings.logging, name=f"mochi.{self.__class__.__name__}")
        
        self.task_executor = TaskExecutor(
            mcp_clients=self.mcp_clients, 
            synthesis_llm=self.synthesis_llm,
            default_max_retries=task_max_retries,
            default_initial_retry_delay=task_initial_retry_delay,
            settings=self.current_settings,
            logger_instance=self.logger # Pass the TFU logger to the executor
        )
        self.max_parallel_tasks = max_parallel_tasks
        self.task_max_retries = task_max_retries # Stored for potential future use, though executor handles its own retries
        self.task_initial_retry_delay = task_initial_retry_delay # Stored for potential future use
        self.semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        
        self.logger.info(f"TaskFetchingUnit initialized. Max parallel tasks: {self.max_parallel_tasks}, Task max retries: {self.task_max_retries}, Task initial retry delay: {self.task_initial_retry_delay}", event_type="TFU_INIT_COMPLETE")

    # Graph-compatible wrapper for process_dag
    async def process_dag_graph_compatible(self, state: dict) -> dict:
        """
        Graph-compatible wrapper for process_dag.
        Accepts a dictionary state, converts to AgentState, calls process_dag,
        and returns the updated state as a dictionary.
        """
        self.logger.info("[TFU.process_dag_graph_compatible] Received state for DAG processing.", 
                         event_type="TFU_GRAPH_WRAPPER_START", 
                         metadata={'input_state_keys': list(state.keys())})

        # Convert dict to AgentState Pydantic model
        try:
            agent_state_model = AgentState(**state)
            self.logger.debug(f"[TFU.process_dag_graph_compatible] Successfully converted input dict to AgentState model. Overall status: {agent_state_model.overall_status}",
                               metadata={'agent_overall_status': agent_state_model.overall_status})
        except Exception as e:
            self.logger.error(f"[TFU.process_dag_graph_compatible] Error converting input dict to AgentState model: {e}", exc_info=True, event_type="TFU_GRAPH_STATE_CONVERSION_ERROR")
            # Update state with error and return as dict
            state["error_message"] = f"Error converting state to AgentState model in TFU: {e}"
            state["overall_status"] = "error" # Ensure overall_status is set
            return state

        if not agent_state_model.task_dag:
            self.logger.warning("[TFU.process_dag_graph_compatible] No TaskDAG found in agent_state. Returning state as is.", 
                                event_type="TFU_GRAPH_WRAPPER_NO_DAG")
            return agent_state_model.model_dump(exclude_none=True) # Return as dict

        # Call the main processing logic
        # Pass the global accumulated outputs from the agent_state_model to the executor
        self.task_executor.global_prior_phase_outputs = agent_state_model.accumulated_global_task_outputs
        
        updated_agent_state_model = await self.process_dag(agent_state_model.task_dag, agent_state_model)
        
        self.logger.info("[TFU.process_dag_graph_compatible] DAG processing complete. Converting AgentState back to dict.", 
                         event_type="TFU_GRAPH_WRAPPER_COMPLETE",
                         metadata={'output_agent_status': updated_agent_state_model.overall_status})
        
        # Convert AgentState Pydantic model back to dict for LangGraph
        return updated_agent_state_model.model_dump(exclude_none=True)

    def _has_cycle_util(self, task_id: str, tasks_map: Dict[str, TaskNode], task_status_dict: Dict[str, str],
                        visited: Set[str], recursion_stack: Set[str]) -> bool:
        visited.add(task_id)
        recursion_stack.add(task_id)

        task = tasks_map.get(task_id)
        if not task: 
            recursion_stack.remove(task_id)
            return False

        for dep_id in task.dependencies:
            if task_status_dict.get(dep_id) != "completed":
                if dep_id not in visited:
                    if self._has_cycle_util(dep_id, tasks_map, task_status_dict, visited, recursion_stack):
                        return True
                elif dep_id in recursion_stack:
                    return True 

        recursion_stack.remove(task_id)
        return False

    def _detect_cycle_in_pending(self, tasks_map: Dict[str, TaskNode], task_status_dict: Dict[str, str]) -> bool:
        pending_task_ids = [str(tid) for tid, status in task_status_dict.items() if status == 'pending']
        if not pending_task_ids:
            return False

        visited_dfs = set()
        recursion_stack_dfs = set()

        for task_id_str in pending_task_ids:
            if task_id_str not in visited_dfs:
                if self._has_cycle_util(task_id_str, tasks_map, task_status_dict, visited_dfs, recursion_stack_dfs):
                    return True 
        return False
    
    async def _execute_task_wrapper(self, task_node: TaskNode, agent_state: AgentState):
        """Wraps task execution, updates state, and manages semaphore."""
        task_id_str = task_node.id
        
        # Ensure agent_state dictionaries exist, though process_dag should have initialized them.
        task_statuses_dict = agent_state.task_statuses
        task_results_dict = agent_state.task_results

        try:
            self.logger.debug(f"_execute_task_wrapper starting for task {task_id_str} (Status: {task_node.status}, Attempts: {task_node.execution_attempts})", event_type="TFU_WRAPPER_START")
            
            result: ToolExecutionResult = await self.task_executor.execute_task(task_node, task_results_dict)
            
            # Update TaskNode object with results
            task_node.raw_output = result
            if result.error:
                task_node.error = str(result.error) # Ensure error is a string
            else: 
                task_node.error = None # Clear any previous error if success

            if result.status == "success":
                task_node.status = "completed_success"
                self.logger.info(f"Task {task_id_str} completed successfully. Output type: {type(result.output).__name__}", event_type="TFU_TASK_SUCCESS_WRAPPED")
            else: # failure or other non-success status from executor
                task_node.status = "completed_failure"
                self.logger.error(f"Task {task_id_str} failed. Error: {result.error}", event_type="TFU_TASK_FAILURE_WRAPPED")

            # Sync TaskNode status and result back to agent_state dictionaries
            task_statuses_dict[task_id_str] = task_node.status
            task_results_dict[task_id_str] = result
            
            agent_state.execution_error = None # Clear general execution error if this task (part of a sequence) was fine

        except Exception as e:
            self.logger.error(f"Unhandled error during _execute_task_wrapper for task {task_id_str}: {e}", exc_info=True, event_type="TFU_WRAPPER_ERROR")
            # Update TaskNode object for wrapper-level failure
            task_node.status = "completed_failure"
            task_node.error = f"Task execution wrapper error: {str(e)}"
            
            # Sync to agent_state dictionaries
            task_statuses_dict[task_id_str] = "completed_failure"
            task_results_dict[task_id_str] = ToolExecutionResult(task_id=task_id_str, status="completed_failure", error=task_node.error)
            agent_state.execution_error = task_node.error # Set general execution error
        finally:
            self.semaphore.release()
            self.logger.debug(f"Semaphore released for task {task_id_str}", event_type="TFU_SEMAPHORE_RELEASE")
    
    async def process_dag(self, dag: TaskDAG, agent_state: AgentState) -> AgentState:
        """
        Processes a DAG by fetching and executing tasks based on dependencies and status.
        Updates agent_state with task results, statuses, and potentially a final response.
        """
        start_time = asyncio.get_event_loop().time()
        self.logger.info(f"[TFU.process_dag] Starting DAG processing. Tasks: {len(dag.tasks)}", 
                         event_type="TFU_PROCESS_DAG_START", 
                         metadata={'num_tasks': len(dag.tasks), 'agent_status': agent_state.overall_status})

        # Initialize or verify statuses and attempts on the input DAG's TaskNodes
        for task_node in dag.tasks:
            if not hasattr(task_node, 'status') or not task_node.status:
                task_node.status = "pending"
            if not hasattr(task_node, 'execution_attempts') or task_node.execution_attempts is None:
                task_node.execution_attempts = 0
        
        # Update agent_state with the (potentially modified) DAG
        agent_state.task_dag = dag
        agent_state.task_statuses = {task.id: task.status for task in dag.tasks}
        agent_state.task_results = agent_state.task_results or {}
        agent_state.all_completed = False
        agent_state.has_ready_tasks = False
        agent_state.execution_error = None
        agent_state.final_response = None # Clear previous final response for this execution cycle

        # --- Main task processing loop ---
        tasks_map: Dict[str, TaskNode] = {tn.id: tn for tn in dag.tasks}
        current_task_statuses = agent_state.task_statuses
        current_task_results = agent_state.task_results

        active_asyncio_tasks: Set[asyncio.Task] = set()
        launched_this_run_ids: Set[str] = set() # Tracks tasks launched in the current process_dag invocation

        while True: 
            ready_to_launch_nodes: List[TaskNode] = []
            for task_id, task_node in tasks_map.items():
                # Use task_node.status as the source of truth, ensure current_task_statuses is consistent
                # Condition 1: Task is pending and its dependencies are met
                if task_node.status == "pending" and task_id not in launched_this_run_ids:
                    dependencies = task_node.dependencies
                    # Check dependencies based on the updated completed_success status on TaskNode or current_task_statuses
                    # Use current_task_statuses for dep checking as it reflects execution results more directly during the loop
                    if not dependencies or all(current_task_statuses.get(dep_id) == "completed_success" for dep_id in dependencies):
                        # MODIFICATION: Update TaskNode.status to ready_to_run
                        task_node.status = "ready_to_run"
                        current_task_statuses[task_id] = "ready_to_run" # Keep dict in sync
                        ready_to_launch_nodes.append(task_node)
                # Condition 2: Task is already ready_to_run but not yet launched (e.g., deferred due to parallelism)
                elif task_node.status == "ready_to_run" and task_id not in launched_this_run_ids:
                    ready_to_launch_nodes.append(task_node)
            
            # Sort by priority (existing logic)
            ready_to_launch_nodes.sort(key=lambda t: PRIORITY_ORDER.get(t.priority, PRIORITY_ORDER["medium"]))

            for task_node_to_launch in ready_to_launch_nodes:
                if len(active_asyncio_tasks) >= self.max_parallel_tasks:
                    self.logger.debug(f"Max parallel tasks ({self.max_parallel_tasks}) reached. Will not launch {task_node_to_launch.id} in this iteration.", event_type="TFU_MAX_PARALLEL_HIT")
                    break

                if self.semaphore.locked() and len(active_asyncio_tasks) >= self.max_parallel_tasks :
                    self.logger.debug(f"Semaphore locked and active tasks at limit. Deferring {task_node_to_launch.id}.", event_type="TFU_SEMAPHORE_LOCKED_DEFER")
                    break

                await self.semaphore.acquire()
                
                # MODIFICATION: Update TaskNode.status to in_progress and increment attempts
                task_node_to_launch.status = "in_progress"
                task_node_to_launch.execution_attempts += 1
                current_task_statuses[task_node_to_launch.id] = "in_progress" # Keep dict in sync
                launched_this_run_ids.add(task_node_to_launch.id)

                self.logger.info(f"Launching task {task_node_to_launch.id} (Tool: {task_node_to_launch.tool_name}, Attempt: {task_node_to_launch.execution_attempts}). Active asyncio tasks: {len(active_asyncio_tasks) + 1}", event_type="TFU_LAUNCHING_TASK")
                
                new_task = asyncio.create_task(
                    self._execute_task_wrapper(task_node_to_launch, agent_state) # Pass the task_node object itself
                )
                active_asyncio_tasks.add(new_task)

            if not active_asyncio_tasks and not ready_to_launch_nodes:
                final_check_ready_nodes: List[TaskNode] = []
                for task_id_check, task_node_check in tasks_map.items():
                    # Use task_node_check.status for truth
                    if task_node_check.status == "pending" and task_id_check not in launched_this_run_ids:
                        if (not task_node_check.dependencies or 
                           all(current_task_statuses.get(dep_id_check) == "completed_success" for dep_id_check in task_node_check.dependencies)):
                            # MODIFICATION: Update status before adding to final_check_ready_nodes
                            task_node_check.status = "ready_to_run" 
                            current_task_statuses[task_id_check] = "ready_to_run"
                            final_check_ready_nodes.append(task_node_check)
                
                if final_check_ready_nodes:
                    self.logger.info(f"Final check found {len(final_check_ready_nodes)} newly ready tasks. Continuing DAG processing.", event_type="TFU_FINAL_CHECK_CONTINUE")
                    agent_state.has_ready_tasks = True # Signal that ready tasks were found
                    continue # Re-loop to process these newly ready tasks

                self.logger.info("No active tasks and no newly ready tasks found after final check. Determining DAG outcome.", event_type="TFU_DETERMINING_OUTCOME")

                # Determine overall DAG status based on TaskNode.status values
                all_dag_tasks_completed_successfully = True
                any_failed = False
                for tn_id, tn_obj in tasks_map.items():
                    synced_status = current_task_statuses.get(tn_id, tn_obj.status) # Prefer dict if out of sync, but should be tn_obj.status
                    if synced_status != "completed_success":
                        all_dag_tasks_completed_successfully = False
                    if synced_status == "completed_failure":
                        any_failed = True
                
                if all_dag_tasks_completed_successfully:
                    self.logger.info("TaskFetchingUnit: All DAG tasks completed successfully.", event_type="TFU_ALL_COMPLETED")
                    agent_state.all_completed = True
                    agent_state.has_ready_tasks = False
                    agent_state.execution_error = None
                    # DAG object in agent_state already has updated TaskNodes
                    return agent_state 
                else:
                    # Cycle detection and failure reporting (existing logic, ensure it uses updated statuses)
                    if self._detect_cycle_in_pending(tasks_map, current_task_statuses): # Pass current_task_statuses which reflects node statuses
                        error_msg = "DAG execution stalled due to a cycle in dependencies among pending tasks."
                        self.logger.error(f"TaskFetchingUnit: {error_msg}", event_type="TFU_CYCLE_DETECTED")
                        agent_state.execution_error = error_msg
                    else:
                        failed_task_ids = [tn_id for tn_id, tn_obj in tasks_map.items() if current_task_statuses.get(tn_id, tn_obj.status) == "completed_failure"]
                        if failed_task_ids:
                            # Construct error message using TaskNode details if available
                            error_details_list = []
                            for ft_id in failed_task_ids:
                                ft_node = tasks_map.get(ft_id)
                                error_detail = ft_id
                                if ft_node and ft_node.error:
                                    error_detail += f" (error: {str(ft_node.error)[:100]}...)"
                                error_details_list.append(error_detail)
                            error_msg = f"DAG execution failed. Failed tasks: {', '.join(error_details_list)}."
                            self.logger.error(f"TaskFetchingUnit: {error_msg}", event_type="TFU_TASKS_FAILED")
                            # Append to existing execution_error if any
                            existing_err = agent_state.execution_error
                            agent_state.execution_error = (existing_err + "; " + error_msg) if existing_err else error_msg
                        else:
                            pending_tasks_count = sum(1 for tn_obj in tasks_map.values() if current_task_statuses.get(tn_obj.id, tn_obj.status) == "pending")
                            running_tasks_count = sum(1 for tn_obj in tasks_map.values() if current_task_statuses.get(tn_obj.id, tn_obj.status) == "in_progress") # Changed from "running"
                            ready_tasks_count = sum(1 for tn_obj in tasks_map.values() if current_task_statuses.get(tn_obj.id, tn_obj.status) == "ready_to_run")

                            error_msg = f"DAG execution stalled: Not all tasks completed. Pending: {pending_tasks_count}, Ready: {ready_tasks_count}, InProgress: {running_tasks_count}."
                            self.logger.error(f"TaskFetchingUnit: {error_msg}", event_type="TFU_STALLED_UNKNOWN")
                            existing_err = agent_state.execution_error
                            agent_state.execution_error = (existing_err + "; " + error_msg) if existing_err else error_msg
                    
                    agent_state.all_completed = False
                    agent_state.has_ready_tasks = False # No more tasks can be run if stalled or failed like this
                    # DAG object in agent_state already has updated TaskNodes
                    return agent_state

            if not active_asyncio_tasks:
                agent_state.has_ready_tasks = bool(ready_to_launch_nodes)
                await asyncio.sleep(0.01) 
                continue

            self.logger.debug(f"Waiting for one of {len(active_asyncio_tasks)} active tasks to complete.", event_type="TFU_WAITING_TASKS")
            done, pending = await asyncio.wait(active_asyncio_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task_just_done in done:
                try:
                    await task_just_done
                except asyncio.CancelledError:
                    self.logger.warning(f"An asyncio task was cancelled during TFU processing.", event_type="TFU_ASYNC_TASK_CANCELLED")
                except Exception as e_async_task_unhandled:
                    self.logger.error(f"Asyncio task wrapper encountered an unhandled exception: {e_async_task_unhandled}", exc_info=True, event_type="TFU_ASYNC_TASK_UNHANDLED_ERROR")
            
            active_asyncio_tasks = pending
            self.logger.debug(f"{len(done)} task(s) completed. {len(active_asyncio_tasks)} task(s) still active.", event_type="TFU_TASK_COMPLETION_UPDATE")

            potential_ready_exist = False
            for task_id_potential, task_node_potential in tasks_map.items():
                if current_task_statuses.get(task_id_potential) == "pending" and task_id_potential not in launched_this_run_ids:
                    if (not task_node_potential.dependencies or 
                       all(current_task_statuses.get(dep_id_potential) == "completed_success" for dep_id_potential in task_node_potential.dependencies)):
                        potential_ready_exist = True
                        break
            agent_state.has_ready_tasks = potential_ready_exist or bool(active_asyncio_tasks)

class TaskExecutorService:
    """Manages the execution of tasks using MCP clients."""

    def __init__(self, mcp_clients: Dict[str, McpClient], logger_instance: Optional[MochiLogger] = None):
        self.mcp_clients = mcp_clients
        self.logger: MochiLogger = logger_instance or MochiLogger(config=get_settings().logging, name=f"mochi.{self.__class__.__name__}")
        self.settings = get_settings()
        
        self.default_timeout = self.settings.execution.default_task_execution_timeout_seconds
        self.min_timeout = self.settings.execution.min_task_execution_timeout_seconds
        self.max_timeout = self.settings.execution.max_task_execution_timeout_seconds
        self.logger.info(f"TaskExecutorService initialized with timeouts: Default={self.default_timeout}s, Min={self.min_timeout}s, Max={self.max_timeout}s")

    async def execute_task(
        self,
        task: TaskNode,
        task_inputs: Dict[str, Any],
        timeout_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Executes a single task using the appropriate MCP client with timeout and retries."""
        if not task.server_id or not task.tool_name:
            error_msg = f"TaskNode {task.id} is missing required server_id ('{task.server_id}') or tool_name ('{task.tool_name}')."
            self.logger.error(error_msg, event_type="TASK_EXEC_PREFLIGHT_ERROR")
            raise McpError(error_msg)
             
        client = self.mcp_clients.get(task.server_id)
        if not client:
            error_msg = f"No MCP client found for server_id: '{task.server_id}' required by task {task.id}. Available: {list(self.mcp_clients.keys())}"
            self.logger.error(error_msg, event_type="TASK_EXEC_MCP_CLIENT_MISSING")
            raise McpError(error_msg)

        effective_timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout
        clamped_timeout = max(self.min_timeout, min(effective_timeout, self.max_timeout))
        if clamped_timeout != effective_timeout:
            self.logger.warning(f"Task '{task.id}' requested timeout {effective_timeout}s was clamped to {clamped_timeout}s (Min: {self.min_timeout}s, Max: {self.max_timeout}s)")
        self.logger.debug(f"Executing task '{task.id}' with overall timeout {clamped_timeout}s. Client '{client.server_name}' internal retries/timeouts apply to underlying requests.")

        try:
            self.logger.info(f"Executing task '{task.id}' ({task.tool_name}) via server '{task.server_id}' with overall timeout {clamped_timeout}s.")
            
            result = await asyncio.wait_for(
                client.call_tool(
                    task_id_for_result=task.id,
                    tool_id=task.tool_name, 
                    inputs=task_inputs
                ),
                timeout=clamped_timeout
            )

            self.logger.info(f"Task '{task.id}' completed successfully.")
            
            if not isinstance(result, ToolExecutionResult):
                self.logger.warning(f"Unexpected result type from client.call_tool for task '{task.id}': {type(result)}. Expected ToolExecutionResult.")
                return ToolExecutionResult(task_id=task.id, status="success", output=result).model_dump()
            
            return result.model_dump() 

        except asyncio.TimeoutError:
            self.logger.error(f"Task '{task.id}' timed out after {clamped_timeout} seconds (overall execution).", event_type="TASK_TIMEOUT")
            raise McpTimeoutError(f"Task execution timed out after {clamped_timeout} seconds.") from None # Raise specific timeout error
        except McpError as e:
            self.logger.error(f"MCP Error executing task '{task.id}': {e}", event_type="TASK_EXECUTION_MCP_ERROR", exc_info=True)
            raise # Re-raise McpError
        except Exception as e:
            self.logger.error(f"Unexpected error executing task '{task.id}': {e}", event_type="TASK_EXECUTION_UNEXPECTED_ERROR", exc_info=True)
            raise McpError(f"Unexpected error during task execution: {e}") from e # Wrap unexpected errors
