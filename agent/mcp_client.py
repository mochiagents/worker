import asyncio
from typing import Optional, Dict, Any, Literal, List, Union, Callable
import asyncio.subprocess
import logging
import aiohttp
from worker.core.logging import MochiLogger
from worker.core.models import ToolExecutionResult, ToolSchema
from worker.config.models import MCPToolServerConfig
import json
import uuid
from httpx import AsyncClient, Timeout, HTTPStatusError, RequestError, ConnectTimeout, ReadTimeout, Response
import httpx
from ..core.models import StructuredError
from ..config import get_settings

class McpError(Exception):
    """Base exception for MCP client errors."""
    pass

class McpTimeoutError(McpError):
    """Raised when an MCP operation times out."""
    pass

class McpConnectionError(McpError):
    """Raised for issues related to connecting to the MCP server."""
    pass

class McpNotInitializedError(McpError):
    """Raised when an operation is attempted before the client is initialized."""
    pass


class McpClient:
    """
    A client for interacting with Model Context Protocol (MCP) servers.

    This client supports different transport layers (stdio, http+sse) and
    provides both synchronous and asynchronous methods for sending requests.
    """

    def __init__(
        self,
        server_name: str,
        server_address: str,
        logger_instance: Optional[MochiLogger] = None,
        transport_type: Literal["http+json", "http+sse"] = "http+json",
        http_client_instance: Optional[AsyncClient] = None,
        server_config_override: Optional[MCPToolServerConfig] = None,
    ):
        """
        Initializes the McpClient.

        Args:
            server_name: The logical name/identifier for this server instance (e.g., 'web_search_v1').
            server_address: The address of the MCP server.
                            For "http+sse", this is the URL (e.g., "http://localhost:8000/mcp").
                            For "stdio", this could be the command to start the server process.
            transport_type: The transport protocol to use ("stdio" or "http+sse").
            timeout_connect: Timeout in seconds for establishing a connection.
            timeout_request: Default timeout in seconds for individual requests if not overridden by server_config_override.
            default_capabilities: Optional dictionary of capabilities the client wishes to declare.
            logger_instance: An optional MochiLogger instance.
            server_config_override: Optional MCPToolServerConfig for this specific client instance.
                                      Its values (e.g., request_timeout_seconds, schema_discovery_url, mock_tool_schemas)
                                      can override client defaults or provide specific configurations for this server.
        """
        self.logger = logger_instance or MochiLogger(config=get_settings().logging)
        self.settings = get_settings()
        
        self.request_timeout = self.settings.mcp_client_defaults.default_request_timeout_seconds
        self.connect_timeout = self.settings.mcp_client_defaults.default_connect_timeout_seconds
        self.sse_heartbeat_timeout = self.settings.mcp_client_defaults.sse_heartbeat_timeout_seconds
        self.sse_max_retries = self.settings.mcp_client_defaults.sse_max_retries
        self.sse_retry_delay = self.settings.mcp_client_defaults.sse_retry_delay_seconds
        
        self.server_name = server_name
        self.server_address = server_address.rstrip("/")
        self.transport_type = transport_type
        self._is_initialized = False
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        self.openapi_spec: Optional[Dict[str, Any]] = None
        self.server_description: Optional[str] = None

        if server_config_override:
            self.logger.info(f"Applying server-specific overrides for MCP client: {server_name}")
            self.request_timeout = server_config_override.request_timeout_seconds or self.request_timeout
            self.max_retries = server_config_override.max_retries if server_config_override.max_retries is not None else self.settings.execution.max_retries
            self.retry_delay_seconds = server_config_override.retry_delay_seconds or self.settings.execution.initial_retry_delay_seconds
            self.server_description = server_config_override.description
            self.schema_discovery_url = server_config_override.schema_discovery_url
            self.auth_token = server_config_override.auth_token
            self.custom_headers = server_config_override.custom_headers or {}
            self.mock_tool_schemas = server_config_override.mock_tool_schemas
        else:
            self.max_retries = self.settings.execution.max_retries
            self.retry_delay_seconds = self.settings.execution.initial_retry_delay_seconds
            self.schema_discovery_url = None
            self.auth_token = None
            self.custom_headers = {}
            self.mock_tool_schemas = None

        self.server_config_override = server_config_override

        self._session = None
        self._process: Optional[Any] = None
        self._manages_http_client: bool = False
        
        self.logger.info(f"McpClient for '{self.server_name}' initialized. Address: {self.server_address}, Request Timeout: {self.request_timeout}, Connect Timeout: {self.connect_timeout}")

        if http_client_instance:
            self.http_client = http_client_instance
            self._manages_http_client = False
        else:
            timeout_config = httpx.Timeout(self.request_timeout, connect=self.connect_timeout)
            self.http_client = AsyncClient(timeout=timeout_config)
            self._manages_http_client = True

    async def initialize(self) -> None:
        """
        Initializes the client and establishes a connection to the MCP server.

        This method must be called before sending any requests.
        Actual connection logic will depend on the transport type and will be
        implemented in a later subtask (2.2).
        """
        if self._is_initialized:
            self.logger.info("Client is already initialized.")
            return

        self.logger.info(f"Attempting to initialize MCP client for {self.server_address} via {self.transport_type}...")

        if self.transport_type == "http+json":
            self.logger.info(f"HTTP+JSON client for {self.server_address} is ready (httpx client initialized).")
        elif self.transport_type == "http+sse":
            self.logger.info(f"HTTP+SSE client for {self.server_address} is ready (httpx client initialized). SSE handled by _handle_sse_connection.")
        elif self.transport_type == "stdio":
            try:
                self.logger.info(f"Attempting to start STDIO process with command: {self.server_address}")
                if not isinstance(self.server_address, str) or not self.server_address:
                    self._is_initialized = False
                    raise McpConnectionError("For STDIO transport, server_address must be a valid command string.")

                self._process = await asyncio.create_subprocess_shell(
                    self.server_address,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                self.logger.info(f"STDIO process started successfully with PID: {self._process.pid}.")
            except Exception as e:
                self.logger.error(f"Failed to start STDIO process with command '{self.server_address}': {e}", exc_info=True)
                self._is_initialized = False
                raise McpConnectionError(f"Failed to initialize STDIO process for {self.server_address}: {e}") from e
        else:
            self.logger.error(f"Unsupported transport type: {self.transport_type}")
            self._is_initialized = False
            raise ValueError(f"Unsupported transport type: {self.transport_type}")

        self._is_initialized = True
        try:
            await self.async_get_server_capabilities()
            self.logger.info(f"Successfully fetched and processed capabilities for {self.server_name} during initialization.")
        except Exception as e_caps:
            self.logger.warning(f"Failed to fetch/process capabilities for {self.server_name} during initialization: {e_caps}", exc_info=True)
        
        self.logger.info("MCP Client initialized.")

    async def closeGracefully(self) -> None:
        """
        Closes the connection to the MCP server gracefully and releases resources.
        """
        if not self._is_initialized and not self._session:
            self.logger.debug("Client not initialized or no active session to close.")
            return

        self.logger.info(f"Closing MCP client connection to {self.server_address}...")
        
        if self._manages_http_client and hasattr(self.http_client, 'aclose') and not self.http_client.is_closed:
             try:
                 await self.http_client.aclose()
                 self.logger.info(f"Managed httpx.AsyncClient for {self.server_address} closed.")
             except Exception as e:
                 self.logger.error(f"Error closing managed httpx.AsyncClient: {e}", exc_info=True)
        self._manages_http_client = False
        
        if self._process:
            if self._process.returncode is None:
                self.logger.info(f"Terminating STDIO process (PID: {self._process.pid})...")
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    self.logger.info(f"STDIO process (PID: {self._process.pid}) terminated with code {self._process.returncode}.")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout waiting for STDIO process (PID: {self._process.pid}) to terminate. Killing...")
                    try:
                        self._process.kill()
                        await self._process.wait()
                        self.logger.info(f"STDIO process (PID: {self._process.pid}) killed.")
                    except Exception as e_kill:
                        self.logger.error(f"Error killing STDIO process (PID: {self._process.pid}): {e_kill}", exc_info=True)
                except Exception as e_term:
                    self.logger.error(f"Error during STDIO process termination (PID: {self._process.pid}): {e_term}", exc_info=True)
            else:
                self.logger.info(f"STDIO process (PID: {self._process.pid}) already exited with code {self._process.returncode}.")
            self._process = None

        self._is_initialized = False
        self.logger.info("MCP Client closed.")

    def send_request_sync(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends a synchronous request to the MCP server.

        Note: This is a placeholder. True synchronous execution in an async environment
        often involves running the async version in a separate event loop or thread.
        For simplicity in this scaffolding phase, it will raise NotImplementedError.

        Args:
            method: The MCP method name.
            params: Optional dictionary of parameters for the method.

        Returns:
            The server's response as a dictionary.

        Raises:
            NotImplementedError: As this is a placeholder.
            McpNotInitializedError: If the client is not initialized.
        """
        if not self._is_initialized:
            self.logger.error("Client must be initialized before sending synchronous requests.")
            raise McpNotInitializedError("Client must be initialized before sending requests.")
        
        # For true sync, one might do:
        # try:
        #     loop = asyncio.get_event_loop()
        # except RuntimeError: # No current event loop
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     return loop.run_until_complete(self.send_request_async(method, params))
        # else:
        #     if loop.is_running():
        #          # This is more complex: need to run in a separate thread or use nest_asyncio
        #          raise McpError("Cannot run sync method from a running async event loop without special handling.")
        #     else:
        #          return loop.run_until_complete(self.send_request_async(method, params))
        self.logger.warning("Synchronous send_request_sync is not fully implemented yet.")
        raise NotImplementedError("Synchronous send_request_sync is not fully implemented yet.")

    async def send_request_async(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sends an asynchronous request to the MCP server.

        Args:
            method: The MCP method name.
            params: Optional dictionary of parameters for the method.

        Returns:
            The server's response as a dictionary.

        Raises:
            McpNotInitializedError: If the client is not initialized.
            McpConnectionError: If there's an issue sending the request or receiving a response.
            McpTimeoutError: If the request times out.
        """
        if not self._is_initialized:
            self.logger.error("Client must be initialized before sending async requests.")
            raise McpNotInitializedError("Client must be initialized before sending requests.")

        if params is None:
            params = {}

        headers = {}
        if self.server_config_override:
            if self.server_config_override.custom_headers:
                headers.update(self.server_config_override.custom_headers)
            if self.server_config_override.auth_token:
                if 'Authorization' in headers and headers['Authorization'] != f"Bearer {self.server_config_override.auth_token}":
                    self.logger.warning(f"'Authorization' header was in custom_headers and will be overwritten by auth_token for server {self.server_name}.")
                headers['Authorization'] = f"Bearer {self.server_config_override.auth_token}"

        self.logger.info(f"Sending async request: {method} with params: {params} and headers: {list(headers.keys())}")
        
        log_entry_response: Optional[Dict[str, Any]] = None
        log_entry_error: Optional[Exception] = None

        try:
            if self.transport_type == "http+json" or self.transport_type == "http+sse": 
                if self.http_client.is_closed:
                    self.logger.error("httpx.AsyncClient is closed. Cannot send HTTP request.")
                    current_error = McpConnectionError("HTTP client closed for RPC call.")
                    log_entry_error = current_error
                    raise current_error
                
                request_id = str(uuid.uuid4())
                payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": request_id
                }
                self.logger.debug(f"Sending JSON-RPC payload via httpx to {self.server_address}: {json.dumps(payload)}")

                response = await self.http_client.post(
                    self.server_address, 
                    json=payload, 
                    headers=headers 
                )
                
                try:
                    response_text = response.text
                    self.logger.debug(f"Received raw response from {self.server_address} (status: {response.status_code}): {response_text[:500]}...")
                    
                    response.raise_for_status()

                    response_json = response.json()
                    self.logger.debug(f"Parsed JSON-RPC response: {json.dumps(response_json)}")

                    if not isinstance(response_json, dict):
                        current_error = McpError(f"Invalid JSON-RPC response: not a dictionary. Response: {response_text[:200]}")
                        log_entry_error = current_error
                        raise current_error

                    if response_json.get("jsonrpc") != "2.0":
                        current_error = McpError(f"Invalid JSON-RPC version: {response_json.get('jsonrpc')}. Expected '2.0'. Response: {response_text[:200]}")
                        log_entry_error = current_error
                        raise current_error

                    if response_json.get("id") != request_id:
                        current_error = McpError(f"JSON-RPC ID mismatch: Expected {request_id}, got {response_json.get('id')}. Response: {response_text[:200]}")
                        log_entry_error = current_error
                        raise current_error

                    if "result" in response_json:
                        log_entry_response = response_json
                        return response_json
                    
                    elif "error" in response_json:
                        error_obj = response_json["error"]
                        err_msg = error_obj.get("message", "Unknown MCP error")
                        err_code = error_obj.get("code", "N/A")
                        err_data = error_obj.get("data")
                        self.logger.error(f"MCP server returned error: Code {err_code}, Message: {err_msg}, Data: {err_data}")
                        log_entry_response = response_json
                        current_error = McpError(f"MCP Error {err_code}: {err_msg} (Data: {err_data})")
                        log_entry_error = current_error
                        raise current_error
                    else:
                        current_error = McpError(f"Invalid JSON-RPC response: Missing 'result' or 'error' fields. Response: {response_text[:200]}")
                        log_entry_error = current_error
                        raise current_error
                            
                except json.JSONDecodeError as e_json:
                    self.logger.error(f"Failed to decode JSON response from {self.server_address} (status: {response.status_code}). Raw text: {response_text[:500]}... Error: {e_json}", exc_info=True)
                    current_error = McpError(f"JSON decode error: {e_json}. Response: {response_text[:200]}")
                    log_entry_error = current_error
                    raise current_error from e_json

            elif self.transport_type == "stdio":
                if not self._process or self._process.stdin is None or self._process.stdout is None:
                    self.logger.error("STDIO process or its streams are not available. Cannot send STDIO request.")
                    current_error = McpConnectionError("STDIO process/streams not available for RPC call.")
                    log_entry_error = current_error
                    raise current_error

                request_id = str(uuid.uuid4())
                payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": request_id
                }
                self.logger.debug(f"Sending JSON-RPC payload to STDIO process (PID: {self._process.pid}): {json.dumps(payload)}")

                try:
                    payload_bytes = (json.dumps(payload) + "\\n").encode('utf-8')
                    self._process.stdin.write(payload_bytes)
                    await self._process.stdin.drain()
                    self.logger.debug("Payload written to STDIO process stdin and drained.")

                    try:
                        self.logger.debug(f"Attempting to read line from STDIO with timeout: {self.request_timeout}s")
                        response_line_bytes = await asyncio.wait_for(
                            self._process.stdout.readline(),
                            timeout=self.request_timeout
                        )
                    except asyncio.TimeoutError:
                        self.logger.error(f"Timeout ({self.request_timeout}s) waiting for response line from STDIO process (PID: {self._process.pid}) for method {method}.")
                        if self._process.stderr:
                            try:
                                stderr_output = await asyncio.wait_for(self._process.stderr.read(2048), timeout=0.5)
                                if stderr_output:
                                    self.logger.error(f"STDIO process (PID: {self._process.pid}) stderr dump on timeout: {stderr_output.decode(errors='ignore')}")
                            except asyncio.TimeoutError:
                                self.logger.warning(f"Timeout reading stderr from STDIO process (PID: {self._process.pid}) after primary read timeout.")
                            except Exception as e_stderr:
                                self.logger.warning(f"Error reading stderr from STDIO process (PID: {self._process.pid}): {e_stderr}")
                                
                        current_error = McpTimeoutError(f"Timeout reading from STDIO process for method {method}.")
                        log_entry_error = current_error
                        raise current_error

                    if not response_line_bytes:
                        self.logger.error(f"STDIO process (PID: {self._process.pid}) closed stdout or sent empty line before response for method {method}.")
                        current_error = McpConnectionError("STDIO process closed stdout or sent empty line.")
                        log_entry_error = current_error
                        raise current_error
                    
                    response_line = response_line_bytes.decode('utf-8').strip()
                    self.logger.debug(f"Received raw response line from STDIO process (PID: {self._process.pid}): {response_line[:500]}...")

                    response_json = json.loads(response_line)
                    self.logger.debug(f"Parsed JSON-RPC response from STDIO: {json.dumps(response_json)}")

                    if not isinstance(response_json, dict):
                        current_error = McpError(f"Invalid JSON-RPC response from STDIO: not a dictionary. Response: {response_line[:200]}")
                        log_entry_error = current_error
                        raise current_error

                    if response_json.get("jsonrpc") != "2.0":
                        current_error = McpError(f"Invalid JSON-RPC version from STDIO: {response_json.get('jsonrpc')}. Expected '2.0'. Response: {response_line[:200]}")
                        log_entry_error = current_error
                        raise current_error

                    if response_json.get("id") != request_id:
                        current_error = McpError(f"JSON-RPC ID mismatch from STDIO: Expected {request_id}, got {response_json.get('id')}. Response: {response_line[:200]}")
                        log_entry_error = current_error
                        raise current_error
                    
                    if "result" in response_json:
                        log_entry_response = response_json
                        return response_json
                    
                    elif "error" in response_json:
                        error_obj = response_json["error"]
                        err_msg = error_obj.get("message", "Unknown MCP error from STDIO")
                        err_code = error_obj.get("code", "N/A")
                        err_data = error_obj.get("data")
                        self.logger.error(f"MCP STDIO server returned error: Code {err_code}, Message: {err_msg}, Data: {err_data}")
                        log_entry_response = response_json
                        current_error = McpError(f"MCP STDIO Error {err_code}: {err_msg} (Data: {err_data})")
                        log_entry_error = current_error
                        raise current_error
                    else:
                        current_error = McpError(f"Invalid JSON-RPC response from STDIO: Missing 'result' or 'error'. Response: {response_line[:200]}")
                        log_entry_error = current_error
                        raise current_error

                except BrokenPipeError as e_pipe:
                    self.logger.error(f"BrokenPipeError communicating with STDIO process for {method}: {e_pipe}", exc_info=True)
                    current_error = McpConnectionError(f"Broken pipe with STDIO process: {e_pipe}")
                    log_entry_error = current_error
                    raise current_error from e_pipe
                except ConnectionResetError as e_reset:
                    self.logger.error(f"ConnectionResetError communicating with STDIO process for {method}: {e_reset}", exc_info=True)
                    current_error = McpConnectionError(f"Connection reset by STDIO process: {e_reset}")
                    log_entry_error = current_error
                    raise current_error from e_reset
                except json.JSONDecodeError as e_json:
                    self.logger.error(f"Failed to decode JSON response from STDIO process for {method}. Raw line: {response_line[:500]}... Error: {e_json}", exc_info=True)
                    current_error = McpError(f"JSON decode error from STDIO: {e_json}. Response: {response_line[:200]}")
                    log_entry_error = current_error
                    raise current_error from e_json
                except Exception as e_stdio_comm:
                    self.logger.error(f"Unexpected error during STDIO communication for {method}: {e_stdio_comm}", exc_info=True)
                    current_error = McpError(f"Unexpected STDIO communication error for {method}: {e_stdio_comm}")
                    log_entry_error = current_error
                    raise current_error from e_stdio_comm
            else:
                current_error = McpError(f"Unsupported transport type for send_request_async: {self.transport_type}")
                log_entry_error = current_error
                raise current_error
        
        except HTTPStatusError as e_http:
             self.logger.error(f"HTTP error during MCP request {method} to {self.server_address}: {e_http.response.status_code} {e_http.request.url}", exc_info=True)
             log_entry_error = McpConnectionError(f"HTTP error {e_http.response.status_code} for {method}")
             raise log_entry_error from e_http
        except RequestError as e_conn:
            self.logger.error(f"Request error during MCP request {method} to {self.server_address}: {e_conn}", exc_info=True)
            if isinstance(e_conn, ConnectTimeout):
                 log_entry_error = McpTimeoutError(f"Connect timeout for {method}: {e_conn}")
            elif isinstance(e_conn, ReadTimeout):
                 log_entry_error = McpTimeoutError(f"Read timeout for {method}: {e_conn}")
            else:
                 log_entry_error = McpConnectionError(f"Connection/Request error for {method}: {e_conn}")
            raise log_entry_error from e_conn
        except Exception as e_general:
             self.logger.error(f"Unexpected error during send_request_async for {method}: {e_general}", exc_info=True)
             log_entry_error = McpError(f"Unexpected error for {method}: {e_general}")
             raise log_entry_error
        finally:
            if isinstance(self.logger, MochiLogger):
                self.logger.log_mcp_call(
                    server_id=self.server_name,
                    tool=method,
                    inputs=params,
                    response=log_entry_response if log_entry_response is not None else { "error_details": str(log_entry_error) } if log_entry_error is not None else { "status": "No response or error captured" }
                )

    async def list_tools(self, tool_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves a list of available tools from the MCP server or mock configuration.

        Args:
            tool_ids: Optional list of specific tool IDs to retrieve. 
                      If None, all tools are listed.

        Returns:
            A list of tool definition dictionaries, where each dictionary is expected
            to conform to a structure that includes at least 'tool_name' (or 'id'),
            'description', 'input_schema', 'output_schema', and crucially 'server_id'
            (which this method ensures is set to self.server_name).
        """
        params = {}
        if tool_ids is not None:
            params["tool_ids"] = tool_ids
        
        self.logger.info(f"Fetching list of tools from server: {self.server_name} ({self.server_address}). Params: {params}")
        response_data = await self.send_request_async("mcp_list_tools", params)

        if isinstance(response_data.get("result"), list):
            tools_list = response_data["result"]
            processed_tools_list = []
            for tool_def in tools_list:
                if isinstance(tool_def, dict):
                    tool_copy = tool_def.copy()
                    if tool_copy.get('server_id') != self.server_name:
                        self.logger.debug(f"Injecting/overwriting server_id='{self.server_name}' from client into actual tool schema for tool '{tool_copy.get('tool_name', tool_copy.get('id', 'UnknownTool'))}'")
                        tool_copy['server_id'] = self.server_name
                    processed_tools_list.append(tool_copy)
                else:
                    self.logger.warning(f"Item in tools_list from server {self.server_name} is not a dictionary: {tool_def}. Skipping.", event_type="MCP_INVALID_ITEM")
            return processed_tools_list
        else:
            error_msg = f"Invalid response for mcp_list_tools from server {self.server_name}: Expected 'result' to be a list, got {type(response_data.get('result'))}. Response: {response_data}"
            self.logger.error(error_msg, event_type="MCP_INVALID_RESPONSE")
            raise McpError(f"Invalid mcp_list_tools response from {self.server_name}: result is not a list.")

    async def call_tool(self, task_id_for_result: str, tool_id: str, inputs: Dict[str, Any]) -> ToolExecutionResult:
        """
        Executes a specific tool with the given inputs.
        Corresponds to the MCP 'mcp_call_tool' method.

        Args:
            task_id_for_result: The parent task's ID, used for populating ToolExecutionResult.
            tool_id: The ID of the tool to call.
            inputs: A dictionary of input parameters for the tool.

        Returns:
            A ToolExecutionResult Pydantic model containing the output or error.

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError: For any MCP specific errors during the call, including tool execution errors.
            McpConnectionError: For network or transport issues.
            McpTimeoutError: If the request times out.
        """
        if not self._is_initialized:
            self.logger.error("Client not fully initialized for call_tool. Call initialize() first.")
            # Raise McpNotInitializedError, which can be caught by the caller
            # and translated into a StructuredError if appropriate there.
            # For now, McpClient itself won't directly return ToolExecutionResult with StructuredError for this case.
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")

        params = {"tool_id": tool_id, "inputs": inputs}
        response_data: Optional[Dict[str, Any]] = None
                
        try:
            response_data = await self.send_request_async("mcp_call_tool", params)
            
            if response_data and isinstance(response_data, dict) and "result" in response_data:
                tool_output_payload = response_data["result"]
                if isinstance(tool_output_payload, dict):
                    # Check if the tool execution itself resulted in an error reported by the tool server
                    if tool_output_payload.get("status") == "failure" or tool_output_payload.get("error"):
                        tool_error_details = tool_output_payload.get("error")
                        structured_tool_error = StructuredError(
                            error_type="ToolReportedError", # Specific type for errors reported by the tool itself
                            message=str(tool_error_details) if not isinstance(tool_error_details, dict) else tool_error_details.get("message", "Tool reported an unspecified error."),
                            details=tool_error_details if isinstance(tool_error_details, dict) else {"raw_error": tool_error_details},
                            is_repairable=True # Assume tool-reported errors might be repairable
                        )
                        return ToolExecutionResult(
                            task_id=task_id_for_result, 
                            status="failure",
                            output=tool_output_payload.get("output"), # Include output if any, even on failure
                            error=structured_tool_error
                        )
                    else:
                        return ToolExecutionResult(
                            task_id=task_id_for_result, 
                            status=tool_output_payload.get("status", "success"), # Default to success if not specified
                            output=tool_output_payload.get("output"),
                            error=None
                        )
                else:
                    self.logger.error(f"Invalid 'result' payload structure from mcp_call_tool for tool '{tool_id}': {tool_output_payload}")
                    structured_error = StructuredError(
                        error_type="McpClientInvalidPayloadError", 
                        message=f"Invalid result payload for tool {tool_id}",
                        details={"payload": tool_output_payload}
                    )
                    return ToolExecutionResult(task_id=task_id_for_result, status="failure", output=None, error=structured_error)
            else:
                self.logger.error(f"Invalid or unexpected response structure from mcp_call_tool for tool '{tool_id}': {response_data}")
                structured_error = StructuredError(
                    error_type="McpClientInvalidResponseError", 
                    message=f"Invalid response structure from server for tool {tool_id}",
                    details={"response": response_data}
                )
                return ToolExecutionResult(task_id=task_id_for_result, status="failure", output=None, error=structured_error)

        except McpTimeoutError as mte:
            self.logger.error(f"McpTimeoutError during call_tool for tool '{tool_id}': {mte}", exc_info=True)
            structured_error = StructuredError(
                error_type="McpTimeoutError", 
                message=f"MCP Timeout calling tool {tool_id}: {str(mte)}",
                is_retryable=True, 
                is_repairable=False # Usually timeouts aren't fixed by DAG repair, but by retries or server fixes
            )
            return ToolExecutionResult(task_id=task_id_for_result, status="failure", output=None, error=structured_error)
        except McpConnectionError as mce:
            self.logger.error(f"McpConnectionError during call_tool for tool '{tool_id}': {mce}", exc_info=True)
            structured_error = StructuredError(
                error_type="McpConnectionError", 
                message=f"MCP Connection error calling tool {tool_id}: {str(mce)}",
                is_retryable=True,
                is_repairable=False
            )
            return ToolExecutionResult(task_id=task_id_for_result, status="failure", output=None, error=structured_error)
        except McpError as me:
            self.logger.error(f"McpError during call_tool for tool '{tool_id}': {me}", exc_info=True)
            # This could be an error returned by the MCP server itself (e.g. JSONRPC error formatting)
            # or an error raised by McpClient.send_request_async for parsing issues.
            error_details = None
            if hasattr(me, 'data'): # Attempt to get more details if it's a structured MCP error from the server
                error_details = me.data
            elif hasattr(me, 'args') and me.args:
                error_details = {"mcp_error_args": me.args}

            structured_error = StructuredError(
                error_type="McpServerError" if error_details else "McpClientGenericError", 
                message=f"MCP Error calling tool {tool_id}: {str(me)}",
                details=error_details,
                is_repairable=True # Some MCP errors might be due to bad inputs
            )
            return ToolExecutionResult(task_id=task_id_for_result, status="failure", output=None, error=structured_error)
        except Exception as e:
            self.logger.error(f"Unexpected client-side error during call_tool for tool '{tool_id}': {e}", exc_info=True)
            structured_error = StructuredError(
                error_type="ClientSideToolCallError", 
                message=f"Client-side error calling tool {tool_id}: {str(e)}",
                is_repairable=False # Usually client-side code issues, not DAG repairable
            )
            return ToolExecutionResult(task_id=task_id_for_result, status="failure", output=None, error=structured_error)
        
    async def list_resources(self, resource_type: Optional[str] = None, resource_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Lists available resources or provides details for specified resource IDs.
        Corresponds to the MCP 'mcp_list_resources' method.

        Args:
            resource_type: Optional type of resources to list (e.g., 'file', 'url').
            resource_ids: Optional list of resource IDs/URIs to get details for.

        Returns:
            A list of resource definition objects.

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError, McpConnectionError, McpTimeoutError: For communication issues.
        """
        if not self._is_initialized:
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")
        
        params: Dict[str, Any] = {}
        if resource_type is not None:
            params["resource_type"] = resource_type
        if resource_ids is not None:
            params["resource_ids"] = resource_ids
            
        response_result = await self.send_request_async("mcp_list_resources", params)
        if not isinstance(response_result, list):
            raise McpError(f"Invalid response from mcp_list_resources: Expected list, got {type(response_result)}")
        return response_result

    async def read_resource(self, resource_uri: str, byte_range_start: Optional[int] = None, byte_range_end: Optional[int] = None) -> Dict[str, Any]:
        """
        Reads the content of a specified resource.
        Corresponds to the MCP 'mcp_read_resource' method.

        Args:
            resource_uri: The URI of the resource to read.
            byte_range_start: Optional start of byte range for partial reads.
            byte_range_end: Optional end of byte range for partial reads.

        Returns:
            A dictionary containing resource content and metadata (e.g., content, media_type).

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError, McpConnectionError, McpTimeoutError: For communication issues.
        """
        if not self._is_initialized:
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")

        params: Dict[str, Any] = {"resource_uri": resource_uri}
        if byte_range_start is not None:
            params["start_byte"] = byte_range_start
        if byte_range_end is not None:
            params["end_byte"] = byte_range_end
            
        response_result = await self.send_request_async("mcp_read_resource", params)
        if not isinstance(response_result, dict):
            raise McpError(f"Invalid response from mcp_read_resource: Expected dict, got {type(response_result)}")
        return response_result

    async def list_prompts(self, prompt_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Lists available prompts or provides details for specified prompt IDs.
        Corresponds to the MCP 'mcp_list_prompts' method.

        Args:
            prompt_ids: Optional list of prompt IDs to get details for.

        Returns:
            A list of prompt definition objects.

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError, McpConnectionError, McpTimeoutError: For communication issues.
        """
        if not self._is_initialized:
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")
        
        params: Dict[str, Any] = {}
        if prompt_ids is not None:
            params["prompt_ids"] = prompt_ids
            
        response_result = await self.send_request_async("mcp_list_prompts", params)
        if not isinstance(response_result, list):
            raise McpError(f"Invalid response from mcp_list_prompts: Expected list, got {type(response_result)}")
        return response_result

    async def get_prompt(self, prompt_id: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieves a specific prompt, optionally with template variables substituted.
        Corresponds to the MCP 'mcp_get_prompt' method.

        Args:
            prompt_id: The ID of the prompt to retrieve.
            variables: Optional dictionary of variables for template substitution.

        Returns:
            A dictionary containing the rendered prompt and metadata.

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError, McpConnectionError, McpTimeoutError: For communication issues.
        """
        if not self._is_initialized:
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")

        params: Dict[str, Any] = {"prompt_id": prompt_id}
        if variables is not None:
            params["variables"] = variables
            
        response_result = await self.send_request_async("mcp_get_prompt", params)
        if not isinstance(response_result, dict):
            raise McpError(f"Invalid response from mcp_get_prompt: Expected dict, got {type(response_result)}")
        return response_result

    async def add_root(self, root_uri: str, recursive: bool = True, readahead: Optional[int] = None) -> Dict[str, Any]:
        """
        Adds a new root URI for the MCP server to manage/access.
        Corresponds to the MCP 'mcp_add_root' method.

        Args:
            root_uri: The URI of the root to add.
            recursive: Whether to discover resources recursively (default True).
            readahead: Optional hint for how much data to readahead.

        Returns:
            A dictionary, typically a status confirmation from the server.

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError, McpConnectionError, McpTimeoutError: For communication issues.
        """
        if not self._is_initialized:
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")

        params: Dict[str, Any] = {"root_uri": root_uri, "recursive": recursive}
        if readahead is not None:
            params["readahead"] = readahead
            
        response_result = await self.send_request_async("mcp_add_root", params)
        if not isinstance(response_result, dict):
            raise McpError(f"Invalid response from mcp_add_root: Expected dict, got {type(response_result)}")
        return response_result

    async def remove_root(self, root_uri: str) -> Dict[str, Any]:
        """
        Removes a root URI from the MCP server's management.
        Corresponds to the MCP 'mcp_remove_root' method.

        Args:
            root_uri: The URI of the root to remove.

        Returns:
            A dictionary, typically a status confirmation from the server.

        Raises:
            McpNotInitializedError: If the client is not fully initialized.
            McpError, McpConnectionError, McpTimeoutError: For communication issues.
        """
        if not self._is_initialized:
            raise McpNotInitializedError("Client not fully initialized. Call initialize() first.")

        params: Dict[str, Any] = {"root_uri": root_uri}
            
        response_result = await self.send_request_async("mcp_remove_root", params)
        if not isinstance(response_result, dict):
            raise McpError(f"Invalid response from mcp_remove_root: Expected dict, got {type(response_result)}")
        return response_result

    # Potentially add other helper methods later, e.g., for specific MCP calls
    # like 'get_capabilities', 'describe_tool', etc., as part of capability negotiation.

    # This is an example, actual capabilities fetching would be more robust
    # and might involve specific MCP handshake methods if defined by the protocol.
    async def async_get_server_capabilities(self) -> Dict[str, Any]:
        """
        Fetches the server capabilities, primarily by calling the 'mcp_list_tools' RPC method.
        Optionally falls back to trying an explicitly configured schema_discovery_url or common HTTP GET endpoints.

        Returns:
            A dictionary representing the server's capabilities. Expected to contain a "tools" key
            with a list of tool definitions if successful.
            Returns an empty dict if no capabilities can be fetched.
        """
        self.logger.info(f"Fetching server capabilities for {self.server_name} at {self.server_address}...")
        capabilities: Dict[str, Any] = {}

        if not self._is_initialized:
            await self.initialize()

        # Attempt to use mock tool schemas if provided in config
        if self.mock_tool_schemas is not None:
            self.logger.info(f"Using mock tool schemas for {self.server_name} as per configuration.")
            processed_mock_schemas = []
            for tool_def in self.mock_tool_schemas:
                if isinstance(tool_def, dict):
                    tool_copy = tool_def.copy()
                    tool_copy['server_id'] = self.server_name # Ensure server_id is correctly set
                    processed_mock_schemas.append(tool_copy)
                else:
                    self.logger.warning(f"Mock tool schema item for {self.server_name} is not a dict: {tool_def}")
            self.tool_schemas = processed_mock_schemas
            capabilities["tools"] = self.tool_schemas
            capabilities["name"] = self.server_name
            capabilities["source"] = "mock_config"
            if self.server_description:
                 capabilities["description"] = self.server_description
            return capabilities

        try:
            self.logger.info(f"Attempting to fetch tools via mcp_list_tools for {self.server_name}.")
            tools_list_from_rpc = await self.list_tools() # list_tools already processes and injects server_id
            if tools_list_from_rpc:
                self.logger.info(f"Successfully fetched {len(tools_list_from_rpc)} tools via mcp_list_tools for {self.server_name}.")
                self.tool_schemas = tools_list_from_rpc # Store fetched schemas
                capabilities["tools"] = self.tool_schemas
                capabilities["name"] = self.server_name
                capabilities["source"] = "mcp_list_tools_rpc"
                if self.server_description:
                    capabilities["description"] = self.server_description
                return capabilities
            else:
                self.logger.info(f"mcp_list_tools for {self.server_name} returned no tools or an empty response.")
        except McpError as e:
            self.logger.warning(f"Failed to fetch tools via mcp_list_tools for {self.server_name}: {e}. Will try HTTP GET fallbacks.", exc_info=True)
        except Exception as e_rpc:
            self.logger.error(f"Unexpected error during mcp_list_tools for {self.server_name}: {e_rpc}. Will try HTTP GET fallbacks.", exc_info=True)

        if self.schema_discovery_url:
            specific_schema_url = self.schema_discovery_url
            self.logger.info(f"Attempting HTTP GET for capabilities from explicitly configured schema_discovery_url: {specific_schema_url}")
            try:
                http_get_caps_data = await self._fetch_capabilities_http_get(specific_schema_url)
                if http_get_caps_data:
                    self.logger.info(f"Successfully fetched capabilities from {specific_schema_url}.")
                    if "tools" in http_get_caps_data and isinstance(http_get_caps_data["tools"], list):
                        self.tool_schemas = self._process_raw_tools_list(http_get_caps_data["tools"]) # Store processed schemas
                        capabilities["tools"] = self.tool_schemas
                    elif "openapi" in http_get_caps_data or "paths" in http_get_caps_data: # It's likely an OpenAPI spec
                        self.openapi_spec = http_get_caps_data # Store the full spec
                        self.logger.info(f"Stored OpenAPI spec from {specific_schema_url}. Further parsing may be needed to populate self.tool_schemas.")
                    
                    capabilities.update(http_get_caps_data) # Merge all fetched data
                    capabilities["source"] = specific_schema_url
                    capabilities["name"] = self.server_name # Ensure name is set
                    if self.server_description and "description" not in capabilities:
                        capabilities["description"] = self.server_description
                    return capabilities
            except Exception as e_specific_http:
                self.logger.warning(f"Failed to fetch from {specific_schema_url}: {e_specific_http}. Will try common endpoints.", exc_info=True)

        preferred_endpoints = ["/openapi.json", "/tools", "/schema"]

        self.logger.info(f"Falling back to common HTTP GET endpoints for {self.server_name}.")
        for endpoint_path in preferred_endpoints:
            full_url = self.server_address.rstrip('/') + endpoint_path
            if self.schema_discovery_url == full_url and capabilities:
                pass 

            try:
                http_get_caps_data = await self._fetch_capabilities_http_get(full_url)
                if http_get_caps_data:
                    self.logger.info(f"Successfully fetched capabilities from {full_url}.")
                    if "tools" in http_get_caps_data and isinstance(http_get_caps_data["tools"], list):
                        self.tool_schemas = self._process_raw_tools_list(http_get_caps_data["tools"]) # Store processed schemas
                        capabilities["tools"] = self.tool_schemas
                    elif "openapi" in http_get_caps_data or "paths" in http_get_caps_data:
                        self.openapi_spec = http_get_caps_data # Store the full spec
                        self.logger.info(f"Stored OpenAPI spec from {full_url}. Further parsing may be needed to populate self.tool_schemas.")

                    capabilities.update(http_get_caps_data)
                    capabilities["source"] = full_url
                    capabilities["name"] = self.server_name
                    if self.server_description and "description" not in capabilities:
                        capabilities["description"] = self.server_description
                    return capabilities
            except McpConnectionError as e_conn:
                self.logger.warning(f"Connection error for {full_url}: {e_conn}. Trying next endpoint.")
            except McpTimeoutError as e_timeout:
                self.logger.warning(f"Timeout error for {full_url}: {e_timeout}. Trying next endpoint.")
            except McpError as e_mcp_http:
                 self.logger.warning(f"MCP-related HTTP error for {full_url}: {e_mcp_http}. Trying next endpoint.")
            except Exception as e_http_general:
                self.logger.error(f"Unexpected error fetching capabilities from {full_url}: {e_http_general}", exc_info=True)
        
        self.logger.warning(f"Could not fetch server capabilities for {self.server_name} from any source.")
        return {}

    def _process_raw_tools_list(self, raw_tools: List[Any]) -> List[Dict[str, Any]]:
        """Helper to process a raw list of tools, ensuring server_id is injected."""
        processed_tools = []
        for tool_def in raw_tools:
            if isinstance(tool_def, dict):
                tool_copy = tool_def.copy()
                if tool_copy.get('server_id') != self.server_name:
                    self.logger.debug(f"Injecting/overwriting server_id='{self.server_name}' from client into actual tool schema for tool '{tool_copy.get('tool_name', tool_copy.get('id', 'UnknownTool'))}'")
                    tool_copy['server_id'] = self.server_name
                processed_tools.append(tool_copy)
            else:
                self.logger.warning(f"Item in raw_tools list for server {self.server_name} is not a dictionary: {tool_def}. Skipping.")
        return processed_tools

    async def _fetch_capabilities_http_get(self, url: str) -> Optional[Dict[str, Any]]:
        """Helper method to perform an HTTP GET request using httpx and parse capabilities."""
        if self.http_client.is_closed:
             self.logger.error(f"httpx.AsyncClient closed. Cannot make HTTP GET to {url}.")
             raise McpConnectionError(f"HTTP client closed for GET to {url}.")

        request_headers = {}
        if self.server_config_override:
            if self.server_config_override.custom_headers:
                request_headers.update(self.server_config_override.custom_headers)
            if self.server_config_override.auth_token:
                auth_header_key = 'Authorization'
                if auth_header_key in request_headers and request_headers[auth_header_key] != f"Bearer {self.server_config_override.auth_token}":
                    self.logger.warning(f"'{auth_header_key}' header from custom_headers overwritten by auth_token for GET {url}.")
                request_headers[auth_header_key] = f"Bearer {self.server_config_override.auth_token}"
        
        self.logger.info(f"Attempting httpx GET to {url} with client default timeout")

        try:
            response = await self.http_client.get(url, headers=request_headers)
            response_text = response.text
            self.logger.debug(f"GET {url} - Status: {response.status_code}, Raw Response (first 500 chars): {response_text[:500]}...")
            response.raise_for_status()
            
            try:
                data = response.json()
                if isinstance(data, dict) and ("tools" in data or "openapi" in data or "paths" in data or "components" in data):
                    if "tools" not in data and "openapi" in data:
                        self.logger.info(f"Document from {url} appears to be an OpenAPI spec. Full parsing to extract tools may be needed.")
                    
                    if isinstance(data, list):
                        self.logger.info(f"GET {url} - Response is a list, assuming list of tool schemas.")
                        return {"tools": data} # Standardize to a dict with a 'tools' key
                    elif isinstance(data, dict):
                        return data # Return the fetched dictionary as is
                else:
                    self.logger.warning(f"GET {url} - Response is valid JSON but not in expected capabilities format (missing 'tools', 'openapi', etc.). Content: {response_text[:200]}...")
                    return None
            except json.JSONDecodeError:
                self.logger.error(f"GET {url} - Failed to decode JSON from response: {response_text[:200]}...")
                raise McpError(f"Failed to decode JSON from {url}.")
        except HTTPStatusError as e_http:
            self.logger.warning(f"HTTP GET error for {url}: {e_http.response.status_code} {e_http.request.url}")
            return None
        except RequestError as e_conn:
            self.logger.warning(f"Request error during HTTP GET for {url}: {e_conn}")
            if isinstance(e_conn, ConnectTimeout):
                 raise McpTimeoutError(f"Connect timeout for GET {url}") from e_conn
            elif isinstance(e_conn, ReadTimeout):
                 raise McpTimeoutError(f"Read timeout for GET {url}") from e_conn
            else:
                 raise McpConnectionError(f"Connection/Request error for GET {url}") from e_conn
        except Exception as e_general:
             self.logger.error(f"Unexpected error during HTTP GET for {url}: {e_general}", exc_info=True)
             raise McpError(f"Unexpected error during GET {url}: {e_general}") from e_general

    async def _handle_sse_connection(self, url: str, params: Dict[str, Any], headers: Dict[str, str], stream_callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Handles an SSE connection, parsing events and calling the callback."""
        attempt = 0
        while attempt <= self.sse_max_retries:
            current_event_data = {
                "event": "message",
                "data": "",
                "id": None,
                "retry": None
            }
            data_buffer: List[str] = []

            try:
                self.logger.info(f"Attempting SSE connection (Attempt {attempt + 1}/{self.sse_max_retries + 1}) to {url} with params {params}")
                async with self.http_client.stream("GET", url, params=params, headers=headers, timeout=self.sse_heartbeat_timeout) as response:
                    self.logger.info(f"SSE connection opened to {url}. Status: {response.status_code}")
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        line = line.strip()
                        self.logger.debug(f"SSE raw line received: '{line}'")

                        if not line:
                            if data_buffer:
                                current_event_data["data"] = "\\n".join(data_buffer)
                                try:
                                    self.logger.debug(f"Dispatching SSE event: {current_event_data}")
                                    callback_result = stream_callback(current_event_data)
                                    if asyncio.iscoroutine(callback_result):
                                        await callback_result
                                except Exception as cb_exc:
                                    self.logger.error(f"Error executing SSE stream_callback: {cb_exc}", exc_info=True)

                                data_buffer = []
                                current_event_data = {
                                    "event": "message",
                                    "data": "",
                                    "id": current_event_data["id"],
                                    "retry": None
                                }
                            else:
                                self.logger.debug("SSE empty line received, but no data buffered. Ignoring.")
                            continue

                        if line.startswith(":"):
                            self.logger.debug(f"SSE comment ignored: {line}")
                            continue

                        field, value = line, ""
                        if ":" in line:
                            field, value = line.split(":", 1)
                            value = value.strip()

                        if field == "event":
                            current_event_data["event"] = value
                        elif field == "data":
                            data_buffer.append(value)
                        elif field == "id":
                            if "\\0" in value:
                                self.logger.warning(f"SSE received 'id' field with null byte, ignoring field: {value}")
                            else:
                                current_event_data["id"] = value
                        elif field == "retry":
                            if value.isdigit():
                                current_event_data["retry"] = int(value)
                                self.logger.info(f"SSE server suggested retry delay: {value} ms")
                            else:
                                self.logger.warning(f"SSE received non-integer 'retry' field, ignoring: {value}")
                        else:
                            self.logger.debug(f"SSE unknown field ignored: {field}")

                    self.logger.info(f"SSE stream finished normally for {url}.")
                    if data_buffer:
                        self.logger.warning("SSE stream ended with buffered data but no final empty line. Dispatching last event.")
                        current_event_data["data"] = "\\n".join(data_buffer)
                        try:
                            callback_result = stream_callback(current_event_data)
                            if asyncio.iscoroutine(callback_result):
                                await callback_result
                        except Exception as cb_exc:
                            self.logger.error(f"Error executing final SSE stream_callback: {cb_exc}", exc_info=True)

                    return

            except (ConnectError, ReadTimeout, HTTPStatusError) as e:
                self.logger.warning(f"SSE connection error for {url} (attempt {attempt + 1}/{self.sse_max_retries + 1}): {e}. Response status: {response.status_code if 'response' in locals() else 'N/A'}")
                if attempt < self.sse_max_retries:
                    delay = self.sse_retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying SSE connection in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"SSE connection failed after {self.sse_max_retries + 1} attempts for {url}: {e}")
                    raise McpConnectionError(f"SSE connection failed permanently after retries: {e}") from e

            except Exception as e:
                self.logger.error(f"Unexpected error in SSE connection handling for {url} (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < self.sse_max_retries:
                     delay = self.sse_retry_delay * (2 ** attempt)
                     self.logger.info(f"Retrying SSE connection after unexpected error in {delay:.2f}s...")
                     await asyncio.sleep(delay)
                else:
                     raise McpError(f"Unexpected SSE error after retries: {e}") from e
            finally:
                pass

            attempt += 1
            
        self.logger.error(f"SSE connection attempts exhausted for {url}. Failed to establish connection.")
        raise McpConnectionError(f"SSE connection failed after {self.sse_max_retries + 1} attempts.") 