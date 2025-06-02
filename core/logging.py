import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Union
import coloredlogs
from worker.config import LoggingConfig

class MochiLogger:
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up the logger based on configuration."""
        log_level_str = self.config.level.upper()
        
        logger = logging.getLogger("mochi")
        numeric_level = getattr(logging, log_level_str, None)
        if not isinstance(numeric_level, int):
            print(f"Warning: Invalid logging level '{log_level_str}'. Defaulting to INFO.")
            numeric_level = logging.INFO
        
        logger.setLevel(numeric_level)
        
        if logger.hasHandlers():
            for handler in list(logger.handlers):
                try:
                    handler.flush()
                    handler.close()
                except Exception as e_close:
                    print(f"MochiLogger._setup_logger: Warning: Error closing handler {handler}: {e_close}", file=sys.stderr)
            logger.handlers.clear()

        coloredlogs.install(
            level=numeric_level,
            logger=logger,
            fmt='%(asctime)s [%(levelname)s] [%(name)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            stream=sys.stdout
        )
        
        default_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if self.config.log_file:
            try:
                log_dir = os.path.dirname(self.config.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(self.config.log_file, mode='a')
                file_handler.setFormatter(default_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Error setting up file logger for '{self.config.log_file}': {e}", file=sys.stderr)
            
        return logger

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata for logging, ensuring it's appended cleanly."""
        if not metadata:
            return ""
        try:
            return f" | metadata: {json.dumps(metadata)}"
        except TypeError:
            return " | metadata: (unserializable)"


    def _log(self, level: str, message: str, event_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, exc_info=False):
        """Internal generic logging method."""
        level_upper = level.upper()
        log_fn = getattr(self.logger, level_upper.lower(), self.logger.info)
        
        log_payload = {
            "message": message,
        }
        if event_type:
            log_payload["event_type"] = event_type
        
        formatted_message = f"{message}{self._format_metadata(metadata)}"
        if event_type:
            formatted_message = f"[{event_type}] {formatted_message}"

        log_fn(formatted_message, exc_info=exc_info)
        numeric_log_level = getattr(logging, level_upper, logging.INFO)
        if not isinstance(numeric_log_level, int):
            numeric_log_level = logging.INFO
            self.logger.warning(f"Invalid log level '{level_upper}' passed to _log. Defaulting to INFO.", stacklevel=2)

        self.logger.log(numeric_log_level, formatted_message, exc_info=exc_info, stacklevel=3)


    def debug(self, message: str, event_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self._log("DEBUG", message, event_type, metadata)

    def info(self, message: str, event_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self._log("INFO", message, event_type, metadata)

    def warning(self, message: str, event_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self._log("WARNING", message, event_type, metadata)

    def error(self, message: str, event_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, exc_info=False):
        self._log("ERROR", message, event_type, metadata, exc_info=exc_info)

    def critical(self, message: str, event_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, exc_info=False):
        self._log("CRITICAL", message, event_type, metadata, exc_info=exc_info)

    # Specific logging methods from the task description
    def log_agent_lifecycle(self, action: str, agent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        msg = f"Agent action: {action.upper()}"
        if agent_id:
            msg += f" - Agent ID: {agent_id}"
        self.info(msg, event_type="AGENT_LIFECYCLE", metadata=metadata)

    def log_llm_call(
        self, 
        provider_name: str, 
        model_name: str,    
        prompt_type: str,   
        system_prompt: Optional[str], 
        user_prompt: str,             
        metadata: Optional[Dict[str, Any]] = None,
        attempt_number: Optional[int] = None
    ):
        if not self.config.log_llm_calls:
            return
        
        call_metadata = {
            "provider_name": provider_name,
            "model_name": model_name,
            "prompt_type": prompt_type,
            "system_prompt_summary": (system_prompt[:200] + '...' if system_prompt and len(system_prompt) > 200 else system_prompt) if system_prompt else None,
            "user_prompt_summary": user_prompt[:500] + '...' if len(user_prompt) > 500 else user_prompt,
        }
        if attempt_number is not None:
            call_metadata["attempt_number"] = attempt_number
        if metadata:
            call_metadata.update(metadata)
        
        self.info(
            f"LLM Call: Provider={provider_name}, Model={model_name}, Type={prompt_type}", 
            event_type="LLM_CALL_SENT", 
            metadata=call_metadata
        )

    def log_llm_call_response(
        self, 
        response: Optional[str], 
        error: Optional[str],    
        metadata: Optional[Dict[str, Any]] = None,
        attempt_number: Optional[int] = None
    ):
        if not self.config.log_llm_calls:
            return

        response_metadata = {
            "response_summary": (response[:500] + '...' if response and len(response) > 500 else response) if response else None,
            "error": error,
        }
        if attempt_number is not None:
            response_metadata["attempt_number"] = attempt_number
        if metadata:
            response_metadata.update(metadata)

        log_level = "ERROR" if error else "DEBUG"
        event_t = "LLM_CALL_ERROR" if error else "LLM_CALL_RECEIVED"
        message = f"LLM Response: {'Error: ' + error if error else 'Success'}"
        
        self._log(log_level, message, event_type=event_t, metadata=response_metadata)

    def log_mcp_call(self, server_id: str, tool: str, inputs: Dict[str, Any], response: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        if not self.config.log_mcp_calls:
            return

        call_metadata = {"server_id": server_id, "tool": tool, "inputs": inputs}
        if metadata:
            call_metadata.update(metadata) # type: ignore

        self.debug(f"MCP Call to server: {server_id}, tool: {tool}", event_type="MCP_CALL_SENT", metadata=call_metadata)

        if response is not None:
            response_metadata = {"server_id": server_id, "tool": tool, "response": response}
            if metadata:
                response_metadata.update(metadata) # type: ignore
            self.debug(f"MCP Response from server: {server_id}, tool: {tool}", event_type="MCP_CALL_RECEIVED", metadata=response_metadata)
            
    def log_task_execution(self, task_id: str, status: str, result: Optional[Any] = None, error: Optional[Union[str, Exception]] = None, metadata: Optional[Dict[str, Any]] = None):
        exec_metadata = {"task_id": task_id, "status": status}
        if result is not None:
            exec_metadata["result"] = result
        if error is not None:
            exec_metadata["error"] = str(error)
        if metadata:
            exec_metadata.update(metadata) # type: ignore
        
        level = "ERROR" if status.lower() == "failed" or error else "INFO"
        msg = f"Task Execution: ID '{task_id}', Status '{status.upper()}'"
        
        self._log(level, msg, event_type="TASK_EXECUTION", metadata=exec_metadata, exc_info=(error is not None and isinstance(error, Exception)))
