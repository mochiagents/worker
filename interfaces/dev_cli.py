import argparse
import sys
import json
import asyncio
import os
import time
from typing import Optional, Dict, Any, List, Set
import uuid
import requests

from worker.agent.main import MochiAgent
from worker.config.manager import ConfigurationManager
from worker.config.models import MochiWorkerConfig
from worker.config.exceptions import MochiConfigError
from worker.core.models import HierarchicalPlan, TaskDAG

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table

class MochiCLI:
    def __init__(self, agent_instance: Optional[MochiAgent] = None):
        self.parser = self._create_parser()
        self.agent: Optional[MochiAgent] = agent_instance
        self.current_query: Optional[str] = None
        self.console = Console()
        self.current_conversation_id: Optional[str] = None # Added for managed queries
        self.current_dag_tree: Optional[Tree] = None
        self.current_task_statuses: Dict[str, str] = {}

    def _create_parser(self):
        """Create the argument parser based on the designed command structure."""
        parser = argparse.ArgumentParser(
            description=(
                "Mochi Worker Agent CLI.\n\n"
                "This CLI allows you to interact with the Mochi worker agent. "
                "You can run queries, manage configurations, check status, view logs, "
                "and test the agent's external API."
            ),
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            '--version',
            action='version',
            version='%(prog)s 0.1.0',
            help="Show program's version number and exit"
        )

        subparsers = parser.add_subparsers(dest="command", title="Commands", help="Available commands:", required=True)

        # --- Run Command ---
        run_parser = subparsers.add_parser("run", help="Run the Mochi worker agent with a query. Example: mochi run \"What is the weather?\" -c config.yaml")
        run_parser.add_argument("query", nargs='?', default=None, help="The query for the agent to process. Optional if -i is used without an initial query.") # query is optional if -i
        run_parser.add_argument("-c", "--config", help="Path to the Mochi configuration file (e.g., worker_config.yaml or .json). Overrides default and environment-based configurations.")
        run_parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode. After an initial query (if provided), you can continue to submit queries.")
        run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output, showing more detailed logs and operational messages from the agent.")
        run_parser.set_defaults(func=self._handle_run)

        # --- Config Command ---
        config_parser = subparsers.add_parser("config", help="Manage agent configuration. Allows showing, creating, or updating configuration files.")
        config_subparsers = config_parser.add_subparsers(dest="action", title="Config Actions", help="Action to perform on the configuration:", required=True)

        config_show_parser = config_subparsers.add_parser("show", help="Show current or specified configuration. If no file is specified, shows the default/loaded config.")
        config_show_parser.add_argument("-f", "--file", help="Path to a specific Mochi configuration file to load and display.")
        config_show_parser.set_defaults(func=self._handle_config_show)

        config_create_parser = config_subparsers.add_parser("create", help="Create a new default Mochi configuration file at the specified path.")
        config_create_parser.add_argument("-f", "--file", required=True, help="Path where the new configuration file will be saved (e.g., worker_config.yaml or worker_config.json).")
        config_create_parser.set_defaults(func=self._handle_config_create)

        config_update_parser = config_subparsers.add_parser("update", help="Update a specific key in a Mochi configuration file.")
        config_update_parser.add_argument("-f", "--file", required=True, help="Path to the Mochi configuration file to update.")
        config_update_parser.add_argument("-k", "--key", required=True, help="Configuration key to update (e.g., 'planner.llm_profile_name' or 'mcp_tool_servers[0].name'). Use dot notation for nested keys and bracket notation for list indices.")
        config_update_parser.add_argument("--val", required=True, help="New value for the key. Should be JSON formatted for complex types (e.g., '{\"temperature\": 0.5}', '[\"tool1\", \"tool2\"]', 'true', '123', '\"a string\"').")
        config_update_parser.set_defaults(func=self._handle_config_update)

        # --- Status Command ---
        status_parser = subparsers.add_parser("status", help="Check agent's overall status or the status of a specific task (basic config check).")
        # status_parser.add_argument("-t", "--task-id", help="Specific task ID to check status for. If omitted, shows overall agent status.") # Task status might be too complex for standalone CLI
        status_parser.set_defaults(func=self._handle_status)

        # --- Logs Command ---
        logs_parser = subparsers.add_parser("logs", help="View agent logs. Requires logging to be configured to a file.")
        logs_parser.add_argument("-F", "--follow", action="store_true", help="Follow log output in real-time (similar to 'tail -f').")
        logs_parser.add_argument("-n", "--lines", type=int, default=20, help="Number of recent log lines to show (default: 20).")
        logs_parser.add_argument("-l", "--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Minimum log level to display (e.g., INFO will show INFO, WARNING, ERROR, CRITICAL).")
        logs_parser.set_defaults(func=self._handle_logs)

        # --- API Command ---
        api_parser = subparsers.add_parser("api", help="Test the Mochi agent's External Tasking and Control API (if exposed, e.g., via HTTP).")
        api_parser.add_argument("endpoint", help="API endpoint path to call (e.g., '/tasks' or '/status'). The base URL is typically configured in the agent.")
        api_parser.add_argument("-m", "--method", default="GET", choices=["GET", "POST", "PUT", "DELETE", "PATCH"], help="HTTP method to use for the API call (default: GET).")
        api_parser.add_argument("-d", "--data", help="JSON string data for POST/PUT/PATCH requests body (e.g., '{\"key\": \"value\"}').")
        api_parser.add_argument("--base-url", help="Override the agent's configured base URL for this API call. Useful for testing different environments.")
        api_parser.set_defaults(func=self._handle_api)
        return parser

    def run_cli(self, cli_args=None):
        """Parse arguments and dispatch to the appropriate handler."""
        if cli_args is None:
            cli_args = sys.argv[1:]

        # Handle no command case explicitly for better help message
        if not cli_args:
            self.parser.print_help()
            sys.exit(1)
        
        # Handle config command not having an action
        if cli_args[0] == 'config' and (len(cli_args) == 1 or cli_args[1] not in ['show', 'create', 'update']):
             # Find the config subparser to print its help
            for action in self.parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for name, subparser_object in action.choices.items():
                        if name == 'config':
                            subparser_object.print_help()
                            sys.exit(1)
            self.parser.print_help()
            sys.exit(1)

        args = self.parser.parse_args(cli_args)
        
        if hasattr(args, 'func'):
            try:
                if args.command == 'run':
                    async def run_agent_main_loop():
                        agent_initialized_successfully = False
                        try:
                            self.agent = MochiAgent(config_path=args.config if args.config else None)
                            agent_initialized_successfully = True
                        except MochiConfigError as mce:
                            self.console.print(Panel(
                                Text(f"Configuration Error: {escape(str(mce))}\n\nPlease check your 'worker_config.yaml' (or the specified config file). Ensure that required LLM profiles (e.g., 'planner.llm_profile_name', 'joiner.llm_profile_name') are defined and point to valid entries in the 'llm_profiles' section. Also verify MCP tool server configurations.", style="bold red"), 
                                title="Mochi Configuration Error", 
                                border_style="red"
                            ))
                            return
                        except Exception as e_agent_init:
                            self.console.print(Panel(
                                Text(f"Failed to initialize Mochi Agent: {escape(str(e_agent_init))}\n\nThis could be due to issues with the configuration file structure, an inability to load dependent services, or other critical setup problems. Please check the detailed logs if available.", style="bold red"), 
                                title="Agent Initialization Error", 
                                border_style="red"
                            ))
                            return
                            
                        try:
                            await self.agent.start()

                            if not args.interactive and not args.query:
                                self.console.print(Panel(
                                    Text("Error: A query must be provided if not running in interactive mode (-i).\n\nUsage: mochi run <your_query_here>\n   or: mochi run -i", style="bold red"),
                                    title="Missing Query",
                                    border_style="red"
                                ))
                                for action in self.parser._actions:
                                    if isinstance(action, argparse._SubParsersAction):
                                        if 'run' in action.choices:
                                            action.choices['run'].print_help(self.console.file) # type: ignore
                                        break
                                return

                            if args.interactive and not args.query:
                                self.current_conversation_id = str(uuid.uuid4())
                                self.console.print(Panel(f"Entering interactive mode. Conversation ID: {self.current_conversation_id}", title="💬 Interactive Mode", border_style="magenta"))
                                while True:
                                    try:
                                        user_input = self.console.input("[b blue]mochi>[/] ").strip()
                                        if not user_input: continue
                                        if user_input.lower() in ("exit", "quit"):
                                            self.console.print(Text("Exiting interactive mode.", style="magenta"))
                                            break
                                        await self.agent.run_managed_query(
                                            query=user_input, 
                                            conversation_id=self.current_conversation_id, 
                                            stream_callback=self._tli_update_callback
                                        )
                                    except (KeyboardInterrupt, EOFError):
                                        self.console.print(Text("Exiting interactive mode (KeyboardInterrupt/EOF).", style="magenta"))
                                        break
                            elif args.query:
                                single_run_conversation_id = str(uuid.uuid4())
                                if args.interactive:
                                    self.current_conversation_id = single_run_conversation_id
                                
                                self.console.print(f"[dim]Processing query with Conversation ID: {single_run_conversation_id}...[/dim]")
                                await self.agent.run_managed_query(
                                    query=args.query, 
                                    conversation_id=single_run_conversation_id, 
                                    stream_callback=self._tli_update_callback
                                )
                                
                                if args.interactive:
                                    self.console.print(Panel(f"Initial query processed. Continuing interactive mode. Conversation ID: {self.current_conversation_id}", title="💬 Interactive Mode", border_style="magenta"))
                                    while True:
                                        try:
                                            user_input = self.console.input("[b blue]mochi>[/] ").strip()
                                            if not user_input: continue
                                            if user_input.lower() in ("exit", "quit"):
                                                self.console.print(Text("Exiting interactive mode.", style="magenta"))
                                                break
                                            await self.agent.run_managed_query(
                                                query=user_input, 
                                                conversation_id=self.current_conversation_id, 
                                                stream_callback=self._tli_update_callback
                                            )
                                        except (KeyboardInterrupt, EOFError):
                                            self.console.print(Text("Exiting interactive mode (KeyboardInterrupt/EOF).", style="magenta"))
                                            break

                        finally:
                            if agent_initialized_successfully and self.agent:
                                self.console.print(Text("Main loop finished, ensuring agent cleanup...", style="dim yellow"))
                                await self.agent.shutdown()
                                self.console.print(Text("Agent cleanup complete.", style="dim yellow"))
                            elif self.agent and not agent_initialized_successfully:
                                self.console.print(Text("Agent initialization failed, skipping full shutdown.", style="dim yellow"))
                    
                    asyncio.run(run_agent_main_loop())

                else:
                    args.func(args)
            except Exception as e:
                self.console.print(Panel(Text(f"An unexpected error occurred in CLI: {escape(str(e))}", style="bold red"), title="CLI Error", border_style="red"))
                sys.exit(1)
        else:
            self.parser.print_help()
            sys.exit(1)

    def _handle_config_show(self, args):
        config_manager = ConfigurationManager()
        try:
            if args.file:
                cfg_path = os.path.abspath(args.file)
                if not os.path.exists(cfg_path):
                    self.console.print(Panel(Text(f"Error: Configuration file not found: {cfg_path}", style="bold red"), title="Config Show Error", border_style="red"))
                    return
                config_manager.config_file_path = cfg_path 
                config_data = config_manager.get_config()
                title = f"Configuration from: {cfg_path}"
            else:
                config_data = config_manager.get_config()
                title = "Current Default Configuration"
            
            config_json_str = config_data.model_dump_json(indent=2)
            syntax = Syntax(config_json_str, "json", theme="native", line_numbers=True)
            self.console.print(Panel(syntax, title=title, border_style="blue"))

        except Exception as e:
            self.console.print(Panel(Text(f"Error showing configuration: {escape(str(e))}", style="bold red"), title="Config Show Error", border_style="red"))

    def _handle_config_create(self, args):
        config_manager = ConfigurationManager(default_config_path=args.file) # Set path for saving
        try:
            default_config_data = MochiWorkerConfig() 
            
            file_path = os.path.abspath(args.file)
            
            if os.path.exists(file_path):
                if not self.console.input(f"[yellow]File '{file_path}' already exists. Overwrite? (y/N):[/] ").lower() == 'y':
                    self.console.print(Text("Configuration creation cancelled.", style="yellow"))
                    return

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(default_config_data.model_dump(mode='json'), f, indent=2)
            
            self.console.print(Panel(Text(f"Successfully created default configuration file at: {file_path}", style="green"), title="Config Create Success", border_style="green"))
            self.console.print(Text("Please review and customize the generated file, especially API keys and LLM profile names.", style="dim yellow"))
        except Exception as e:
            self.console.print(Panel(Text(f"Error creating configuration file {args.file}: {escape(str(e))}", style="bold red"), title="Config Create Error", border_style="red"))

    def _update_nested_dict(self, d: Dict, key_path: str, value: Any) -> None:
        """Helper to update a nested dictionary using a dot-separated key path."""
        keys = key_path.split('.')
        current_level = d
        for i, key_segment in enumerate(keys[:-1]):
            if '[' in key_segment and key_segment.endswith(']'):
                list_key, index_str = key_segment[:-1].split('[')
                try:
                    index = int(index_str)
                    if list_key not in current_level or not isinstance(current_level[list_key], list):
                        raise KeyError(f"List key '{list_key}' not found or not a list.")
                    if index >= len(current_level[list_key]):
                         raise IndexError(f"Index {index} out of bounds for list '{list_key}'.")
                    current_level = current_level[list_key][index]
                except ValueError:
                    raise KeyError(f"Invalid index format in '{key_segment}'. Expected number.")
            elif key_segment not in current_level or not isinstance(current_level[key_segment], dict):
                raise KeyError(f"Key '{key_segment}' not found or not a dictionary in path.")
            else:
                current_level = current_level[key_segment]
        
        final_key = keys[-1]
        if '[' in final_key and final_key.endswith(']'):
            list_key, index_str = final_key[:-1].split('[')
            try:
                index = int(index_str)
                if list_key not in current_level or not isinstance(current_level[list_key], list):
                    raise KeyError(f"Final list key '{list_key}' not found or not a list.")
                if index >= len(current_level[list_key]):
                    if index == len(current_level[list_key]):
                         current_level[list_key].append(value)
                         return
                    raise IndexError(f"Index {index} out of bounds for final list '{list_key}'.")
                current_level[list_key][index] = value
            except ValueError:
                raise KeyError(f"Invalid index format in final key '{final_key}'.")
        else:
            current_level[final_key] = value

    def _handle_config_update(self, args):
        config_manager = ConfigurationManager(default_config_path=args.file)
        try:
            current_config_model = config_manager.get_config()
            config_dict = current_config_model.model_dump(mode='json')

            try:
                parsed_value = json.loads(args.val)
            except json.JSONDecodeError:
                val_lower = args.val.lower()
                if val_lower == 'true': parsed_value = True
                elif val_lower == 'false': parsed_value = False
                elif val_lower == 'null' or val_lower == 'none': parsed_value = None
                else:
                    try:
                        if '.' in args.val: parsed_value = float(args.val)
                        else: parsed_value = int(args.val)
                    except ValueError:
                        parsed_value = args.val # Keep as string if all else fails
            
            self._update_nested_dict(config_dict, args.key, parsed_value)
            
            updated_config_model = MochiWorkerConfig(**config_dict)
            
            file_path = os.path.abspath(args.file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(updated_config_model.model_dump(mode='json'), f, indent=2)

            self.console.print(Panel(Text(f"Successfully updated key '{args.key}' in configuration file: {args.file}", style="green"), title="Config Update Success", border_style="green"))
        except KeyError as e:
            self.console.print(Panel(Text(f"Error updating config: Key path error - {escape(str(e))}", style="bold red"), title="Config Update Error", border_style="red"))
        except IndexError as e:
            self.console.print(Panel(Text(f"Error updating config: List index error - {escape(str(e))}", style="bold red"), title="Config Update Error", border_style="red"))
        except Exception as e:
            self.console.print(Panel(Text(f"Error updating configuration {args.file}: {escape(str(e))}", style="bold red"), title="Config Update Error", border_style="red"))

    def _handle_status(self, args):
        self.console.print(Panel(Text("Attempting to load configuration for status check...", style="dim blue"), title="Agent Status", border_style="blue"))
        try:
            config_manager = ConfigurationManager()
            config = config_manager.get_config()
            
            table = Table(title="Mochi Agent Configuration Status")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            table.add_column("Status", justify="right", style="green")

            table.add_row("Agent ID", escape(str(config.agent_id)), "✅ Loaded")
            table.add_row("Logging Level", escape(str(config.logging.level)), "✅ Loaded")
            if config.logging.log_file:
                table.add_row("Log File", escape(config.logging.log_file), "✅ Configured")
            else:
                table.add_row("Log File", "[dim]Not Configured[/dim]", "⚠️ Using stdout")
            
            table.add_row("Planner LLM Profile", escape(config.planner.llm_profile_name) if config.planner else "[N/A]", "✅ Loaded" if config.planner else "❌ Missing")
            table.add_row("Joiner LLM Profile", escape(config.joiner.llm_profile_name) if config.joiner else "[N/A]", "✅ Loaded" if config.joiner else "❌ Missing")

            if config.mcp_tool_servers:
                table.add_row("MCP Tool Servers", f"{len(config.mcp_tool_servers)} configured", "✅ Present")
                for i, server in enumerate(config.mcp_tool_servers):
                    table.add_row(f"  Server {i+1} Name", escape(server.name), "✅")
                    table.add_row(f"  Server {i+1} Endpoint", escape(server.endpoint_url), "✅")
            else:
                table.add_row("MCP Tool Servers", "None configured", "⚠️ No tools")

            self.console.print(table)
            self.console.print(Text("Basic configuration loaded successfully. For live status, ensure the agent is running and query its API if available.", style="dim green"))

        except Exception as e:
            self.console.print(Panel(Text(f"Could not load agent configuration for status check: {escape(str(e))}", style="bold red"), title="Status Error", border_style="red"))

    def _get_log_file_path_from_config(self) -> Optional[str]:
        try:
            config_manager = ConfigurationManager()
            config = config_manager.get_config()
            if config.logging and config.logging.log_file:
                return os.path.abspath(config.logging.log_file)
        except Exception:
            return None

    def _handle_logs(self, args):
        log_file = self._get_log_file_path_from_config()

        if not log_file or not os.path.exists(log_file):
            self.console.print(Panel(Text("Log file not found or logging to file is not configured. Please check your Mochi agent configuration (logging.log_file).", style="yellow"), title="Logs Error", border_style="yellow"))
            return

        self.console.print(Panel(Text(f"Displaying logs from: {log_file}", style="dim blue"), title="Agent Logs", border_style="blue"))
        try:
            if args.follow:
                self.console.print(Text(f"Following log file: {log_file} (Ctrl+C to stop)", style="italic cyan"))
                current_pos = os.path.getsize(log_file)
                while True:
                    try:
                        with open(log_file, 'r') as f:
                            f.seek(current_pos)
                            new_lines = f.readlines()
                            if new_lines:
                                for line in new_lines:
                                    if args.level and f"[{args.level.upper()}]" not in line.upper():
                                        continue
                                    self.console.print(escape(line.strip()))
                            current_pos = f.tell()
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        self.console.print(Text("Stopped following logs.", style="yellow"))
                        break
            else:
                with open(log_file, 'r') as f:
                    lines = f.readlines() 
                
                if args.level:
                    lines = [line for line in lines if f"[{args.level.upper()}]" in line.upper()]

                start_index = max(0, len(lines) - args.lines)
                for line in lines[start_index:]:
                    self.console.print(escape(line.strip()))
        except Exception as e:
            self.console.print(Panel(Text(f"Error reading logs from {log_file}: {escape(str(e))}", style="bold red"), title="Logs Error", border_style="red"))

    def _handle_api(self, args):
        base_url = args.base_url
        if not base_url:
            try:
                config_manager = ConfigurationManager()
                agent_config = config_manager.get_config()
                if hasattr(agent_config, 'api_settings') and agent_config.api_settings and hasattr(agent_config.api_settings, 'base_url'):
                     base_url = agent_config.api_settings.base_url
                else:
                    base_url = "http://localhost:8000"
                    self.console.print(Text(f"Base URL not specified and not found in config. Defaulting to {base_url}", style="yellow"))
            except Exception as e:
                self.console.print(Text(f"Could not load base_url from config ({e}). Defaulting to http://localhost:8000", style="yellow"))
                base_url = "http://localhost:8000"

        full_url = base_url.rstrip('/') + '/' + args.endpoint.lstrip('/')
        
        json_data_payload = None
        if args.data:
            try:
                json_data_payload = json.loads(args.data)
            except json.JSONDecodeError:
                self.console.print(Panel(Text(f"Error: --data '{args.data}' is not valid JSON.", style="bold red"), title="API Call Error", border_style="red"))
                return

        self.console.print(Panel(Text(f"Making {args.method.upper()} request to {full_url}", style="cyan") + 
                                 (Text(f"Data: {json.dumps(json_data_payload, indent=2)}") if json_data_payload else Text("")),
                                 title="🚀 API Call", border_style="blue"))
        try:
            response = requests.request(
                method=args.method.upper(),
                url=full_url,
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                json=json_data_payload if args.method.upper() in ["POST", "PUT", "PATCH"] else None,
                params=json_data_payload if args.method.upper() == "GET" and json_data_payload else None # GET uses params
            )
            
            status_style = "green" if response.ok else "bold red"
            self.console.print(Panel(Text(f"Status Code: {response.status_code}", style=status_style), title="Response Status", border_style=status_style))

            headers_text = Text()
            for k, v in response.headers.items():
                headers_text.append(f"  {k}: {escape(v)}")
            self.console.print(Panel(headers_text, title="Response Headers", border_style="dim blue", expand=False))
            
            try:
                resp_json = response.json()
                syntax = Syntax(json.dumps(resp_json, indent=2), "json", theme="native", line_numbers=True)
                self.console.print(Panel(syntax, title="Response Body (JSON)", border_style="blue"))
            except json.JSONDecodeError:
                self.console.print(Panel(Text(escape(response.text) if response.text else "[Empty Response Body]", style="white"), title="Response Body (Text)", border_style="blue"))

        except requests.exceptions.RequestException as e:
            self.console.print(Panel(Text(f"API request failed: {escape(str(e))}", style="bold red"), title="API Request Error", border_style="red"))
        except Exception as e:
            self.console.print(Panel(Text(f"An unexpected error occurred during API call: {escape(str(e))}", style="bold red"), title="API Call Error", border_style="red"))

    def _build_dag_tree(self,
                        dag_data: Optional[TaskDAG],
                        task_statuses: Dict[str, str],
                        hierarchical_plan: Optional[HierarchicalPlan] = None,
                        accumulated_global_task_outputs: Optional[Dict[str, Any]] = None,
                        query: Optional[str] = None) -> Tree:
        query_str = escape(query[:50] + "..." if query and len(query) > 50 else query or "Not set")
        
        if hierarchical_plan and hierarchical_plan.phases:
            root_text = Text(f"Plan & Status (Query: {query_str}) - Hierarchical", style="bold cyan")
            tree = Tree(root_text)
            phase_nodes_map: Dict[str, Tree] = {}
            tasks_added_to_phases: Set[str] = set() # Track tasks already displayed under a phase

            for i, phase in enumerate(hierarchical_plan.phases):
                phase_status_str = phase.status if phase.status else "unknown"
                phase_description_escaped = escape(phase.description if phase.description else "No description")
                phase_node = tree.add(f"Phase {i+1}: {phase.phase_id} ({phase_description_escaped}) [Status: {phase_status_str}]")
                phase_nodes_map[phase.phase_id] = phase_node

            if accumulated_global_task_outputs and dag_data and dag_data.tasks:
                original_tasks_map: Dict[str, Any] = {t.id: t for t in dag_data.tasks}

                for global_task_id, task_output_data in accumulated_global_task_outputs.items():
                    matched_phase_id = None
                    actual_task_id_in_dag = global_task_id 

                    for phase_id_key in phase_nodes_map.keys():
                        if global_task_id.startswith(phase_id_key + "_"):
                            matched_phase_id = phase_id_key
                            actual_task_id_in_dag = global_task_id[len(phase_id_key) + 1:]
                            break
                    
                    parent_node_for_task = phase_nodes_map.get(matched_phase_id) if matched_phase_id else None
                    original_task_node = original_tasks_map.get(actual_task_id_in_dag)

                    if parent_node_for_task and original_task_node:
                        # Try to get status using original task ID first, then global (prefixed) ID
                        task_status = task_statuses.get(actual_task_id_in_dag, task_statuses.get(global_task_id, "unknown"))
                        
                        task_node_text = f"Task {actual_task_id_in_dag}: {original_task_node.tool_name} [Status: {task_status}]"
                        task_node_text_rich = Text.from_markup(task_node_text)
                        
                        if original_task_node.dependencies:
                            deps_str = ", ".join(original_task_node.dependencies)
                            task_node_text_rich.append(f" (depends on: {deps_str})", style="dim white")
                        
                        parent_node_for_task.add(task_node_text_rich)
                        tasks_added_to_phases.add(actual_task_id_in_dag)

            for phase_id, phase_node_tree_item in phase_nodes_map.items():
                if not phase_node_tree_item.children:
                    has_tasks_for_this_phase = False
                    if accumulated_global_task_outputs:
                        for global_id_check in accumulated_global_task_outputs.keys():
                            if global_id_check.startswith(phase_id + "_"):
                                has_tasks_for_this_phase = True
                                break
                    if not has_tasks_for_this_phase:
                        phase_node_tree_item.add(Text("  └── No tasks executed or defined for this phase in outputs.", style="dim white"))
            
            if dag_data and dag_data.tasks:
                other_tasks_node = None
                for task in dag_data.tasks:
                    if task.id not in tasks_added_to_phases:
                        if not other_tasks_node:
                            other_tasks_node = tree.add("Other Tasks (not directly under a phase):")
                        
                        task_status = task_statuses.get(task.id, "unknown")
                        task_node_text = f"Task {task.id}: {task.tool_name} [Status: {task_status}]"
                        task_node_text_rich = Text.from_markup(task_node_text)
                        if task.dependencies:
                            deps_str = ", ".join(task.dependencies)
                            task_node_text_rich.append(f" (depends on: {deps_str})", style="dim white")
                        if other_tasks_node: # Ensure other_tasks_node was created
                            other_tasks_node.add(task_node_text_rich)

        elif dag_data and dag_data.tasks:
            root_text = Text(f"Plan & Status (Query: {query_str})", style="bold cyan")
            tree = Tree(root_text)
            for task in dag_data.tasks:
                task_status = task_statuses.get(task.id, "unknown")
                task_node_text = f"Task {task.id}: {task.tool_name} [Status: {task_status}]"
                task_node_text_rich = Text.from_markup(task_node_text)
                if task.dependencies:
                    deps_str = ", ".join(task.dependencies)
                    task_node_text_rich.append(f" (depends on: {deps_str})", style="dim white")
                tree.add(task_node_text_rich)
        else:
            tree = Tree(Text(f"📋 No plan or tasks to display (Query: {query_str})", style="bold yellow"))

        return tree

    def _tli_update_callback(self, event_data: Dict[str, Any]):
        event_type = event_data.get("event_type")
        
        if event_type == "agent_run_start":
            self.current_query = event_data.get('query')
            self.console.print(f"[bold yellow]▶️ Agent Run Start[/bold yellow] (Run ID: {event_data.get('run_id')}, Query: '{self.current_query[:50] if self.current_query else 'N/A'}...')")
        elif event_type == "graph_loop_start":
            self.console.print(f"  [cyan]🔄 Graph Loop Start[/cyan] ({event_data.get('loop_count')}/{event_data.get('max_loops')})")
        elif event_type == "planner_start":
            self.console.print(f"    [blue]🧠 Planning phase started...[/blue] (Attempt: {event_data.get('attempt', 1)})")
        elif event_type == "planner_end":
            if event_data.get("error"):
                self.console.print(f"    [red]🧠 Planner Error:[/red] {event_data.get('error_message')}")
            else:
                dag_summary = event_data.get("dag_summary", {})
                num_tasks = dag_summary.get("num_tasks", "N/A")
                self.console.print(f"    [green]🧠 Planner End:[/green] DAG generated with {num_tasks} tasks.")
        elif event_type == "task_fetching_start":
            self.console.print(f"    [blue]⚙️ Task Fetching/Execution phase started...[/blue]")
        elif event_type == "task_status_update":
            task_id = event_data.get('task_id')
            status = event_data.get('status')
            tool_name = event_data.get('tool_name', '')
            self.console.print(f"      [dim]Task Update:[/] {task_id} ({tool_name}) -> {status}")
        elif event_type == "task_result_received":
            task_id = event_data.get('task_id')
            self.console.print(f"      [dim]Task Result:[/dim] {task_id} received.")
        elif event_type == "task_fetching_end":
             self.console.print(f"    [green]⚙️ Task Fetching/Execution End.[/green]")
        elif event_type == "joiner_start":
            self.console.print(f"    [blue]🤝 Joiner phase started...[/blue]")
        elif event_type == "joiner_end":
            if event_data.get("error"):
                self.console.print(f"    [red]🤝 Joiner Error:[/red] {event_data.get('error_message')}")
            else:
                self.console.print(f"    [green]🤝 Joiner End.[/green] Final response might be available.")
        elif event_type == "graph_loop_end":
            self.console.print(f"  [cyan]🔄 Graph Loop End[/cyan] ({event_data.get('loop_count')})")
            if event_data.get("output_state", {}).get("needs_replanning"):
                 self.console.print("    [yellow]↪️ Replanning indicated.[/yellow]")
        
        elif event_type == "agent_run_end" or event_type == "final_result":
            self.console.print(f"[bold yellow]🏁 Agent Run End.[/bold yellow] (Event: {event_type})")

            response_to_display: Optional[str] = None
            error_to_display: Optional[str] = None
            dag_for_tree: Optional[TaskDAG] = None
            statuses_for_tree: Dict[str, str] = {}
            h_plan_for_tree: Optional[HierarchicalPlan] = None

            if event_type == "agent_run_end":
                # Compatibility for a hypothetical "agent_run_end" event structure
                final_state_payload = event_data.get("final_state", {})
                if not isinstance(final_state_payload, dict): final_state_payload = {} # ensure dict
                
                response_to_display = final_state_payload.get("final_response")
                error_to_display = final_state_payload.get("error_message")
                
                dag_candidate_dict = final_state_payload.get("dag") # Try old "dag" key
                if not dag_candidate_dict:
                    dag_candidate_dict = final_state_payload.get("task_dag") # Try new "task_dag" key
                
                if dag_candidate_dict and isinstance(dag_candidate_dict, dict):
                    try:
                        dag_for_tree = TaskDAG(**dag_candidate_dict)
                    except Exception as e:
                        self.console.print(f"[dim red]Error parsing DAG from agent_run_end: {escape(str(e))}[/dim red]")

                statuses_for_tree = final_state_payload.get("task_statuses", {})
                h_plan_payload_dict = final_state_payload.get("hierarchical_plan")
                if h_plan_payload_dict and isinstance(h_plan_payload_dict, dict):
                    try:
                        h_plan_for_tree = HierarchicalPlan(**h_plan_payload_dict)
                    except Exception as e:
                        self.console.print(f"[dim red]Error parsing HierarchicalPlan from agent_run_end: {escape(str(e))}[/dim red]")

            elif event_type == "final_result":
                response_to_display = event_data.get("answer")
                error_to_display = event_data.get("error")
                
                agent_state_payload = event_data.get("agent_state", {})
                if not isinstance(agent_state_payload, dict): agent_state_payload = {} # ensure dict

                dag_dict = agent_state_payload.get("task_dag")
                if dag_dict and isinstance(dag_dict, dict):
                    try:
                        dag_for_tree = TaskDAG(**dag_dict)
                    except Exception as e:
                        self.console.print(f"[dim red]Error parsing TaskDAG from final_result: {escape(str(e))}[/dim red]")
                
                statuses_for_tree = agent_state_payload.get("task_statuses", {})
                h_plan_dict = agent_state_payload.get("hierarchical_plan")
                if h_plan_dict and isinstance(h_plan_dict, dict):
                    try:
                        h_plan_for_tree = HierarchicalPlan(**h_plan_dict)
                    except Exception as e:
                        self.console.print(f"[dim red]Error parsing HierarchicalPlan from final_result: {escape(str(e))}[/dim red]")
            
            if error_to_display:
                self.console.print(Panel(Text(escape(str(error_to_display))), title="❌ Agent Error", border_style="red", expand=False))
            
            if dag_for_tree or h_plan_for_tree:
                self.current_dag_tree = self._build_dag_tree(dag_for_tree, statuses_for_tree, h_plan_for_tree)
                self.console.print(Panel(self.current_dag_tree, title="📊 Final Plan Structure", border_style="blue"))
            elif not response_to_display and not error_to_display:
                 self.console.print("[italic]Agent run ended with no explicit response, error, or plan output.[/italic]")

            if response_to_display:
                self.console.print(Panel(Text(escape(str(response_to_display))), title="✅ Final Response", border_style="green", expand=False))

        elif event_type == "loop_exit_no_replan":
            self.console.print("  [green]✅ Agent loop finished. No further replanning needed.[/green]")
        elif event_type == "cannot_answer_without_tools":
            self.console.print("[yellow]⚠️ Agent determined it cannot answer without tools (or appropriate tools).[/yellow]")
        elif event_type == "max_loops_reached":
            self.console.print("[bold red]🚫 Maximum replanning loops reached. Terminating current query processing.[/bold red]")
        elif event_type == "agent_critical_error":
            self.console.print(f"[bold red]🚨 AGENT CRITICAL ERROR:[/bold red] {event_data.get('message', 'Unknown critical error.')}")
        else:
            def pydantic_aware_default(obj):
                from pydantic import BaseModel
                from datetime import date, datetime

                if isinstance(obj, BaseModel):
                    return obj.model_dump(mode='json') 
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                try:
                    return list(obj) if isinstance(obj, (set, frozenset)) else obj 
                except TypeError:
                    pass
                
                try:
                    return str(obj) 
                except Exception:
                    return f"<Unserializable object: {type(obj).__name__}>"

            try:
                json_output = json.dumps(event_data, indent=2, default=pydantic_aware_default)
                self.console.print(f"[magenta]STREAM EVENT ({event_type}):[/magenta] {json_output}")
            except Exception as e_json:
                self.console.print(f"[magenta]STREAM EVENT ({event_type}) - Error serializing to JSON:[/magenta] {escape(str(e_json))}")
                self.console.print(f"[magenta]Raw event data (might be incomplete/unserializable):[/magenta] {escape(str(event_data))}")

    def _handle_run(self, args):
        pass

def main():
    """Main entry point for the Mochi CLI."""
    cli = MochiCLI()
    cli.run_cli()

if __name__ == "__main__":
    main() 