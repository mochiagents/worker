# Mochi Worker Agent

A sophisticated AI-powered task execution system that decomposes user queries into actionable plans and executes them using external tools through the Model Context Protocol (MCP).

## 🚀 Features

- **Intelligent Task Planning**: Automatically breaks down complex queries into executable task graphs (DAGs)
- **Hierarchical Planning**: Supports multi-phase execution for complex workflows
- **MCP Tool Integration**: Seamlessly connects to external tools via Model Context Protocol
- **Smart Error Recovery**: Automatic DAG repair and replanning when tasks fail
- **Multiple Interfaces**: CLI, REST API, and programmatic interfaces
- **Conversation Management**: Supports multi-turn conversations with context
- **Parallel Execution**: Concurrent task execution with dependency management
- **Semantic Tool Recommendation**: Uses embeddings to suggest relevant tools

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Development](#development)
- [TODO & Roadmap](#todo--roadmap)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.13+
- PyTorch (for embedding models)
- Access to LLM providers (Google Gemini, OpenAI, or Anthropic)

### Install from Source

```bash
git clone https://github.com/mochiagents/worker.git
cd worker
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Configure the Agent

Copy the default configuration and customize it:

```bash
cp worker_config.yaml my_config.yaml
```

Edit `my_config.yaml` to add your API keys and configure MCP tool servers:

```yaml
agent_id: "my-mochi-worker"

llm_profiles:
  default_planner:
    provider: google
    model: gemini-2.5-flash-preview-05-20
    api_key: "YOUR_GEMINI_API_KEY"
    temperature: 0.5

mcp_tool_servers:
  - name: "toolkit"
    endpoint_url: "http://localhost:8001/mcp"
    description: "Default toolkit server"
```

### 2. Start an MCP Tool Server

The agent requires at least one MCP server providing tools. Example using a hypothetical toolkit server:

```bash
# Start your MCP server (implementation depends on your tools)
python -m your_mcp_server --port 8001
```

### 3. Run the Agent

#### CLI Interface

```bash
# Single query
python -m worker.interfaces.dev_cli run "What's the weather like today?"

# Interactive mode
python -m worker.interfaces.dev_cli run -i

# With custom config
python -m worker.interfaces.dev_cli run "Analyze sales data" -c my_config.yaml
```

#### Programmatic Usage

```python
from worker.agent.main import MochiAgent
import asyncio

async def main():
    agent = MochiAgent(config_path="my_config.yaml")
    await agent.start()
    
    result = await agent.run_query("Find recent AI research papers")
    print(result["final_response"])
    
    await agent.shutdown()

asyncio.run(main())
```

#### REST API

```bash
# Start the API server
python -m worker.api.main

# Or using the agent's built-in server
python -m worker.interfaces.dev_cli api-server --host 0.0.0.0 --port 8000
```

## ⚙️ Configuration

The agent is configured via YAML files. Key sections:

### LLM Profiles

Define language model configurations:

```yaml
llm_profiles:
  planner_model:
    provider: google  # google, openai, anthropic
    model: gemini-2.5-flash-preview-05-20
    api_key: "YOUR_API_KEY"
    temperature: 0.5
    max_tokens: 32768
  
  joiner_model:
    provider: openai
    model: gpt-4-turbo
    api_key: "YOUR_OPENAI_KEY"
    temperature: 0.7
```

### Planning Settings

Configure the planning behavior:

```yaml
planner:
  llm_profile_name: planner_model
  max_dag_depth: 5
  max_sequential_tasks_per_tool: 1
  tool_recommendation_top_n: 5
  semantic_score_min_threshold: 0.1
  embedding_model_name: 'all-MiniLM-L6-v2'
```

### MCP Tool Servers

Configure external tool servers:

```yaml
mcp_tool_servers:
  - name: "web_tools"
    endpoint_url: "http://localhost:8001/mcp"
    schema_discovery_url: "http://localhost:8001/mcp/schema"
    request_timeout_seconds: 30
    max_retries: 3
    description: "Web search and scraping tools"
    
  - name: "data_tools"
    endpoint_url: "http://localhost:8002/mcp"
    auth_token: "your-auth-token"
    custom_headers:
      X-API-Version: "v2"
```

### Execution Settings

Control task execution behavior:

```yaml
execution:
  max_parallel_tasks: 3
  max_retries: 2
  initial_retry_delay_seconds: 1.0
  retry_backoff_factor: 2.0
  default_task_execution_timeout_seconds: 60.0
```

## 🔧 Usage

### Recent Improvements

#### Enhanced MCP Timeout Handling
- **Circuit Breaker Pattern**: Automatically opens circuit after repeated failures, preventing cascading issues
- **Exponential Backoff with Jitter**: Smart retry logic that reduces server load during recovery
- **Operation-Specific Timeouts**: Different timeout values for tool calls (longer) vs schema fetches (shorter)
- **Better Error Classification**: Distinguishes between retryable and non-retryable errors

#### Robust DAG Validation
- **Enhanced Cycle Detection**: Now provides exact cycle paths in error messages
- **Self-Dependency Detection**: Catches tasks that depend on themselves
- **Orphaned Reference Detection**: Identifies dependencies pointing to non-existent tasks
- **Detailed Error Messages**: Clear explanations with task details and suggestions for fixes

#### Tool Schema Caching
- **TTL-Based Caching**: Schemas cached for 1 hour by default (configurable)
- **LRU Eviction**: Automatic cleanup of old cache entries
- **Fallback Mechanisms**: Uses last successful schemas when servers are unavailable
- **Cache Statistics**: Monitor cache hit rates and performance
- **Manual Cache Control**: Force refresh or invalidate specific servers

### CLI Commands

```bash
# Run queries
mochi run "Find and summarize AI papers"
mochi run -i  # Interactive mode
mochi run "complex task" -v  # Verbose output

# Configuration management
mochi config show
mochi config create -f new_config.yaml
mochi config update -f config.yaml -k "planner.temperature" --val "0.8"

# Status and monitoring
mochi status
mochi logs -F  # Follow logs
mochi logs -n 50 -l INFO

# API testing
mochi api /tasks -m POST -d '{"query": "test task"}'
```

### Task Types

The agent supports various task patterns:

#### Direct Answers
For queries that don't require external tools:
```json
{
  "id": "direct_answer_1",
  "server_id": null,
  "tool_name": "direct_answer",
  "inputs": {"answer_text": "The answer is..."},
  "dependencies": []
}
```

#### Tool-Based Tasks
For queries requiring external capabilities:
```json
{
  "id": "search_task",
  "server_id": "web_tools",
  "tool_name": "tavily_search",
  "inputs": {"query": "AI research 2024"},
  "dependencies": []
}
```

#### Dependent Tasks
Tasks that use outputs from previous tasks:
```json
{
  "id": "summary_task",
  "server_id": "text_tools",
  "tool_name": "summarize",
  "inputs": {"text": "$result.search_task.results.0.content"},
  "dependencies": ["search_task"]
}
```

### Conversation Management

```python
# Managed conversations maintain context
conversation_id = "conv_123"

result1 = await agent.run_managed_query(
    "Find AI papers", 
    conversation_id=conversation_id
)

result2 = await agent.run_managed_query(
    "Summarize the first one",  # References previous context
    conversation_id=conversation_id
)
```

## 🏗 Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Planner     │───▶│    Executor     │───▶│     Joiner      │
│                 │    │                 │    │                 │
│ • Query analysis│    │ • Task execution│    │ • Result synth  │
│ • DAG generation│    │ • MCP tool calls│    │ • Error analysis│
│ • Tool selection│    │ • Parallel proc │    │ • Replanning    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       │                       │
         │                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DAG Editor    │    │   MCP Clients   │    │   Response      │
│                 │    │                 │    │                 │
│ • DAG repair    │    │ • HTTP/SSE/STDIO│    │ • Final answer  │
│ • Task updates  │    │ • Schema disco  │    │ • Status info   │
│ • Dependencies  │    │ • Error handling│    │ • Debug data    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Workflow States

The agent uses a LangGraph-based state machine:

1. **Initialize**: Set up agent state and components
2. **Planner**: Analyze query and generate task DAG
3. **Execution**: Run tasks with dependency management
4. **Joiner**: Synthesize results and determine next steps
5. **Response**: Format final output for user

### Data Models

Key Pydantic models:

- **`TaskNode`**: Individual task with inputs, dependencies, status
- **`TaskDAG`**: Collection of tasks forming execution graph
- **`AgentState`**: Complete agent state during execution
- **`ToolExecutionResult`**: Standardized task execution results
- **`HierarchicalPlan`**: Multi-phase execution plans

## 📚 API Reference

### REST API Endpoints

#### Tasks
- `POST /tasks` - Submit a new task
- `GET /tasks/{task_id}` - Get task status
- `DELETE /tasks/{task_id}` - Cancel task

#### Conversations  
- `POST /conversations` - Start new conversation
- `POST /conversations/{conv_id}/messages` - Send message
- `GET /conversations/{conv_id}` - Get conversation history

#### Control
- `GET /control/status` - Get agent status
- `POST /control/shutdown` - Shutdown agent
- `GET /control/config` - Get configuration

### Python API

```python
from worker.agent.main import MochiAgent

# Initialize agent
agent = MochiAgent(config_path="config.yaml")
await agent.start()

# Run single query
result = await agent.run_query("Your query here")

# Run managed conversation
result = await agent.run_managed_query(
    query="Your query",
    conversation_id="conv_123",
    stream_callback=callback_func
)

# Direct DAG execution
result = await agent.run_direct_query(
    "Query", 
    initial_state={"custom": "state"}
)
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone <repo-url>
cd mochi/worker

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=worker --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

```bash
# Linting
flake8 worker/
black worker/
isort worker/

# Type checking
mypy worker/
```

### Creating MCP Tool Servers

Example MCP server implementation:

```python
from fastapi import FastAPI
from worker.core.models import ToolSchema

app = FastAPI()

@app.get("/mcp/schema")
async def get_schema():
    return {
        "tools": [
            {
                "tool_name": "my_tool",
                "description": "Does something useful",
                "inputs": [
                    {
                        "name": "input_param",
                        "type": "string",
                        "description": "Input parameter",
                        "required": True
                    }
                ]
            }
        ]
    }

@app.post("/mcp/tools/my_tool")
async def my_tool(input_param: str):
    # Tool implementation
    result = do_something(input_param)
    return {"result": result}
```

## 📝 TODO & Roadmap

### High Priority

- [ ] **Documentation**: Complete API documentation and tutorials
- [ ] **Testing**: Comprehensive test suite (unit, integration, e2e)
- [ ] **Error Handling**: Improve error recovery and user feedback
- [ ] **Configuration Validation**: Better config validation and error messages
- [ ] **Performance**: Optimize DAG generation and execution times
- [ ] **Monitoring**: Add metrics, telemetry, and observability features

### Medium Priority

- [ ] **Security**: Authentication, authorization, and input sanitization
- [ ] **Caching**: Cache tool schemas, embeddings, and LLM responses  
- [ ] **Tool Discovery**: Automatic MCP server discovery and registration
- [ ] **Streaming**: Real-time streaming of task execution updates
- [ ] **Web UI**: Browser-based interface for agent interaction
- [ ] **Plugin System**: Extensible plugin architecture for custom behaviors

### Low Priority

- [ ] **Multi-Agent**: Support for multiple cooperating agents
- [ ] **Workflow Templates**: Pre-defined workflow templates
- [ ] **Database Integration**: Persistent storage for conversations and results
- [ ] **Kubernetes**: Helm charts and K8s deployment guides
- [ ] **CLI Improvements**: Better progress indicators and output formatting
- [ ] **Configuration UI**: Web-based configuration management

### Code Quality Improvements

- [ ] **Type Safety**: Complete type annotations throughout codebase
- [ ] **Logging**: Structured logging with better event correlation
- [ ] **Error Recovery**: More sophisticated DAG repair strategies
- [ ] **Resource Management**: Better cleanup of MCP connections and processes
- [ ] **Async Optimization**: Optimize async/await patterns and concurrency
- [ ] **Memory Management**: Prevent memory leaks in long-running processes

### Known Issues

- [x] **MCP Timeouts**: ~~Better handling of MCP server timeouts~~ - **COMPLETED**: Enhanced with circuit breaker pattern, exponential backoff with jitter, and operation-specific timeouts
- [x] **DAG Validation**: ~~More robust DAG cycle detection~~ - **COMPLETED**: Enhanced cycle detection with path tracking, detailed error messages, and additional validations for self-dependencies and orphaned references
- [x] **Tool Schema Caching**: ~~Tool schemas should be cached between runs~~ - **COMPLETED**: Implemented comprehensive caching with TTL, LRU eviction, cache statistics, and fallback mechanisms
- [ ] **CLI Help**: Some CLI commands need better help text
- [ ] **Config Examples**: Need more example configurations

### Features Partially Implemented

- [x] **Heartbeat System**: ~~Framework exists but needs completion~~ - **COMPLETED**: Dedicated heartbeat system fully integrated
- [ ] **State Persistence**: Basic framework but needs full implementation  
- [ ] **Tool Recommendation**: Works but could use ML improvements
- [ ] **Conversation Context**: Basic support, needs enhancement
- [ ] **API Security**: Basic API key auth, needs improvement

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for public APIs
- Include tests for new features
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Documentation**: [Full Documentation](link-to-docs)

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [Model Context Protocol](https://github.com/modelcontextprotocol/mcp) - Tool integration standard

---

**Mochi Worker Agent** - Intelligent AI task execution with tool integration
