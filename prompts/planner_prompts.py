import json
from typing import List, Dict, Optional, Any, Union
import logging

from langchain_core.prompts import (
    PromptTemplate,
)

# Placeholder for ServerToolSchemaGroup if direct import is an issue
# In a typed environment, this would be: from worker.core.models import ServerToolSchemaGroup
ServerToolSchemaGroup = Any

CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS = """
CRITICAL: You MUST ONLY use tools that are explicitly listed in the 'Available Tools' section below. Do NOT invent or hallucinate tool names like "ai_task", "direct_llm_call", "analyze_data", etc. If you need capabilities not provided by the available tools, you should either:
- Use the "direct_answer" special tool (with server_id: null) if you can answer directly
- Use the "cannot_answer_without_tools" special tool (with server_id: null) if you need tools that aren't available

CRITICAL: For any tool you use, you MUST strictly adhere to its input schema as defined in the 'Available Tools' section.
- Only use the exact parameter names specified in the tool's `inputs` list.
- Do NOT invent or use any parameter names not explicitly defined in the schema for that tool.
- Ensure the data type you provide for each input matches the `type` specified in its schema (e.g., string, array, object).
- Pay close attention to the `description` of each input parameter to understand what kind of data it expects.

If tools are available (check the 'Available Tools' section):
    Each task using a tool MUST include the following fields. DO NOT use "task_id" or "tool_code".
    - "id" (string): A unique identifier for the task (e.g., "task_1", "search_step"). Ensure it's a string.
    - "server_id" (string): The identifier of the MCP Tool Server that provides the tool (must match one from the 'Available Tools' section).
    - "tool_name" (string): The specific name of the tool to be called on the `server_id` (must match a tool name within that server's schema).
    - "priority" (string, optional): The priority of the task, can be "high", "medium", or "low". Defaults to "medium" if not specified. Helps the agent decide execution order for parallelizable tasks.
    - "inputs" (object): A JSON object where keys are input parameter names (as defined in the tool's schema) and values are the inputs.
        - Inputs can be literal values (strings, numbers, booleans).
        - Inputs can reference outputs from other tasks in the DAG using a special string format starting with "$result.":
            - To use the *entire output* of a preceding task: "$result.<dependency_task_id>"
            - To use a *specific field* from a preceding task's output: "$result.<dependency_task_id>.<field_path>"
                - Use dot notation for the field path to access nested fields (e.g., "$result.task_1.user.address.city", "$result.search_task.results.0.id" for array index 0).
                - Ensure the referenced task ID exists and is a valid dependency.
                - The extracted value (whole output or specific field) MUST be compatible with the type expected by the consuming tool's input parameter (refer to the tool's input schema).
    - "dependencies" (array of strings): A list of task `id`s (strings) that must be completed before this task can start. This list should be empty if there are no dependencies. Ensure all dependency IDs exist in the DAG and do not create circular dependencies.

**Example of a Correctly Formatted Task JSON Object:**
```json
{
    "id": "search_for_topic",
    "server_id": "toolkit_server_01",
    "tool_name": "tavily_search",
    "inputs": {"query": "latest advancements in AI"},
    "dependencies": []
}
```
**DO NOT include a "tool_code" field. Use "id", "server_id", and "tool_name" as shown above.**
"""

SYSTEM_MESSAGE_CONTENT = """You are an expert AI task planner. Your role is to decompose a user's request into a sequence of tasks that can be executed by an AI agent. These tasks will be represented as a Directed Acyclic Graph (DAG) in JSON format. Adhere strictly to the provided JSON schema and instructions. Ensure all task IDs are unique strings.

If you are provided with an EXISTING_TASK_STATUSES section, it represents the current state of a plan you might be augmenting or modifying. Use this information to inform your new plan.
{existing_task_statuses_section}
"""

HIGH_LEVEL_PHASE_SYSTEM_MESSAGE_CONTENT = """You are an expert AI high-level planner. Your primary role is to analyze a complex user request and decompose it into a sequence of distinct, logical phases. Each phase should represent a major step towards fulfilling the overall request. Focus on identifying the main stages of work, not the detailed tool calls within them. Adhere strictly to the provided JSON schema for outputting these phases."""

HIGH_LEVEL_PHASE_GENERATION_INSTRUCTIONS = """
**Goal: Decompose the User Request into High-Level Phases**

Based on the user's request, identify a sequence of high-level phases required to address it.
Each phase must be a JSON object with the following fields:
- "phase_id" (string): A unique identifier for the phase (e.g., "phase_1_research", "phase_2_analysis"). Ensure it's a string and unique within the list of phases.
- "description" (string): A concise but clear description of what this phase aims to achieve. This description will be used later to generate detailed tasks for this phase.

**Considerations:**
- Think about the logical flow of work needed to satisfy the user's overall goal.
- Aim for 2-5 major phases for most complex requests. Too few phases might not be helpful, and too many might be overly granular for this high-level planning stage.
- The output should be a list of these phase objects.
- Do NOT define specific tools or tasks at this stage. Focus only on the high-level breakdown.

**Example User Request:** "Analyze our company's last quarter sales data, identify key trends, and generate a presentation summarizing the findings for the leadership team."

**Example JSON Output for Phases:**
```json
{
  "phases": [
    {
      "phase_id": "phase_1_data_collection",
      "description": "Collect and consolidate all sales data from the last quarter from various sources."
    },
    {
      "phase_id": "phase_2_trend_analysis",
      "description": "Analyze the consolidated sales data to identify key performance trends, growth areas, and areas of concern."
    },
    {
      "phase_id": "phase_3_presentation_prep",
      "description": "Prepare a presentation summarizing the identified trends and key findings for the leadership team."
    }
  ]
}
```
"""

HIGH_LEVEL_PHASE_OUTPUT_FORMAT_INSTRUCTIONS = """
Respond ONLY with the generated phases in a single, valid JSON object. Do NOT include any other text, explanations, or markdown formatting (like ```json ... ```) outside of the JSON structure itself.
The root of the JSON object must be: `{{"phases": [...]}}`
"""

QUERY_TYPE_CLASSIFICATION_SYSTEM_MESSAGE = """You are an AI assistant helping to determine the nature of a user's request.
Based on the user's current query and the preceding conversation context (if any), classify the query.
Your response MUST be a single JSON object with a key "query_type" and one of the following string values:
- "COMPLEX": The query is a new, complex task requiring multi-phase hierarchical planning.
- "SIMPLE": The query is a new, simple task likely addressable by a single-level plan (using tools or direct answer).
- "NO_PLAN": The query does not require a task plan (e.g., it's a greeting, conversational filler like 'how are you?', a simple continuation request like 'continue', or unresolvable).
"""

QUERY_TYPE_CLASSIFICATION_INSTRUCTIONS = """
Analyze the user's current query in the context of the conversation history.
Classify the query into one of the types defined in the system message: COMPLEX, SIMPLE, or NO_PLAN.

**Considerations for "COMPLEX":**
- Does it clearly involve multiple distinct goals?
- Does it require a sequence of dependent high-level operations (e.g., research THEN analyze THEN report)?
- Does it imply a project-like structure?

**Considerations for "SIMPLE":**
- Can it likely be achieved with a few tool calls in a relatively flat structure?
- Is it a request for a specific piece of information or calculation that might need one or two tool calls?
- Is it a request for a direct answer you can provide confidently without tools?

**Considerations for "NO_PLAN":**
- Is it conversational filler (e.g., "hello", "thanks", "how are you?")?
- Is it a simple request to continue the previous output (e.g., "continue", "go on")?
- Is the query too vague or nonsensical to form a plan?
- Is it not a question or a request for information that requires a plan or execution of tasks like a tool call?

User's Current Query: {query}

Conversation Context (previous turns, most recent first):
{conversation_context}

Respond with a single JSON object as specified in the system message. Example: {{"query_type": "SIMPLE"}}
"""

BASE_DAG_INSTRUCTIONS = """
**Primary Goal: Fulfill the User's Request Efficiently**

**1. Assess the Query for Direct Answering:**
   - Before defaulting to tools, critically evaluate if the user's query can be answered directly using your own knowledge and capabilities.
   - For requests like simple greetings, conversational remarks, requests for general knowledge you likely possess (e.g., "Tell me a fun fact", "What is the capital of France?"), or simple calculations you can perform internally, you SHOULD prioritize generating a single task with:
     - `"id"`: "direct_answer_1"
     - `"server_id"`: null (JSON null, not the string "null")
     - `"tool_name"`: "direct_answer"
     - `"inputs"`: {{"answer_text": "Your complete and direct answer here..."}}
     - `"dependencies"`: []
   - Only proceed to step 2 (Tool-Based Planning) if the query clearly requires external information retrieval, complex computation beyond your direct capabilities, interaction with an external system, or actions that necessitate one of the 'Available Tools'.

**2. Tool-Based Planning (If Direct Answering is Not Suitable):**
   Based on the user's request and the available tools (if any), generate a DAG of tasks.
   Each task in the DAG must be a JSON object.
{CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS}

**Think step-by-step to construct the plan. Critically examine the user's request: if it involves multiple distinct actions or requires a sequence of operations, break it down into a series of smaller, interconnected tasks. Each task should ideally represent a single, clear action.**

**Example of Breaking Down a Complex Request:**
User Request: "Find recent AI research papers on reinforcement learning, summarize the top 3, and email the summaries to me."

This request involves multiple steps. A good plan would break this into:
1.  A task to search for AI research papers on reinforcement learning (e.g., using a `web_search` tool).
2.  A task to analyze the search results and identify the top 3 relevant papers (e.g., using a `text_analysis` or `content_extraction` tool, potentially repeated or with internal logic if the tool processes one paper at a time).
3.  For each of the top 3 papers:
    a.  A task to retrieve the full content of the paper (if not already done).
    b.  A task to summarize the paper (e.g., using a `summarization` tool).
4.  A task to compile the summaries.
5.  A task to send an email with the compiled summaries (e.g., using an `email` tool).

**Example of Using Task Output References:**
If task "search_task" uses a search tool that returns:
```json
{
  "query": "AI research papers",
  "results": [
    {{"title": "Paper 1", "url": "...", "content": "..."}},
    {{"title": "Paper 2", "url": "...", "content": "..."}}
  ]
}
```

To reference the first result's title in a subsequent task:
- Use: "$result.search_task.results.0.title"
- NOT: "$result.search_task.answer" (if 'answer' field doesn't exist)

Always check the tool's output schema to understand what fields are available!

**Example of Extracting Values from Search Results:**
If you need to extract a value from search results, consider using an intermediate direct_answer task to interpret the results first.

CRITICAL: When the goal is to take output from a search tool (e.g., `tavily_search`, which often returns a list of results like `results: [{{"title": "...", "content": "..."}}]`) and use a piece of that information (like a specific text snippet) as an input for a tool that expects simple string values in a structured way (e.g., `google_sheets_append` for cell values, or a database update tool), you MUST use the following 3-step pattern:
1.  **Search Task**: The initial tool call to the search service (e.g., `tavily_search`). This task's output will contain a `results` field, which is an array of result objects.
2.  **Extraction Task (`direct_answer`)**: A subsequent `direct_answer` task. The `answer_text` input for this `direct_answer` task MUST be a prompt that takes the *entire `results` array* from the Search Task's output (e.g., by referencing `"$result.YOUR_LOCAL_SEARCH_TASK_ID.results"`). The prompt must then instruct the LLM to process this array: check if it's empty, and if not, access the relevant content (e.g., from the first result item like `results[0].content`) to extract or summarize the specific information needed into a clean, simple string. If no relevant data is found, it should output a clear indicator like "Data not found".
3.  **Update Task**: The final tool call (e.g., `google_sheets_append`) which takes the simple string output from the Extraction Task (`"$result.YOUR_LOCAL_EXTRACTION_TASK_ID.answer_text"`) as its input.
Do NOT attempt to directly reference complex objects or specific deep fields from search results (like `.results[0].snippet` or `.results[0].content`) within the input parameters of tools like `google_sheets_append`. This direct referencing will likely fail. The `direct_answer` extraction step is mandatory for this scenario.

**Important Pattern - Extracting Values for Structured Data Updates:**
When you need to extract specific information from search results (which are often complex objects or arrays) to update structured data (like spreadsheets or database entries that expect simple string/scalar values), you MUST follow this pattern:

1.  **Search for information**: Use appropriate search tools (e.g., `tavily_search`) to find the data. This task will output a complex object (e.g., with a `results` array, where each item might have a `content` field). Define this task with a simple, local ID (e.g., "search_for_country_data").
2.  **Extract/Interpret/Summarize the Value**: Create a `direct_answer` task whose input (`answer_text`) is a carefully constructed prompt. This prompt should take the *entire `results` array* from the search task's output (e.g., by referencing `"$result.search_for_country_data.results"`) and instruct the LLM to:
    a. Check if the `results` array is empty.
    b. If not empty, examine the `content` field of the first result item (i.e., `results[0].content`).
    c. Extract or summarize the required information into a concise string suitable for the target cell/field.
    d. If the `results` array is empty or the desired information cannot be found, output "Data not found" or a similar clear indicator.
    - Task ID for this extraction step should also be simple and local (e.g., "extract_specific_value").
    - `tool_name`: "direct_answer"
    - `server_id`: null
    - `inputs`: {{ "answer_text": "Analyze the following search results: $result.search_for_country_data.results. If the results are not empty, extract the key information from the 'content' of the first result. If no results or key information found, state 'Data not found'." }} (CRITICAL: Ensure you reference the correct field from the search result, typically `content` for the main text body. Do NOT invent fields like 'snippet' or 'answer' if they are not part of the search tool's actual output schema for its results array items.).
3.  **Update the Target**: Use the output of the `direct_answer` task (which will be `"$result.extract_specific_value.answer_text"`) as the input for your structured data update tool (e.g., `google_sheets_append`). This output MUST be a simple string.

**Example - Appending a Search Result Summary to a Sheet (Illustrating Local IDs within a Phase):**
This example illustrates the 3-step pattern and correct schema adherence for `google_sheets_append`.
Assume you are generating tasks for a phase. The IDs used below (`search_country_gdp`, `extract_gdp_value`, `append_extracted_data_to_sheet`) are LOCAL to this phase plan.

*Task 1 (within current phase): Search for Data (e.g., GDP)*
```json
{
  "id": "search_country_gdp",
  "server_id": "toolkit",
  "tool_name": "tavily_search",
  "inputs": {{"query": "GDP per capita for SpecificCountry"}},
  "dependencies": []
}
```

*Task 2 (within current phase): Extract GDP Value*
```json
{
  "id": "extract_gdp_value",
  "server_id": null,
  "tool_name": "direct_answer",
  "inputs": {
    "answer_text": "Analyze the search results provided here: $result.search_country_gdp.results. From these results, what is the GDP per capita for SpecificCountry? If the results are empty or the GDP per capita is not found, state 'GDP not found'. Focus on the 'content' of the first relevant result if available."
  },
  "dependencies": ["search_country_gdp"]
}
```

*Task 3 (within current phase): Update Spreadsheet*
```json
{
  "id": "append_extracted_data_to_sheet",
  "server_id": "toolkit",
  "tool_name": "google_sheets_append",
  "inputs": {
    "spreadsheet_id": "your_spreadsheet_id_here",
    "range": "Sheet1!A:B",
    "values": [
      ["France GDP 2024", "$result.extract_gdp_value.answer_text"]
    ],
    "value_input_option": "USER_ENTERED"
  },
  "dependencies": ["extract_gdp_value"]
}
```

This ensures that tools like `google_sheets_append` receive simple, clean string data for their cells, formatted according to the tool's `values` input schema. DO NOT directly pass complex objects or arrays from search results into spreadsheet cells or invent input parameter names. ALWAYS use an intermediate `direct_answer` task to create a suitable string representation first, referencing the correct output fields (like `content`) from the search results. Use the correct input parameter names (like `"values"` for `google_sheets_append`).

**Note on Task References in Arrays:**
When using task references inside arrays (common in tools like `google_sheets_update`), the references will be resolved recursively. For example:
```json
"values": [["Label", "$result.previous_task.output_field"]]
```
The system will resolve the reference and replace it with the actual value before sending to the tool.

**Important for Hierarchical Planning (Generating Tasks for a Specific Phase):**
When you are generating tasks as part of a specific phase (you will be told which phase you are working on):
- All task `id`s you define in your current output (the plan for this phase) MUST be simple, unique strings that are local to this phase (e.g., "search_web", "extract_info", "update_sheet_1").
- When referencing outputs from other tasks *within this same phase plan you are currently generating*, use these same simple, local `id`s in the `$result` path (e.g., `"$result.search_web.results"`).
- Do NOT include the overall phase ID prefix (like `phase_1_` or `phase_research_`) in the task `id`s you define for this phase, nor in the `$result` references between tasks *within this phase's plan*.
- The system will automatically add the necessary phase prefixes to your locally defined task IDs and update references when it combines this phase's plan with other phases into a final global plan.
- You can, however, reference outputs from tasks that were part of *previously completed phases* using their full, globally unique, phase-prefixed IDs (these will be provided to you in the "Available Outputs from Previous Phases" section).

Organize the tasks logically to achieve the user's request. Optimize for parallel execution where possible (e.g., summarizing papers can be done in parallel once all content is fetched) by minimizing unnecessary sequential dependencies.
Ensure all necessary information for each tool is provided or sourced from a preceding task's output using the correct input referencing format.
"""

NO_TOOLS_DAG_INSTRUCTION_EXTENSION = """

IMPORTANT: No tools are currently available to the agent.
- If you (the LLM) can directly answer the user's query without needing external tools or information, generate a single task with:
    - `"id"`: "direct_answer_1"
    - `"server_id"`: null  (JSON null, not the string "null")
    - `"tool_name"`: "direct_answer"
    - `"inputs"`: `{{"answer_text": "Your complete and direct answer here..."}}`
    - `"dependencies"`: []
- If the user's query CANNOT be answered directly by you without tools, generate a single task with:
    - `"id"`: "cannot_answer_1"
    - `"server_id"`: null (JSON null, not the string "null")
    - `"tool_name"`: "cannot_answer_without_tools"
    - `"inputs"`: `{{"reason": "Explain clearly and politely why the query cannot be answered without tools (e.g., 'I need a web search tool to find current information, which is not available.')."}}`
    - `"dependencies"`: []
Do not attempt to use or reference any tools if none are listed as available.
"""

OUTPUT_FORMAT_INSTRUCTIONS = """
Respond ONLY with the generated DAG in a single, valid JSON object. Do NOT include any other text, explanations, or markdown formatting (like ```json ... ```) outside of the JSON structure itself.
The root of the JSON object must be: `{"tasks": [...]}`.
Each task object within the "tasks" list MUST conform to the schema: {"id": "...", "server_id": "...", "tool_name": "...", "inputs": ..., "dependencies": ...}.
DO NOT use "task_id". DO NOT use "tool_code".
"""

# System prompt for phase-specific planning
PHASE_SPECIFIC_SYSTEM_MESSAGE_TEMPLATE = """You are an expert AI task planner creating a detailed plan for a specific phase of a larger operation.
This phase has the ID: '{phase_id_str}'.
Adhere strictly to all instructions, especially regarding task ID assignment, JSON output format, and referencing outputs from previous tasks or tasks within this current phase.
"""

# User-facing instructions for phase-specific planning
PHASE_SPECIFIC_USER_INSTRUCTIONS_TEMPLATE = """
**Overall User Request:**
{original_query}

**Current Phase Objective:**
Phase ID: {phase_id}
Description: {phase_description}

**Context & Instructions:**
{conversation_history}
{available_prior_outputs_section}

**Available Tools (JSON Schema):**
{tool_schemas_str}
{few_shot_examples}

**DAG Generation Instructions for THIS PHASE (Phase ID: {phase_id}):**
Follow the general `BASE_DAG_INSTRUCTIONS` for overall planning logic, but critically, pay attention to the `Important for Hierarchical Planning` section within `BASE_DAG_INSTRUCTIONS` and the specific instructions below for how to assign task `id`s and reference outputs when planning for this specific phase (`{phase_id}`).

You are currently generating tasks for the phase with ID: '{phase_id}'.
When defining task `id`s and their $result references *within this current phase's plan*, use simple, unique, phase-local IDs (e.g., "search_data", "analyze_result").
Do NOT include the phase ID '{phase_id}' as a prefix in these local task `id`s or in $result references *between tasks of this current phase*.
For example, if you create a task with local id "my_search", reference its output as `"$result.my_search.output"` (NOT `"$result.{phase_id}_my_search.output"`).
The system will handle prefixing your local IDs with '{phase_id}_' later. Refer to 'Important for Hierarchical Planning' in the main instructions for more details.

{CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS}

**Output Format:**
{OUTPUT_FORMAT_INSTRUCTIONS}
"""

REPLANNING_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI REPLANNER.
Your goal is to analyze the user's original query, the tool schemas, conversation history, and the CONTEXT FROM THE PREVIOUS EXECUTION CYCLE (including current task statuses) to generate a NEW or MODIFIED plan (DAG).

USER'S ORIGINAL QUERY:
{query}

AVAILABLE TOOLS (JSON Schema):
{tool_schema_str}

CONVERSATION HISTORY (if any):
{conversation_history}

--- CONTEXT FROM CURRENT/PREVIOUS EXECUTION CYCLE ---
EXISTING_TASK_STATUSES (All tasks in the current DAG with their status and attempts):
{existing_task_statuses_section}

Successful Tasks (from last execution attempt, with status and attempts):
{successful_tasks}

Failed Tasks (from last execution attempt, with status and attempts):
{failed_tasks}

Available Results from Successful Tasks (JSON):
{available_results}
{failed_repair_info}
--- END OF EXECUTION CYCLE CONTEXT ---

INSTRUCTIONS:
Based on the original query, available tools, conversation history, and the full execution context (EXISTING_TASK_STATUSES, successful/failed tasks, results, and repair info), you MUST generate a NEW or MODIFIED plan (DAG) to achieve the user's goal.
- If the EXISTING_TASK_STATUSES indicates a partially complete plan, you might add new tasks, modify existing pending tasks, or re-evaluate dependencies.
- Focus on addressing reasons for previous failures (see Failed Tasks).
- Your output MUST be a JSON object representing the new or updated TaskDAG.
{CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS}
Output ONLY the JSON for the TaskDAG, adhering to the schema in the critical instructions above. Do not include any other explanatory text, greetings, or markdown formatting.
The root of the JSON object must be: `{{"tasks": [...]}}`.
"""

HIERARCHICAL_PLANNING_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI hierarchical planner.
""" # This seems to be a placeholder, ensure it's sufficient or expanded if hierarchical planning has more specific system needs.

class PlannerPromptBuilder:
    """
    Builds prompts for the Planner LLM to generate task DAGs.
    """

    def __init__(self):
        """
        Initializes the PlannerPromptBuilder.
        """
        self.logger = logging.getLogger(f"mochi.{self.__class__.__name__}")

    def _format_tool_schemas_for_prompt(self, tool_schemas: List[ServerToolSchemaGroup]) -> str:
        """Formats the list of ServerToolSchemaGroup Pydantic models into a JSON string for the prompt."""
        if not tool_schemas:
            return "No tools are currently available to the agent."
        try:
            # Assuming ServerToolSchemaGroup is a Pydantic model or can be dumped.
            # If ServerToolSchemaGroup is 'Any', this might need adjustment if the actual type isn't directly serializable.
            # For now, assuming it's a list of Pydantic models or dicts.
            schemas_as_dicts = []
            for s_group in tool_schemas:
                if hasattr(s_group, 'model_dump'):
                    schemas_as_dicts.append(s_group.model_dump(exclude_none=True))
                elif isinstance(s_group, dict): # Handle if it's already a dict
                    schemas_as_dicts.append(s_group)
                else:
                    self.logger.warning(f"Cannot serialize tool schema group of type {type(s_group)} for prompt.")
            
            return json.dumps(schemas_as_dicts, indent=2)
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error serializing tool schemas: {e}", exc_info=True)
            return "Error: Could not format tool schemas for the prompt."

    def _format_few_shot_examples(
        self, few_shot_examples: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Formats few-shot examples for inclusion in the prompt."""
        if not few_shot_examples:
            return ""

        formatted_examples = "\n\nHere are some examples of how to structure the DAG:\n"
        for i, example in enumerate(few_shot_examples):
            query = example.get("user_query", "N/A")
            tools = example.get("tool_schemas_json_str", "N/A")
            dag = example.get("expected_dag_json", "N/A")
            formatted_examples += (
                f"\n--- Example {i+1} ---\n"
                f"User Request: {query}\n"
                f"Available Tools (summary): {tools}\n"
                f"Expected DAG:\n{dag}\n"
                f"--- End Example {i+1} ---\n"
            )
        return formatted_examples

    def get_system_prompt(self) -> str:
        """Returns the system prompt string for standard planning."""
        return SYSTEM_MESSAGE_CONTENT

    def get_query_type_classification_system_prompt(self) -> str:
        """Returns the system prompt for query type classification."""
        return QUERY_TYPE_CLASSIFICATION_SYSTEM_MESSAGE

    def get_query_type_classification_prompt(self, query: str, conversation_context: Optional[str]) -> str:
        """Formats the prompt for query type classification."""
        context_str = conversation_context if conversation_context else "No previous conversation."
        pt = PromptTemplate.from_template(QUERY_TYPE_CLASSIFICATION_INSTRUCTIONS)
        formatted_prompt = pt.format(query=query, conversation_context=context_str)
        return formatted_prompt

    def get_main_prompt(
        self,
        query: str,
        tool_schemas: List[ServerToolSchemaGroup],
        conversation_context: Optional[str] = None,
        replan_context: Optional[Dict[str, Any]] = None,
        failed_repair_error: Optional[str] = None,
        failed_repair_instructions: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None,
    ) -> tuple[str, str]:
        """
        Generates the main user-facing prompt content for DAG generation,
        adapting for standard planning or replanning.
        Returns a tuple: (main_prompt_content, existing_task_statuses_section_str)
        The overall system message (SYSTEM_MESSAGE_CONTENT or REPLANNING_SYSTEM_PROMPT_TEMPLATE)
        will be prepended by the LLM calling mechanism after being formatted with existing_task_statuses_section_str.
        """
        
        formatted_tools_section = self._format_tool_schemas_for_prompt(tool_schemas)
        few_shot_examples_str = self._format_few_shot_examples(few_shot_examples)

        truncated_context = conversation_context
        if conversation_context and max_context_tokens is not None:
            tokens = conversation_context.split()
            if len(tokens) > max_context_tokens:
                tokens = tokens[-max_context_tokens:]
                truncated_context = ' '.join(tokens)
        conversation_context_str = f"\n\nFor additional context, consider the following history:\n{truncated_context}" if truncated_context else ""

        # Prepare existing_task_statuses_section_str regardless of plan type first
        current_plan_tasks_list = []
        if replan_context and 'current_plan_tasks' in replan_context:
            current_plan_tasks_list = replan_context['current_plan_tasks']
        
        existing_task_statuses_str = self._format_tasks_for_prompt(current_plan_tasks_list)
        existing_task_statuses_section_for_system_prompt_str = ""
        if existing_task_statuses_str.strip() and existing_task_statuses_str.strip().lower() != "none":
            existing_task_statuses_section_for_system_prompt_str = f"\n\n--- EXISTING_TASK_STATUSES ---\n{existing_task_statuses_str}\n--- END EXISTING_TASK_STATUSES ---"

        main_prompt_content_str = ""

        if replan_context:
            self.logger.info("Constructing REPLANNING prompt content.")
            
            successful_tasks_str = self._format_tasks_for_prompt(replan_context.get('successful_tasks', []))
            failed_tasks_str = self._format_tasks_for_prompt(replan_context.get('failed_tasks', []))
            available_results_str = json.dumps(replan_context.get('available_results', {}), indent=2)
            
            failed_repair_info_str = ""
            if failed_repair_error or (failed_repair_instructions and failed_repair_instructions.get("repair_actions")):
                failed_repair_info_str = "\n\n--- Information on Previous Failed Repair Attempt ---"
                if failed_repair_instructions and failed_repair_instructions.get("repair_actions"):
                    try:
                        instr_json = json.dumps(failed_repair_instructions, indent=2)
                        failed_repair_info_str += f"\nAttempted Repair Instructions (JSON):\n{instr_json}"
                    except TypeError:
                        failed_repair_info_str += f"\nAttempted Repair Instructions (raw): {failed_repair_instructions}"
                if failed_repair_error:
                    failed_repair_info_str += f"\nError During Repair Attempt: {failed_repair_error}"
                failed_repair_info_str += "\n--- End of Failed Repair Attempt Information ---"
            
            # The REPLANNING_SYSTEM_PROMPT_TEMPLATE is the main template here.
            # The calling code will format it with existing_task_statuses_section_for_system_prompt_str.
            # The main_prompt_content_str for replanning will be built from a template that *doesn't* include the status section itself.
            # For replanning, the system prompt contains all context including current statuses.
            # The user-facing part is effectively defined by REPLANNING_SYSTEM_PROMPT_TEMPLATE's placeholders 
            # *other than* existing_task_statuses_section, plus query, tools, history.

            # We need a user-facing template for replanning that EXCLUDES the status section, as that goes in the system prompt.
            # This part is a bit tricky. The REPLANNING_SYSTEM_PROMPT_TEMPLATE itself *is* the main prompt for the user (LLM's perspective)
            # So, when Planner calls _call_llm, the 'user_prompt' will be the fully formatted REPLANNING_SYSTEM_PROMPT_TEMPLATE.
            # The existing_task_statuses_section_for_system_prompt_str is *already part* of the REPLANNING_SYSTEM_PROMPT_TEMPLATE via its {existing_task_statuses_section} placeholder.
            # Thus, main_prompt_content_str for replanning should essentially be an empty string or a simple instruction to refer to the system prompt if we were strictly separating system/user.
            # However, given _call_llm structure, we should format the full replanning prompt (which is effectively user-facing due to containing all context) here.

            pt_replan_system = PromptTemplate.from_template(REPLANNING_SYSTEM_PROMPT_TEMPLATE)
            main_prompt_content_str = pt_replan_system.format(
                query=query,
                tool_schema_str=formatted_tools_section,
                conversation_history=conversation_context_str,
                existing_task_statuses_section=existing_task_statuses_section_for_system_prompt_str, # This now correctly slots in here
                successful_tasks=successful_tasks_str,
                failed_tasks=failed_tasks_str,
                available_results=available_results_str,
                failed_repair_info=failed_repair_info_str
            )
            # For replanning, the system prompt for _call_llm should perhaps be a very minimal "You are an AI replanner."
            # And this main_prompt_content_str becomes the full user prompt.
            # Let's adjust _call_llm in Planner to handle this: if is_replan, system_prompt is minimal.
            # For now, this function returns the 'user part'. So for replan, it's the content of REPLANNING_SYSTEM_PROMPT_TEMPLATE *excluding* the system parts.

            # The actual system message for replanning will be "You are an expert AI REPLANNER." (from REPLANNING_SYSTEM_PROMPT_TEMPLATE's first line)
            # The content below forms the "user" part of the prompt for the LLM.
            REPLAN_USER_FACING_CONTENT_TEMPLATE = """USER'S ORIGINAL QUERY:
{query}

AVAILABLE TOOLS (JSON Schema):
{tool_schema_str}

CONVERSATION HISTORY (if any):
{conversation_history}

--- CONTEXT FROM CURRENT/PREVIOUS EXECUTION CYCLE ---
EXISTING_TASK_STATUSES (All tasks in the current DAG with their status and attempts):
{existing_task_statuses_section} 

Successful Tasks (from last execution attempt, with status and attempts):
{successful_tasks}

Failed Tasks (from last execution attempt, with status and attempts):
{failed_tasks}

Available Results from Successful Tasks (JSON):
{available_results}
{failed_repair_info}
--- END OF EXECUTION CYCLE CONTEXT ---

INSTRUCTIONS:
Based on the original query, available tools, conversation history, and the full execution context (EXISTING_TASK_STATUSES, successful/failed tasks, results, and repair info), you MUST generate a NEW or MODIFIED plan (DAG) to achieve the user's goal.
- If the EXISTING_TASK_STATUSES indicates a partially complete plan, you might add new tasks, modify existing pending tasks, or re-evaluate dependencies.
- Focus on addressing reasons for previous failures (see Failed Tasks).
- Your output MUST be a JSON object representing the new or updated TaskDAG.
{CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS}
Output ONLY the JSON for the TaskDAG, adhering to the schema in the critical instructions above. Do not include any other explanatory text, greetings, or markdown formatting.
The root of the JSON object must be: `{{"tasks": [...]}}`.
"""
            pt_replan_user_content = PromptTemplate.from_template(REPLAN_USER_FACING_CONTENT_TEMPLATE)
            main_prompt_content_str = pt_replan_user_content.format(
                query=query,
                tool_schema_str=formatted_tools_section,
                conversation_history=conversation_context_str,
                existing_task_statuses_section=existing_task_statuses_section_for_system_prompt_str,
                successful_tasks=successful_tasks_str,
                failed_tasks=failed_tasks_str,
                available_results=available_results_str,
                failed_repair_info=failed_repair_info_str,
                CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS=CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS # Add constant here
            )
            # The existing_task_statuses_section_for_system_prompt_str is no longer needed separately for the system prompt
            # as it's now directly formatted into the user-facing part of the replan prompt.
            # However, the `SYSTEM_MESSAGE_CONTENT` for standard planning still uses it.
            # The `REPLANNING_SYSTEM_PROMPT_TEMPLATE`'s *first line* is used as the system message, and the rest is user.

        else: # Standard planning
            self.logger.info("Constructing STANDARD planning prompt content.")

            current_dag_instructions = BASE_DAG_INSTRUCTIONS # This now includes the CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS
        if formatted_tools_section == "No tools are currently available to the agent.":
            current_dag_instructions += NO_TOOLS_DAG_INSTRUCTION_EXTENSION

        prompt_parts = [
                f"User Request:\\n{query}",
                conversation_context_str,
                # DO NOT ADD existing_task_statuses_section_str here for standard planning; it goes into the system prompt.
                "\\nAvailable Tools (JSON Schema):",
            formatted_tools_section,
            few_shot_examples_str,
                f"\\nDAG Generation Instructions:\\n{current_dag_instructions}",
                f"\\nOutput Format:\\n{OUTPUT_FORMAT_INSTRUCTIONS}"
        ]
        main_prompt_content_str = "\\n".join(filter(None, prompt_parts))
        
        self.logger.debug(f"Generated main prompt content:\\n{main_prompt_content_str[:500]}... Existing task status section for system prompt: {existing_task_statuses_section_for_system_prompt_str[:200]}...")
        return main_prompt_content_str, existing_task_statuses_section_for_system_prompt_str # Return tuple

    def get_high_level_phase_system_prompt(self) -> str:
        """Returns the system prompt string for high-level phase generation."""
        return HIGH_LEVEL_PHASE_SYSTEM_MESSAGE_CONTENT

    def get_high_level_phase_generation_prompt(
        self,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Generates the main prompt string for high-level phase generation."""
        conversation_context_str = f"\n\nFor additional context, consider the following history:\n{conversation_context}" if conversation_context else ""
        prompt_parts = [
            f"User Request:\n{query}",
            conversation_context_str,
            f"\nPhase Generation Instructions:\n{HIGH_LEVEL_PHASE_GENERATION_INSTRUCTIONS}",
            f"\nOutput Format:\n{HIGH_LEVEL_PHASE_OUTPUT_FORMAT_INSTRUCTIONS}"
        ]
        final_prompt_string = "\n".join(filter(None, prompt_parts))
        self.logger.debug(f"Generated High-Level Phase Prompt:\n{final_prompt_string}")
        return final_prompt_string

    def get_phase_specific_dag_prompt(
        self,
        original_query: str,
        current_phase: Any, # This is a Phase object from core.models
        tool_schemas: List[ServerToolSchemaGroup],
        available_prior_outputs: Optional[Dict[str, Any]] = None, # global_task_id -> actual_output_value
        conversation_context: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        max_context_tokens: Optional[int] = None, # TODO: Implement context truncation
    ) -> str:
        """Generates the user-facing prompt for planning a specific phase in a hierarchical plan."""
        phase_id_str = current_phase.phase_id if hasattr(current_phase, 'phase_id') else "UnknownPhase"
        phase_description = current_phase.description if hasattr(current_phase, 'description') else "No description."

        formatted_tools_section = self._format_tool_schemas_for_prompt(tool_schemas)
        few_shot_examples_str = self._format_few_shot_examples(few_shot_examples)

        conversation_context_str = f"\n\nFor additional context, consider the following history:\n{conversation_context}" if conversation_context else ""

        # Format outputs from previous phases
        prior_outputs_section = "\n\n--- AVAILABLE OUTPUTS FROM PREVIOUS PHASES ---"
        if available_prior_outputs:
            prior_outputs_section += "\nYou can reference outputs from tasks in PREVIOUS phases using their globally unique IDs in the format '$result.GLOBAL_TASK_ID.output_field'."
            prior_outputs_section += "\nThe GLOBAL_TASK_ID is typically structured as 'PHASEID_ORIGINALTASKIDINPHASE'."
            prior_outputs_section += "\nAvailable outputs are listed below with their global IDs:\n"
            found_outputs = False
            for global_task_id, output_value in available_prior_outputs.items():
                found_outputs = True
                output_summary = self._summarize_output_for_prompt(output_value)
                prior_outputs_section += f"  - Global Task ID: '{global_task_id}', Output Summary: {output_summary}\n"
            if not found_outputs:
                prior_outputs_section += "No outputs from previous phases are available or they were empty.\n"
        else:
            prior_outputs_section += "No previous phases have produced outputs, or this is the first phase.\n"
        prior_outputs_section += "--- END OF AVAILABLE PREVIOUS PHASE OUTPUTS ---\n"

        # Phase-specific instructions and context
        # The PHASE_SPECIFIC_USER_INSTRUCTIONS_TEMPLATE already includes {CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS}
        user_pt = PromptTemplate.from_template(PHASE_SPECIFIC_USER_INSTRUCTIONS_TEMPLATE)
        main_prompt_content = user_pt.format(
            original_query=original_query,
            phase_id=phase_id_str,
            phase_description=phase_description,
            tool_schemas_str=formatted_tools_section,
            available_prior_outputs_section=prior_outputs_section, 
            conversation_history=conversation_context_str,
            few_shot_examples=few_shot_examples_str,
            CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS=CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS, # Explicitly pass it here
            OUTPUT_FORMAT_INSTRUCTIONS=OUTPUT_FORMAT_INSTRUCTIONS # Also pass this
        )
        
        system_message_template = PromptTemplate.from_template(PHASE_SPECIFIC_SYSTEM_MESSAGE_TEMPLATE)
        system_message_content = system_message_template.format(phase_id_str=phase_id_str)
        
        return f"{system_message_content}\n\n{main_prompt_content}"

    def _summarize_output_for_prompt(self, output_value: Any, max_length: int = 200) -> str:
        """Helper to summarize output for prompt, truncating if necessary."""
        if isinstance(output_value, str):
            return output_value[:max_length]
        elif isinstance(output_value, (dict, list)):
            return json.dumps(output_value)[:max_length]
        else:
            return str(output_value)[:max_length]

    def build_planning_prompt(
        self,
        query: str,
        tool_schemas: List[ServerToolSchemaGroup], 
        conversation_context: Optional[str] = None,
        replan_context: Optional[Dict[str, Any]] = None,
        failed_repair_error: Optional[str] = None,
        failed_repair_instructions: Optional[Dict[str, Any]] = None,
        query_type: Optional[str] = "SIMPLE", # Default based on previous state if not complex
        current_phase: Optional[Any] = None, # Phase object from core.models.Phase
        hierarchical_plan: Optional[Any] = None, # HierarchicalPlan object
        # available_prior_results: Optional[Dict[str, str]] = None # Dict of task_id to output summary string
        available_prior_outputs: Optional[Dict[str, Any]] = None, # global_task_id -> actual_output_value (aligning with get_phase_specific_dag_prompt)
    ) -> tuple[str, str]: # Returns (system_prompt, user_prompt)
        """
        Builds the appropriate prompt string for the planner LLM.
        This method decides whether to generate a prompt for a hierarchical phase,
        a replan, or a standard single-level plan.
        Returns a tuple: (system_prompt_string, user_prompt_string)
        """
        phase_id_for_log = current_phase.phase_id if current_phase and hasattr(current_phase, 'phase_id') else 'N/A'
        self.logger.debug(f"build_planning_prompt called. Query Type: {query_type}, Replan: {replan_context is not None}, Phase: {phase_id_for_log}")

        system_prompt_str: str = ""
        user_prompt_str: str = ""

        if current_phase and hierarchical_plan: # Indicates hierarchical planning for a specific phase
            self.logger.info(f"Using HIERARCHICAL PHASE-SPECIFIC prompt for phase: {phase_id_for_log}")
            
            phase_id_str = current_phase.phase_id if hasattr(current_phase, 'phase_id') else "UnknownPhase"
            system_message_template = PromptTemplate.from_template(PHASE_SPECIFIC_SYSTEM_MESSAGE_TEMPLATE)
            system_prompt_str = system_message_template.format(phase_id_str=phase_id_str)
            
            user_prompt_str = self.get_phase_specific_dag_prompt(
                original_query=query,
                current_phase=current_phase,
                tool_schemas=tool_schemas,
                available_prior_outputs=available_prior_outputs,
                conversation_context=conversation_context
            )
            # get_phase_specific_dag_prompt now returns the full prompt (system + user combined for that specific use case)
            # For consistency with the tuple return type, we split it here conceptually or adjust get_phase_specific_dag_prompt
            # Let's assume get_phase_specific_dag_prompt returns the "user" part.
            
            # Re-evaluating: get_phase_specific_dag_prompt was changed to return the full prompt.
            # For build_planning_prompt to return a tuple (system, user), it needs to construct them separately.
            
            # Let's make get_phase_specific_dag_prompt return the "user" part.
            # The system part is PHASE_SPECIFIC_SYSTEM_MESSAGE_TEMPLATE.
            
            # Re-adjusting the logic within `get_phase_specific_dag_prompt` to only return the user part
            # And then combining it here.
            # For now, let's assume `get_phase_specific_dag_prompt` returns the "user_prompt" part
            # and we get the "system_prompt" part directly.

            # Let's assume this correctly gets the 'user' part as defined in that method.
            user_prompt_str = self._get_phase_specific_user_prompt_content( # Create a helper or inline
                original_query=query,
                current_phase=current_phase,
                tool_schemas=tool_schemas,
                available_prior_outputs=available_prior_outputs,
                conversation_context=conversation_context
            )

        else: # Standard or Replan (non-hierarchical phase)
            # get_main_prompt returns: (main_prompt_content_for_user, existing_task_statuses_section_for_system_prompt)
            user_prompt_str_from_main, existing_task_statuses_section_for_system_prompt_str = self.get_main_prompt(
                query=query,
                tool_schemas=tool_schemas,
                conversation_context=conversation_context,
                replan_context=replan_context,
                failed_repair_error=failed_repair_error,
                failed_repair_instructions=failed_repair_instructions
            )
            user_prompt_str = user_prompt_str_from_main

            if replan_context:
                self.logger.info("Using REPLAN prompt system message part.")
                # The first line of REPLANNING_SYSTEM_PROMPT_TEMPLATE is the system message.
                # The rest (formatted by get_main_prompt) is the user message.
                system_prompt_str = REPLANNING_SYSTEM_PROMPT_TEMPLATE.split('\n', 1)[0] 
                # user_prompt_str is already set from get_main_prompt which formats the body of REPLANNING_SYSTEM_PROMPT_TEMPLATE
            else:
                self.logger.info("Using STANDARD prompt system message part.")
                system_template = PromptTemplate.from_template(SYSTEM_MESSAGE_CONTENT)
                system_prompt_str = system_template.format(existing_task_statuses_section=existing_task_statuses_section_for_system_prompt_str)
        
        self.logger.debug(f"Final System Prompt for LLM: {system_prompt_str[:300]}...")
        self.logger.debug(f"Final User Prompt for LLM: {user_prompt_str[:500]}...")
        return system_prompt_str, user_prompt_str

    def _get_phase_specific_user_prompt_content(
        self,
        original_query: str,
        current_phase: Any, 
        tool_schemas: List[ServerToolSchemaGroup],
        available_prior_outputs: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Helper to construct the user-facing part of a phase-specific DAG prompt."""
        phase_id_str = current_phase.phase_id if hasattr(current_phase, 'phase_id') else "UnknownPhase"
        phase_description = current_phase.description if hasattr(current_phase, 'description') else "No description."

        formatted_tools_section = self._format_tool_schemas_for_prompt(tool_schemas)
        few_shot_examples_str = self._format_few_shot_examples(few_shot_examples)
        conversation_context_str = f"\n\nFor additional context, consider the following history:\n{conversation_context}" if conversation_context else ""

        prior_outputs_section = "\n\n--- AVAILABLE OUTPUTS FROM PREVIOUS PHASES ---"
        if available_prior_outputs:
            prior_outputs_section += "\nYou can reference outputs from tasks in PREVIOUS phases using their globally unique IDs in the format '$result.GLOBAL_TASK_ID.output_field'."
            prior_outputs_section += "\nThe GLOBAL_TASK_ID is typically structured as 'PHASEID_ORIGINALTASKIDINPHASE'."
            prior_outputs_section += "\nAvailable outputs are listed below with their global IDs:\n"
            found_outputs = False
            for global_task_id, output_value in available_prior_outputs.items():
                found_outputs = True
                output_summary = self._summarize_output_for_prompt(output_value)
                prior_outputs_section += f"  - Global Task ID: '{global_task_id}', Output Summary: {output_summary}\n"
            if not found_outputs:
                prior_outputs_section += "No outputs from previous phases are available or they were empty.\n"
        else:
            prior_outputs_section += "No previous phases have produced outputs, or this is the first phase.\n"
        prior_outputs_section += "--- END OF AVAILABLE PREVIOUS PHASE OUTPUTS ---\n"

        user_pt = PromptTemplate.from_template(PHASE_SPECIFIC_USER_INSTRUCTIONS_TEMPLATE)
        return user_pt.format(
            original_query=original_query,
            phase_id=phase_id_str,
            phase_description=phase_description,
            tool_schemas_str=formatted_tools_section,
            available_prior_outputs_section=prior_outputs_section,
            conversation_history=conversation_context_str,
            few_shot_examples=few_shot_examples_str,
            CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS=CRITICAL_JSON_OUTPUT_FORMAT_INSTRUCTIONS,
            OUTPUT_FORMAT_INSTRUCTIONS=OUTPUT_FORMAT_INSTRUCTIONS
        )

    def _format_tasks_for_prompt(self, tasks: List[Dict[str, Any]]) -> str:
        """Helper to format lists of tasks (TaskNode-like dicts) for prompts, including status and attempts."""
        if not tasks:
            return "  None"
        
        formatted_list = []
        for task in tasks:
            task_id = task.get('id', 'Unknown ID')
            tool_name = task.get('tool_name', task.get('tool', 'Unknown Tool')) # Accommodate different key names
            status = task.get('status', 'N/A')
            attempts = task.get('execution_attempts', 'N/A')
            
            error_info = task.get('error')
            
            details_parts = [f"Status: {status}", f"Attempts: {attempts}"]
            if error_info:
                details_parts.append(f"Error: {str(error_info)[:100]}{'...' if len(str(error_info)) > 100 else ''}")
            
            details = ", ".join(details_parts)
                
            formatted_list.append(f"  - Task ID: {task_id}, Tool: {tool_name}, ({details})")
            
        return "\n".join(formatted_list)

# Example of use (outside class, for illustration):
# builder = PlannerPromptBuilder()
# schemas = [...] # Your ServerToolSchemaGroup list
# std_system_prompt, std_user_prompt = builder.build_planning_prompt("simple query", schemas)
# replan_system_prompt, replan_user_prompt = builder.build_planning_prompt("replan query", schemas, replan_context={...})
# phase_system_prompt, phase_user_prompt = builder.build_planning_prompt("complex query", schemas, query_type="COMPLEX", current_phase=Phase(...), hierarchical_plan=HierarchicalPlan(...), available_prior_outputs={...})