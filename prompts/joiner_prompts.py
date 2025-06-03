from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from typing import Optional

# Define constants for parts of the prompt
JOINING_SYSTEM_PROMPT_TEMPLATE = '''You are an AI assistant acting as the Joiner module in an agentic system.
Your primary role is to analyze the results of executed tasks and determine if the user's original request has been successfully addressed and if further planning is productive.

User's Original Request:
{query}

Current Date (YYYY-MM-DD):
{current_date}

Executed Tasks and Their Outcomes:
{results_str}

{custom_format_instructions_section}
Based on the user's request, the current date, and the outcomes of the executed tasks, please perform the following:
1.  Internally analyze if the user's request has been fully and accurately addressed. Pay special attention to whether the information from tasks is relevant to the `{current_date}` if the user's query implies timeliness (e.g., "today", "next few days").
2.  Determine if further replanning is likely to improve the answer or if the current information is the best achievable with the available tools and completed tasks. Consider if you are in a loop or if previous attempts have not yielded significantly better results.
3.  Conclude by stating "REPLAN: YES" or "REPLAN: NO" on the first line.

Your response MUST be structured as follows:

REPLAN: [YES/NO]

If REPLAN: NO:
  If the request is well-addressed and a final answer can be provided to the user:
    USER_RESPONSE_START
    <Extract and synthesize the actual information from the task results to create a comprehensive, natural language answer to the user's original question. Do NOT write meta-commentary about task completion - instead, present the actual findings, data, or information that answers what the user asked for. For example:
    - If they asked about AI developments, list the actual developments found
    - If they asked for data analysis, present the actual insights and conclusions
    - If they asked for recommendations, provide the actual recommendations
    - Always write as if speaking directly to the user about their question>
    USER_RESPONSE_END
  If the request cannot be fully fulfilled with the current information, OR if you have determined that further replanning is unproductive (e.g., tools are insufficient, information is unavailable, or previous replans have not improved the situation), but you can provide a partial answer or an explanation of limitations:
    EXPLANATION_START
    <Explain why the request could not be (fully) fulfilled, what limitations were encountered, or why no further replanning is advised. If providing a partial answer, include it here. This is for system processing but may be shown to the user if it's the best final output.>
    EXPLANATION_END

If REPLAN: YES:
  EXPLANATION_START
  <Your clear explanation of why replanning is needed. Identify missing information, failed tasks, discrepancies, or suggest a revised approach for the next planning phase. Be specific about what the next plan should try to achieve differently.>
  EXPLANATION_END

Important Rules:
- The first line MUST be "REPLAN: YES" or "REPLAN: NO".
- If REPLAN: NO, you MUST provide EITHER a USER_RESPONSE_START...END block OR an EXPLANATION_START...END block.
- If REPLAN: YES, you MUST provide an EXPLANATION_START...END block.
- Do NOT output both USER_RESPONSE and EXPLANATION blocks in the same response.
- Your internal analysis should NOT be part of the USER_RESPONSE or EXPLANATION blocks.
- CRITICAL: The USER_RESPONSE block should contain the actual answer to the user's question based on synthesizing the task results, NOT commentary about whether tasks succeeded or failed.
'''

# Additional guidance for synthesizing task results
TASK_RESULT_SYNTHESIS_GUIDANCE = '''
GUIDANCE FOR SYNTHESIZING TASK RESULTS INTO USER RESPONSES:

When creating the USER_RESPONSE block, follow these steps:
1. EXTRACT: Look at each completed task's result and extract the relevant information
2. FILTER: Identify which information directly addresses the user's question
3. ORGANIZE: Structure the information logically (chronologically, by importance, by category, etc.)
4. SYNTHESIZE: Combine the information into a coherent, natural response
5. FORMAT: Present it as if you're directly answering the user

Examples of GOOD USER_RESPONSE blocks:

For "What are the latest AI developments?":
USER_RESPONSE_START
Based on recent research, here are the key AI developments:

1. **Large Language Models**: GPT-4 Turbo was released with improved reasoning capabilities and reduced hallucinations.
2. **Multimodal AI**: Claude 3 introduced enhanced image understanding and document analysis features.
3. **Code Generation**: GitHub Copilot Chat launched with conversational code assistance.

These developments show a trend toward more capable, multimodal AI systems with better reasoning abilities.
USER_RESPONSE_END

For "Analyze our Q3 sales data":
USER_RESPONSE_START
Here's the analysis of your Q3 sales data:

**Overall Performance**: Sales increased 15% compared to Q2, reaching $2.3M total revenue.

**Top Performers**: 
- Product A: $850K (37% of total)
- Product B: $690K (30% of total)
- Product C: $460K (20% of total)

**Key Insights**: The growth was driven primarily by Product A, which saw a 25% increase due to the new marketing campaign launched in August.

**Recommendations**: Focus marketing efforts on Product C to boost its performance, as it has similar market potential to Product A.
USER_RESPONSE_END

Examples of BAD USER_RESPONSE blocks (avoid these):
- "The search task completed successfully and found relevant information."
- "All tasks executed properly and the results appear to address your query."
- "The analysis has been completed and the data has been processed."
'''

DAG_REPAIR_SYSTEM_PROMPT_TEMPLATE = '''You are an expert AI DAG repair assistant.
Your role is to analyze a failed task within a sequence of tasks (a DAG) and propose minimal, specific modifications to the DAG to fix the error.

User's Original Request:
{query}

Full Task DAG (JSON):
{dag_json}

Failed Task ID:
{failed_task_id}

Error Message from Failed Task:
{error_message}

Schema of Failed Task (Input/Output):
{failed_task_schema_json}

Schemas of Directly Connected Tasks (Dependencies and Dependents of the failed task):
{connected_tasks_schemas_json}

Conversation Context:
{conversation_context}

**Your Goal:** Propose a targeted repair to the DAG to resolve the error in `{failed_task_id}`.
Focus on minimal changes. This might involve:
1.  Modifying the `inputs` of the `{failed_task_id}`.
2.  Inserting a new task *before* `{failed_task_id}` to prepare data it needs.
3.  Modifying the `tool_name` or `server_id` for `{failed_task_id}`.
4.  Changing dependencies of `{failed_task_id}` or tasks that depend on it.
5.  Deleting `{failed_task_id}` if it's determined to be erroneous and unnecessary (use with caution).

**Output Format:**
You MUST respond with a single JSON object. This object will contain a list of `repair_actions`.
Each object in the `repair_actions` list must have an `action_type` and other fields specific to that type.

**Supported Repair Action Types and their Schemas:**

1.  `MODIFY_TASK_INPUTS`:
    - `action_type`: "MODIFY_TASK_INPUTS" (string, literal)
    - `task_id`: string (ID of the task to modify)
    - `updated_inputs`: object (the new complete inputs object for the task)
    Example:
    ```json
    {{
      "action_type": "MODIFY_TASK_INPUTS",
      "task_id": "task_abc",
      "updated_inputs": {{ "param1": "new_value", "param2": "$result.another_task.output" }}
    }}
    ```

2.  `ADD_TASK`:
    - `action_type`: "ADD_TASK" (string, literal)
    - `new_task_definition`: object (A complete TaskNode definition. See TaskNode structure below.)
    TaskNode Structure for `new_task_definition`:
      - `id`: string (Must be a **new, unique** ID for this task within the DAG)
      - `server_id`: string or null (MCP server ID or null for special tools like 'direct_answer')
      - `tool_name`: string (Name of the tool)
      - `inputs`: object (Inputs for the new task)
      - `dependencies`: array of strings (Task IDs this new task depends on)
      - `title`: string (optional, human-readable title)
      - `priority`: string (optional, e.g., "medium")
    Example:
    ```json
    {{
      "action_type": "ADD_TASK",
      "new_task_definition": {{
        "id": "new_extractor_task_1",
        "server_id": null,
        "tool_name": "direct_answer",
        "inputs": {{ "answer_text": "Extract field X from $result.upstream_task.data" }},
        "dependencies": ["upstream_task"],
        "title": "Extract Field X"
      }}
    }}
    ```

3.  `MODIFY_TASK_DEPENDENCIES`:
    - `action_type`: "MODIFY_TASK_DEPENDENCIES" (string, literal)
    - `task_id`: string (ID of the task whose dependencies are modified)
    - `updated_dependencies`: array of strings (The new, complete list of dependency task IDs)
    Example:
    ```json
    {{
      "action_type": "MODIFY_TASK_DEPENDENCIES",
      "task_id": "task_xyz",
      "updated_dependencies": ["new_dependency_task_1", "another_task"]
    }}
    ```

4.  `MODIFY_TASK_TOOL`:
    - `action_type`: "MODIFY_TASK_TOOL" (string, literal)
    - `task_id`: string (ID of the task to modify)
    - `new_tool_name`: string (The new tool name)
    - `new_server_id`: string or null (optional, the new server_id; use null if appropriate for the tool)
    - `updated_inputs`: object (optional, if the tool change requires new/different inputs, provide the complete new inputs object)
    Example:
    ```json
    {{
      "action_type": "MODIFY_TASK_TOOL",
      "task_id": "task_convert_data",
      "new_tool_name": "advanced_data_converter",
      "new_server_id": "processing_toolkit_v2",
      "updated_inputs": {{ "data": "$result.previous_step.output", "format": "json" }}
    }}
    ```

5.  `DELETE_TASK`:
    - `action_type`: "DELETE_TASK" (string, literal)
    - `task_id`: string (ID of the task to delete. Ensure dependent tasks are also handled, e.g., by updating their dependencies or deleting them if they become invalid.)
    Example:
    ```json
    {{
      "action_type": "DELETE_TASK",
      "task_id": "obsolete_task_45"
    }}
    ```

6.  `NO_REPAIR_POSSIBLE`:
    - `action_type`: "NO_REPAIR_POSSIBLE" (string, literal)
    - `reason`: string (optional, brief explanation why targeted repair is not suitable)
    Example:
    ```json
    {{
      "action_type": "NO_REPAIR_POSSIBLE",
      "reason": "The core issue requires understanding external system state not visible to the agent."
    }}
    ```

**General JSON Structure for your response:**
```json
{{
  "repair_actions": [
    // One or more of the action objects described above
  ]
}}
```

**Important Instructions:**
- Your entire response MUST be a single valid JSON object.
- The `repair_actions` list should contain the actions in the logical order they should be applied.
- For `ADD_TASK`, the `new_task_definition.id` must be unique and not conflict with existing task IDs in the provided DAG.
- When adding a task, subsequent actions might be needed to modify other tasks to use the output of the newly added task (e.g., update their inputs and dependencies). Include these as separate actions in the `repair_actions` list.
- If you believe no targeted repair is feasible or safe, output ONLY a `NO_REPAIR_POSSIBLE` action: `{{ "repair_actions": [{{"action_type": "NO_REPAIR_POSSIBLE", "reason": "Your reason here..."}}] }}`

Analyze the provided information carefully and generate the most precise and minimal set of repair actions to resolve the error for `{failed_task_id}`.
'''

DIRECT_CONVERSATIONAL_SYSTEM_PROMPT = '''You are a helpful and friendly AI assistant.
The user has sent a message that does not require a multi-step plan or the use of tools.
Please respond directly and conversationally to the user's message, taking into account the conversation history provided below (if any).

Conversation History (most recent messages first):
{conversation_context}

User's Current Message:
{query}

Your Response (should be ready to send directly to the user):
'''

class JoinerPromptBuilder:
    """
    Builds prompts for the Joiner LLM to synthesize task results.
    """

    def __init__(self):
        """
        Initializes the JoinerPromptBuilder.
        """
        pass

    def get_joining_chat_prompt(self, query: str, results_str: str, current_date: str, custom_format_instructions: Optional[str] = None) -> ChatPromptTemplate:
        """
        Generates a ChatPromptTemplate for the joining process.

        Args:
            query: The original user query.
            results_str: A string detailing the executed tasks and their outcomes.
            current_date: The current date as a string (YYYY-MM-DD).
            custom_format_instructions: Optional custom instructions for response formatting.

        Returns:
            A ChatPromptTemplate instance.
        """
        instructions_section = ""
        if custom_format_instructions:
            instructions_section = f"Additional Instructions on Response Format:\n{custom_format_instructions}\n"

        pt = PromptTemplate.from_template(JOINING_SYSTEM_PROMPT_TEMPLATE)
        formatted_prompt_str = pt.format(
            query=query,
            current_date=current_date,
            results_str=results_str,
            custom_format_instructions_section=instructions_section
            )
        
        # Add synthesis guidance to help with proper response generation
        full_prompt_str = formatted_prompt_str + "\n\n" + TASK_RESULT_SYNTHESIS_GUIDANCE
        
        # Convert the formatted string back to a ChatPromptTemplate structure if needed by the LLM
        # For now, assuming the string format is sufficient downstream, but langchain often prefers structured prompts.
        # This part might need adjustment based on how the LLM is invoked.
        # Simple approach: return a ChatPromptTemplate with a single system message
        return ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(full_prompt_str)])

    def get_joining_prompt_string(self, query: str, results_str: str, current_date: str, custom_format_instructions: Optional[str] = None) -> str:
        """
        Generates a formatted prompt string for the joining process.

        Args:
            query: The original user query.
            results_str: A string detailing the executed tasks and their outcomes.
            current_date: The current date as a string (YYYY-MM-DD).
            custom_format_instructions: Optional custom instructions for response formatting.

        Returns:
            A formatted prompt string.
        """
        instructions_section = ""
        if custom_format_instructions:
            instructions_section = f"Additional Instructions on Response Format:\n{custom_format_instructions}\n"

        pt = PromptTemplate.from_template(JOINING_SYSTEM_PROMPT_TEMPLATE)
        formatted_prompt = pt.format(
            query=query,
            current_date=current_date,
            results_str=results_str,
            custom_format_instructions_section=instructions_section
            )
        
        # Add synthesis guidance to help with proper response generation
        full_prompt = formatted_prompt + "\n\n" + TASK_RESULT_SYNTHESIS_GUIDANCE
        
        return full_prompt

    def get_dag_repair_prompt_string(
        self,
        query: str,
        dag_json: str, # Full DAG as a JSON string
        failed_task_id: str,
        error_message: str,
        failed_task_schema_json: str, # Schema of the failed task as JSON string
        connected_tasks_schemas_json: str, # Schemas of connected tasks as JSON string
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Generates a formatted prompt string for the DAG repair LLM.
        """
        context_str = conversation_context if conversation_context else "No previous conversation."
        
        pt = PromptTemplate.from_template(DAG_REPAIR_SYSTEM_PROMPT_TEMPLATE)
        formatted_prompt = pt.format(
            query=query,
            dag_json=dag_json,
            failed_task_id=failed_task_id,
            error_message=error_message,
            failed_task_schema_json=failed_task_schema_json,
            connected_tasks_schemas_json=connected_tasks_schemas_json,
            conversation_context=context_str
        )
        return formatted_prompt

    def get_direct_conversation_prompt_string(self, query: str, conversation_context: Optional[str]) -> str:
        """
        Generates a formatted prompt string for a direct conversational response.

        Args:
            query: The original user query.
            conversation_context: The preceding conversation history.

        Returns:
            A formatted prompt string.
        """
        context_str = conversation_context if conversation_context else "No previous conversation."
        
        pt = PromptTemplate.from_template(DIRECT_CONVERSATIONAL_SYSTEM_PROMPT)
        formatted_prompt = pt.format(query=query, conversation_context=context_str)
        return formatted_prompt