"""
Prompt Builder
Generates state specific prompts for the LLM
"""

from typing import Dict, Any, List
from agent.states import AgentState, ActionType
from agent.memory import AgentMemory

class PromptBuilder:
    ACTION_DESCRIPTIONS = {
            ActionType.GET_DB_SCHEMA: "Get database tables and columns",
            ActionType.SEARCH_EXAMPLES: "Find similar query examples",
            ActionType.GENERATE_SQL: "Generate SQL query",
            ActionType.VALIDATE_SYNTAX: "Validate SQL syntax",
            ActionType.EXECUTE_QUERY: "Execute the query",
            ActionType.REFINE_QUERY: "Fix errors in the query",
            ActionType.FINISH: "Task complete",
            ActionType.ABORT: "Cannot complete task"
        }

    def __init__(self, config):
        self.config = config

    def build_prompt(
        self,
        state: AgentState,
        memory: AgentMemory
    ) -> str:
        if state == AgentState.ANALYZE:
            return self._build_analyze_prompt(memory)
        elif state == AgentState.GATHER_INFO:
            return self._build_gather_info_prompt(memory)
        elif state == AgentState.GENERATE_SQL:
            return self._build_generate_sql_prompt(memory)
        # elif state == AgentState.VALIDATE_SQL:
        #     return self._build_validate_sql_prompt(memory)
        elif state == AgentState.REFINE_SQL:
            return self._build_refine_sql_prompt(memory)
        else:
            return ""

    def _build_analyze_prompt(self, memory: AgentMemory) -> str:
        """Initial quesiton analysis"""

        prompt = f"""You are an NL2SQL agent. Analyze this question and decided what to do next.

User Question: {memory.question}

Current Status:
- Schema loaded?: {memory.has_schema()}
- Examples loaded?: {memory.has_examples()}

Choose ONE Action:
1. get_db_schema - Get database tables and columns
2. serach_similar_examples - Find similar query examples
3. generate_sql - Generate SQL query now

Respond with ONLY the action name (e.g., "get_db_schema")
"""
        return prompt.strip()
    
    def _build_gather_info_prompt(self, memory: AgentMemory) -> str:
        """Gathering additional info"""

        status = []

        status.append("Schema loaded" if memory.schema else "Schema needed")
        status.append("Examples loaded" if memory.examples else "Examples needed")

        prompt = f"""You are preparing to generate SQL for this question.

Question: {memory.question}

Status: 
{chr(10).join(status)}

What do you need next?
1. get_db_schema - Get database structure
2. search_similar_examples - Get similar queries
3. generate_sql - Ready to generate SQL 

Respond with ONLY the action name (e.g., "get_db_schema")
"""
        return prompt.strip()
    
    def _build_generate_sql_prompt(self, memory: AgentMemory) -> str:
        """Prompt for SQL generation"""

        summary_schema = memory.schema_summary if memory.has_schema() else ""

        formatted_ex = ""

        for i, ex in enumerate(memory.examples, 1):
            formatted_ex += f"Example {i}:\n"
            formatted_ex += f"Question: {ex['input']}\n"
            formatted_ex += f"SQL: {ex['query']}\n"

        prompt = f"""You are a SQLite expert. Learn these natural languages to SQL examples.
Examples:
{formatted_ex}
Now, given the following information, generate the correct SQL query:
Schema: {summary_schema}

Critical Rules:
1. If a table/column is not in the schema above, you CANNOT use it
2. Check spelling carefully (case-sensitive)
3. Do NOT use common sense - use ONLY what's in the schema
4. Return ONLY the SQL query

Question: {memory.question}
"""
        return prompt.strip()
    
#     def _build_validate_sql_prompt(self, memory: AgentMemory) -> str:
#         """Prompt for SQL validation - not sure if it's used"""
        
#         last_sql = memory.get_last_sql()
        
#         prompt = f"""Review this SQL query for errors.

# Question: {memory.question}
# Generated SQL: {last_sql}

# Check for:
# - Syntax errors
# - Wrong table/column names
# - Logic errors

# Is this SQL correct? Reply with:
# - "valid" if correct
# - "invalid: [reason]" if there are issues
# """
        
#         return prompt.strip()
    
    def _build_refine_sql_prompt(self, memory: AgentMemory) -> str:
        """Prompt for SQL refinement after error"""

        last_attempt = memory.sql_attempts[-1] if memory.sql_attempts else None
        error = memory.last_error

        prompt_parts = [
            "The previous SQL query failed. Fix the error and generate a corrected query.",
            "",
            f"Question: {memory.question}",
            "",
            f"Failed SQL:\n{last_attempt.sql if last_attempt else 'N/A'}",
            "",
            f"Error Type: {error.get('error_type', 'unknown') if error else 'unknown'}",
            f"Error Message: {error.get('error', 'Unknown error') if error else 'Unknown error'}",
        ]
        
        # Add schema for reference
        if memory.schema_summary:
            prompt_parts.append("\nCorrect Schema:")
            prompt_parts.append(memory.schema_summary)
        
        # Add error specific hints
        if error:
            error_type = error.get('error_type', 'unknown')
            prompt_parts.append(self._get_error_hint(error_type))
        
        prompt_parts.extend([
            "",
            "Generate the corrected SQLite query:",
            "SQL:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_error_hint(self, error_type: str) -> str:
        """Get helpful hint based on error type"""
        hints = {
            "syntax": "\nHint: Check for missing keywords, commas, or parentheses.",
            "schema": "\nHint: Verify table and column names match the schema exactly.",
            "logic": "\nHint: Review aggregate functions and GROUP BY clauses.",
            "permission": "\nHint: This table may require different access.",
            "timeout": "\nHint: Simplify the query or add indexes."
        }
        return hints.get(error_type, "\nHint: Review the error message carefully.")
    
    def build_action_prompt(
        self, 
        state: AgentState, 
        allowed_actions: List[ActionType]
    ) -> str:
        """
        Prompt for action selection
        
        :param state: Current state
        :param allowed_actions: List of allowed actions
        :return: Formatted prompt for action selection
        """
        options = []
        for i, action in enumerate(allowed_actions, 1):
            desc = self.ACTION_DESCRIPTIONS.get(action, action.value)
            options.append(f"{i}. {action.value} - {desc}")
    
        prompt = f"""Current state: {state.value}

Choose ONE action:
{chr(10).join(options)}

Respond with the action name only:
"""
        
        return prompt.strip()