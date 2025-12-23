"""
Prompt Builder
Generates state specific prompts for the LLM
"""

from typing import Dict, Optional, List
from agent2.states import AgentState, ActionType, get_available_workers
from agent2.memory import AgentMemory, SQLAttempt
from langchain_core.prompts.prompt import PromptTemplate

class PromptBuilder:
    EXAMPLE_OUTPUT = """
{"worker": "get_db_schema", "params": {}, "reasoning": "Need schema first", "confidence": 0.95}
{"worker": "few_shot_select", "params": {"strategy": "jaccard", "k": 5}, "reasoning": "Similar questions found", "confidence": 0.8}
{"worker": "generate_sql", "params": {}, "reasoning": "Schema loaded, ready to generate", "confidence": 0.9}
    """
    ACTION_DESCRIPTIONS = {
            ActionType.VALIDATE_SQL: "Validate SQL syntax",
            ActionType.GENERATE_SQL: "Generate SQL query",
            ActionType.CHECK_SEMANTIC: "Compare the question and SQL query",
            ActionType.EXECUTE_SQL: "Execute the generated SQL query",
            ActionType.FEW_SHOT_SELECT: "Select a few-shot example strategy",
        }
    
    def build_prompt(
        self,
        state: AgentState,
        memory: AgentMemory
    ) -> str:
        if state == AgentState.GET_DB_SCHEMA:
            return self._build_get_db_prompt(state, memory)
        elif state == AgentState.VALIDATE_SQL:
            return self._build_validate_sql_prompt(state, memory)
        elif state == AgentState.GENERATE_SQL:
            return self._build_generate_sql_prompt(state, memory)
        elif state == AgentState.CHECK_SEMANTIC:
            return self._build_check_semantic_prompt(state, memory)
        elif state == ActionType.FEW_SHOT_SELECT:
            return self._build_few_shot_prompt(state, memory)
        else:
            return ""

    def build_decision_prompt(self, state: AgentState, memory:AgentMemory) -> str:
        sql = memory.get_last_sql()
        prev_ex_result = memory.get_last_execution_result()
        schema_summary = memory.schema_summary
        last_action = self._format_action(memory.get_last_action())
        available_workers = self._format_workers(get_available_workers(state))

        return f"""You are an Agent Controller for an NL2SQL system.
Your job is to decide the NEXT worker to call.

[Question]
{memory.question}
[Current SQL]
{sql if sql else "No SQL yet"}

[Execution Result]
{prev_ex_result if prev_ex_result else "No execution result yet"}

[DB Schema Summary]
{schema_summary if schema_summary else "No DB schema loaded yet"}

[Previous Actions]
{last_action if last_action else "No action done yet"}

[Instructions]
- Choose exactly ONE next worker.
- Available workers:
{available_workers}

- If you choose Few-shot Selector, specify:
- strategy: Random | Intent cluster | Jaccard
- k: number of examples

- Respond ONLY in valid JSON.
- Do NOT include explanations outside JSON.

[JSON Output Format]
{{
"worker": "<worker_name>",
"params": {{}},
"reasoning": "...",
"confidence": 0.0-1.0
}}

[Example Outputs]
{self.EXAMPLE_OUTPUT}
    """

    def _format_workers(self, actions: List[ActionType]) -> str:
        formatted = []
        for i, action in enumerate(actions):
            description = self.ACTION_DESCRIPTIONS.get(action, "No description")
            formatted.append(f"{i}. {action.value}: {description}")
        
        return "\n".join(formatted)

    def _format_examples(self, examples: List[Dict[str, str]] = None) -> str:
        if not examples:
            return "No examples"
        formatted = []
        for ex in examples:
            formatted.append(f"Question: {ex['input']}\nSQL:{ex['query']}")
        return '\n'.join(formatted)

    
    def build_generate_sql_prompt(self, memory: AgentMemory) -> str:
        examples = self._format_examples(memory.examples)
        return f"""You are a SQLite expert. Learn these natural languages to SQL examples.
Examples:
{examples}

Now, given the following information, generate the correct SQL query:
Schema:
{memory.schema_summary}

Critical Rules:
1. If a table/column is not in the schema above, you CANNOT use it
2. Check spelling carefully (case-sensitive)
3. Do NOT use common sense - use ONLY what's in the schema
4. Return ONLY the SQL query

Question: {memory.question}
"""
    
    def build_semantic_check_prompt(self, memory:AgentMemory) -> str:
        sql = memory.get_last_sql()
        execution_result = memory.get_last_execution_result()
        schema = memory.schema_summary


    def _format_attempts(self, attempts: Optional[List[SQLAttempt]] = None) -> str:
        """
        Format SQL Attempts for LLM context
        """
        if not attempts:
            return "No previous attempts."

        formatted = []
        for idx, attempt in enumerate(attempts, 1):
            status = "Success" if attempt.success else "Failed"
            parts = [f"Attempt #{idx} [{status}]", f"SQL: {attempt.sql}"]
            if attempt.error:
                error_message = attempt.error.get('error_message', 'Unknown error message')
                error_type = attempt.error.get('error_type', 'Unknown error type')
                parts.append(f"Error type: {error_type}, Error message: {error_message}")
            
            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _format_action(self, actions: Optional[List[Dict[str, str]]]) -> str:
        """
        Format Action history for LLM context
        """
        if not actions:
            return "No previous actions"
        
        formatted = []
        for idx, action in enumerate(actions, 1):
            parts = [f"Action #{idx}", f"Action: {action['action'].val}",
                     f"State: {action['state'].val}", f"Result: {action['result']}"]
            formatted.append("\n".join(parts))
        
        return "\n\n".join(formatted)