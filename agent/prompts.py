"""
Prompt Builder
Generates state specific prompts for the LLM
"""

from typing import Dict, Optional, List
from agent2.states import AgentState, ActionType, get_available_actions
from agent2.memory import AgentMemory, SQLAttempt
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

class PromptBuilder:

    ACTION_DESCRIPTIONS = {
            ActionType.FEW_SHOT_SELECT: "Select a few-shot example strategy",
            ActionType.GENERATE_SQL: "Generate SQL query",
            ActionType.VALIDATE_SQL: "Validate SQL syntax",
            ActionType.REFINE_SQL: "Refine SQL query",
            ActionType.EXECUTE_SQL: "Execute the generated SQL query",
            ActionType.CHECK_SEMANTIC: "Compare the question and SQL query",            
        }
    
    DECISION_INSTRUCTIONS = """
- Choose exactly ONE next worker
- For Few-Shot Selector, specify strategy (Random/Intent/Jaccard) and k(1-5)
- Respond ONLY in valid JSON
- Do NOT include explanations outside JSON
"""

    EXAMPLE_DECISION = """
{"worker": "few_shot_select", "params": {"strategy": "jaccard", "k": 5}, "reasoning": "Question has specific keywords, Jaccard will match terms", "confidence": 0.8},
{"worker": "generate_sql", "params": {}, "reasoning": "SQL Examples loaded, ready to generate","confidence": 0.9}
    """

    EXAMPLE_SEMANTIC = """
{"status": "PASS", "confidence": 0.95, "issues": [], "reasoning": "Correct count query"},
{"status": "PARTIAL", "confidence": 0.6, "issues": ["Missing inactive departments"], "reasoning": "Question says 'all' but SQL filters active only"},
{"status": "FAIL", "confidence": 0.9, "issues": ["No GROUP BY", "Missing department info"], "reasoning": "Gives overall average, not per department"}
"""

    def __init__(self):
        self._build_templates()

    def _build_templates(self):
        """Initialize prompt template"""
        self.templates = {
            "decision": self._create_decision_template(),
            "sql_generation": self._create_sql_generation_template(),
            "semantic_check": self._create_semantic_check_template(),
        }

    def _create_decision_template(self) -> PromptTemplate:
        template = """You are deciding the next action for SQL generation.
QUESTION: {question}
DB SCHEMA: {schema_summary}
FEW-SHOT EXAMPLES: {examples_status}
STATE: {current_state}
PROGRESS: {current_checkpoint}

CURRENT SQL: {current_sql}
LAST RESULT: {execution_result}
LAST ERROR: {last_error}
LAST ACTION: {last_action}

AVAILABLE ACTIONS:
{available_actions}

Choose ONE action and respond in JSON:
{{"action": "<name>", "params": {{}}, "reasoning": "<brief reason>", "confidence": 0.0-1.0}}

Response Examples:
{example_decision}

Your choice:"""

        return PromptTemplate(
            input_variables=[
                "question",
                "schema_summary",
                "examples_status",
                "current_state",
                "current_checkpoint", 
                "current_sql",
                "execution_result",
                "last_error",                
                "last_action",
                "available_actions",
                "example_decision"
            ],
            template=template
        )
    
    def _create_sql_generation_template(self) -> PromptTemplate:
        template ="""You are a SQLite expert. Learn these natural languages to SQL examples.
Examples:
{examples}

Now, given the following information, generate the correct SQL query:
Schema:
{schema_summary}

Critical Rules:
1. If a table/column is not in the schema above, you CANNOT use it
2. Check spelling carefully (case-sensitive)
3. Do NOT use common sense - use ONLY what's in the schema
4. Return ONLY the SQL query

Question: {question}
"""
        return PromptTemplate(
            input_variables=[
                "question",
                "schema_summary",
                "examples"
            ],
            template=template
        )
    
    def _create_semantic_check_template(self) -> PromptTemplate:
        template = """Check if SQL results answer the question correctly.
QUESTION: {question}
SQL: {sql}
RESULT: {execution_result}
SCHEMA: {schema_summary}

Analysis:
1. Does result structure match question?
2. Is row count reasonable?
3. Are values correct type?

Respond in JSON:
{{"status": "PASS|PARTIAL|FAIL", "issues": ["issue1", "issue2"], "reasoning": "brief explanation", "confidence": 0.0-1.0}}

Response Examples:
{example_semantic}

Your analysis:
"""

        return PromptTemplate(
            input_variables=[
                "question",
                "sql",
                "execution_result",
                "schema_summary",
                "example_semantic",
            ],
            template=template
        )

    def build_decision_prompt(self, state: AgentState, memory:AgentMemory) -> str:
        """Build the complete decision prompt"""
        sql = memory.get_last_sql()
        if not memory.examples or len(memory.examples) == 0:
            examples_status = "No examples loaded"
        else:
            examples_status = f"{len(memory.examples)} examples loaded"
        last_execution_result = memory.get_last_execution_result()
        last_error = memory.get_last_error()
        last_action = self._format_action(memory.get_last_action())
        available_actions = self._format_available_actions(get_available_actions(
            state=state,
            memory=memory,
        ))
        return self.templates['decision'].format(
            question=memory.question,
            schema_summary=memory.schema_summary,
            examples_status=examples_status,
            current_state=state.value,
            current_checkpoint=memory.checkpoint.value,
            current_sql=sql,
            execution_result=last_execution_result,
            last_error=last_error,            
            last_action=last_action,
            available_actions=available_actions,
            example_decision=self.EXAMPLE_DECISION
        )       
   
    def build_generate_sql_prompt(self, memory: AgentMemory) -> str:
        examples = self._format_examples(memory.examples)
        return self.templates['sql_generation'].format(
            question=memory.question,
            schema_summary=memory.schema_summary,
            examples=examples
        )
    
    def build_semantic_check_prompt(self, memory: AgentMemory) -> str:
        result = memory.get_last_execution_result()
        return self.templates['semantic_check'].format(
            question=memory.question,
            sql=memory.sql,
            execution_result=result,
            schema_summary=memory.schema_summary,
            example_semantic=self.EXAMPLE_SEMANTIC
        )
    
    def _format_available_actions(self, actions: List[ActionType]) -> str:
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

    def _format_action(self, action: Optional[Dict[str, str]]) -> str:
        """
        Format last action
        """
        if not action:
            return "No previous actions"
        return f"State: {action['state'].value}, Action: {action['action'].value} ({'Success' if action['success'] else 'Failed'}), Iteration: {action['iteration']}"