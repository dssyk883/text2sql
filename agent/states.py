from typing import List
from enum import Enum

class AgentState(Enum):
    """
    States for the agent state machine.
    Each state represents a specific phase in the NL2SQL pipeline.
    """
    PLANNING = "planning"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    TERMINAL = "terminal"

class TerminalStatus(Enum):
    """Terminal state conditions"""
    SUCCESS = "success"
    FAILURE_MAX_ITERATION = "failure_max_iteration"
    FAILURE_UNRECOVERABLE = "failure_unrecoverable"
    FAILURE_AMBIGUOUS_QUESTION = "failure_ambiguous_question"
    PARTIAL_SUCCESS = "partial_success"


class ActionType(Enum):
    """Available actions the agent can take"""
    # Planning actions
    FEW_SHOT_SELECT = "few_shot_select" # Select few-shot strategy and k, and generate examples

    # SQL Generation actions
    GENERATE_SQL = "generate_sql" # Generate new SQL
    REFINE_SQL = "refine_sql" # Refine existing SQL
    VALIDATE_SQL = "validate_sql" # Validate SQL syntax

    # SQL Execution actions
    EXECUTE_SQL = "execute_sql" # Execute SQL
    CHECK_SEMANTIC = "check_semantic" # Check semantic correctness of result with question & result


class Checkpoint(Enum):
    """
    Progress checkpoints for tracking agent's advancement.
    These are achieved sequentially
    """
    NONE = "none"
    SQL_GENERATED = "sql_generated"
    SQL_VALIDATED = "sql_validated"
    SQL_EXECUTED = "sql_executed"
    SEMANTIC_VERIFIED = "semantic_verified"

class SemanticCheckResult(Enum):
    """Result of semantic checking"""
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"

class ErrorType(Enum):
    """ SQL error classification"""
    SYNTAX_ERROR = "syntax"
    SCHEMA_ERROR = "schema"
    LOGIC_ERROR = "logic"
    PERMISSION_ERROR = "permission"
    TIMEOUT_ERROR = "timeout"
    UNKNOWN_ERROR = "unknown"

STATE_ACTIONS = {
    AgentState.PLANNING: [
        ActionType.FEW_SHOT_SELECT,
    ],
    AgentState.SQL_GENERATION: [
        ActionType.GENERATE_SQL,
        ActionType.VALIDATE_SQL,
        ActionType.FEW_SHOT_SELECT, # Only on FAIL from execution

    ],
    AgentState.SQL_EXECUTION: [
        ActionType.EXECUTE_SQL,
        ActionType.CHECK_SEMANTIC,
    ],

    AgentState.TERMINAL: [] # No actions allowed
}


def classify_error(error_msg: str) -> ErrorType:
    error_lower = error_msg.lower()
    
    syntax_keywords = ["syntax error", "near", "unexpected", "invalid syntax"]
    if any(kw in error_lower for kw in syntax_keywords):
        return ErrorType.SYNTAX_ERROR
    
    # Most common errors
    schema_keywords = ["no such table", "no such column", "unknown column",
                       "table doesn't exist", "column not found"]
    if any(kw in error_lower for kw in schema_keywords):
        return ErrorType.SCHEMA_ERROR
    
    # Second common errors
    logic_keywords = ["ambiguous", "subquery", "aggregate", "group by"]
    if any(kw in error_lower for kw in logic_keywords):
        return ErrorType.LOGIC_ERROR

    permission_keywords = ["access denied", "permission", "unauthorized"]
    if any(kw in error_lower for kw in permission_keywords):
        return ErrorType.PERMISSION_ERROR
    
    # Timeout errors
    timeout_keywords = ["timeout", "time limit", "exceeded"]
    if any(kw in error_lower for kw in timeout_keywords):
        return ErrorType.TIMEOUT_ERROR
    
    return ErrorType.UNKNOWN_ERROR

def get_available_actions(
        state: AgentState,
        has_sql: bool = False,
        semantic_result: SemanticCheckResult = None,
        low_confidence: bool = False,
        failure_history: List[ErrorType] = None,
        ) -> List[ActionType]:
    """
    Get available actions based on current state and context.

    Args:
        state: Current agent state
        has_sql: If SQL has been generated
        has_validation: if SQL has been validated
        semantic_result: Result from semantic check
        low_confidence: if LLM confidence is low
        failure_history: List of previous failures
    Returns:
        List of available actions
    """
    base_actions = STATE_ACTIONS.get(state, [])

    if state == AgentState.PLANNING:
        return base_actions
    
    elif state == AgentState.SQL_GENERATION:
        available = []

        if semantic_result == SemanticCheckResult.FAIL:
            available.append(ActionType.FEW_SHOT_SELECT)
        
        if not has_sql:
            available.append(ActionType.GENERATE_SQL)
        else:
            if low_confidence or (failure_history and len(failure_history) > 0):
                # Allow regeneration if confidence low and has failure
                available.append(ActionType.GENERATE_SQL)
                available.append(ActionType.REFINE_SQL)
            
            available.append(ActionType.VALIDATE_SQL)

        return available

    return base_actions

def get_next_state(
    current_state: AgentState,
    last_action: ActionType,
    action_success: bool,
    semantic_result: SemanticCheckResult = None,
) -> AgentState:
    """
    Determine next state based on current state and action result.

    Args:
        current_state: Current state
        last_action: Last action performed
        action_success: Whether the last action succeeded
        semantic_result: REsult from semantic check if any
        error_type: Type of error if action failed
    
    Returns:
        Next state to transition to
    """
    if current_state == AgentState.PLANNING:
        return AgentState.SQL_GENERATION
    
    elif current_state == AgentState.SQL_GENERATION:
        if last_action == ActionType.VALIDATE_SQL and action_success:
            return AgentState.SQL_EXECUTION
        return AgentState.SQL_GENERATION
    
    elif current_state == AgentState.SQL_EXECUTION:
        if last_action == ActionType.EXECUTE_SQL:
            if not action_success:
                # Execution failed -> go back to generation
                return AgentState.SQL_GENERATION
            return AgentState.SQL_EXECUTION
        
        elif last_action == ActionType.CHECK_SEMANTIC:
            if semantic_result == SemanticCheckResult.PASS:
                return AgentState.TERMINAL
            elif semantic_result in [SemanticCheckResult.FAIL, SemanticCheckResult.PARTIAL]:
                return AgentState.SQL_GENERATION
            return AgentState.SQL_EXECUTION # shouldn't fall in this line
        
    return current_state # By default stay in the current state        