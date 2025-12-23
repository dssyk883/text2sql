from enum import Enum

class AgentState(Enum):
    """
    States for the agent state machine.
    Each state represents a specific phase in query generation and refinement
    """
    SQL_STATE = "sql_state"
    POST_EXEC_STATE = "post_exec_state"
    DONE = "done"
    FAIL = "fail"


class ActionType(Enum):
    """Available actions the agent can take"""
    GENERATE_SQL = "generate_sql"
    GET_DB_SCHEMA = "get_db_schema"
    VALIDATE_SQL = "validate_sql"
    EXECUTE_SQL = "execute_sql"
    CHECK_SEMANTIC = "check_semantic"
    FEW_SHOT_SELECT = "few_shot_select"
    # FAIL = "fail" # Need this?

STATE_ACTIONS = {
    AgentState.SQL_STATE = [
        ActionType.GET_DB_SCHEMA,
        ActionType.GENERATE_SQL,
        ActionType.FEW_SHOT_SELECT,
        ActionType.VALIDATE_SQL
    ],
    AgentState.POST_EXEC_STATE = [
        ActionType.EXECUTE_SQL,
        ActionType.CHECK_SEMANTIC
    ]
}


class ErrorType(Enum):
    """ SQL error classification"""
    SYNTAX_ERROR = "syntax"
    SCHEMA_ERROR = "schema"
    LOGIC_ERROR = "logic"
    PERMISSION_ERROR = "permission"
    TIMEOUT_ERROR = "timeout"
    UNKNOWN_ERROR = "unknown"

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