from enum import Enum

class AgentState(Enum):
    """
    States for the agent state machine.
    Each state represents a specific phase in query generation and refinement
    """
    ANALYZE = "analyze"
    GATHER_INFO = "gather_info"
    GENERATE_SQL = "generate_sql"
    VALIDATE_SQL = "validate_sql"
    EXECUTE_SQL = "execute_sql"
    REFINE_SQL = "refine_sql"
    DONE = "done"
    FAIL = "fail"


class ActionType(Enum):
    """Available actions the agent can take"""
    GET_DB_SCHEMA = "get_db_schema"
    SEARCH_EXAMPLES = "search_similar_examples"
    GENERATE_SQL = "generate_sql"
    VALIDATE_SYNTAX = "validate_syntax"
    EXECUTE_QUERY = "execute_query"
    REFINE_QUERY = "refine_query"
    FINISH = "finish"
    ABORT = "abort"

STATE_ACTIONS = {
        AgentState.ANALYZE: [
        ActionType.GET_DB_SCHEMA,
        ActionType.SEARCH_EXAMPLES,
        ActionType.GENERATE_SQL
    ],
    AgentState.GATHER_INFO: [
        ActionType.GET_DB_SCHEMA,
        ActionType.SEARCH_EXAMPLES,
        ActionType.GENERATE_SQL
    ],
    AgentState.GENERATE_SQL: [
        ActionType.GENERATE_SQL
    ],
    AgentState.VALIDATE_SQL: [
        ActionType.VALIDATE_SYNTAX,
        ActionType.EXECUTE_QUERY
    ],
    AgentState.EXECUTE_SQL: [
        ActionType.EXECUTE_QUERY
    ],
    AgentState.REFINE_SQL: [
        ActionType.REFINE_QUERY
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