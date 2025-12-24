from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from agent2.states import AgentState, ActionType, Checkpoint

@dataclass
class SQLAttempt:
    sql: str
    timestamp: datetime
    success: bool
    error: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    confidence: Optional[float] = None


@dataclass
class AgentMemory:

    question: str

    # Database information
    schema: Optional[str] = None
    schema_summary: Optional[str] = None

    # Current checkpoint
    checkpoint: Checkpoint = Checkpoint.NONE
    
    # Few-shot examples
    examples: List[Dict[str, str]] = field(default_factory=list)
    few_shot_history: List[Dict[str, any]] = field(default_factory=list)
    """
    startegy,
    k,
    trigger # Why LLM chose this strategy
    """
    
    # SQL generation history
    sql_attempts: List[SQLAttempt] = field(default_factory=list)

    # Action history
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    # state: AgentState, action: ActionType, success: bool, iteration: int
    
    # Error tracking
    last_error: Optional[Dict[str, Any]] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    # error_message: error message, error_type: error type

    analysis: Optional[Dict[str, Any]] = None
    
    # Execution results
    successful_results: List[Dict[str, Any]] = field(default_factory=list)
    # SQLAttempt with success = True
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)

    def add_action(
            self,
            action: ActionType,
            state: AgentState,
            success: bool,
            iteration: int
    ):
        self.action_history.append({'action': action, 'state': state, 'success': success, 'iteration': iteration})

    def add_sql_attempt(
        self,
        sql: str,
        success: bool = False,
        error: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        attempt = SQLAttempt(
            sql=sql,
            timestamp=datetime.now(),
            success=success,
            error=error,
            result=result,
            confidence=confidence
        )

        self.sql_attempts.append(attempt)

        if error:
            self.last_error = error
            self.error_history.append(error)

        if success and result:
            self.successful_results.append({
                "sql": sql,
                "result": result,
                "timestamp": datetime.now()
            })
    
    def get_last_sql(self) -> Optional[str]: # the most recent SQL query
        if self.sql_attempts:
            return self.sql_attempts[-1].sql
        return None
       
    def get_last_action(self) -> Optional[Dict[str, Any]]: # the most recent action
        if self.action_history:
            return self.action_history[-1]
        return None
    
    def get_last_execution_result(self) -> Optional[str]:
        if self.sql_attempts:
            return self.sql_attempts[-1].result
        return None

    def get_failed_attempts(self) -> List[SQLAttempt]: # all failed SQL attempts
        return [attempt for attempt in self.sql_attempts if not attempt.success]
    
    def get_successful_attempts(self) -> List[SQLAttempt]: # all successful SQL attempts
        return [attempt for attempt in self.sql_attempts if attempt.success]
    
    def get_error_types(self) -> List[str]: # list of unique error types encountered
        return list(set([
            err.get("error_type", "unknown") 
            for err in self.error_history
        ]))
    

