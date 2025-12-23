from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SQLAttempt:
    sql: str
    timestamp: datetime
    state: str
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
    
    # Error tracking
    last_error: Optional[Dict[str, Any]] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    analysis: Optional[Dict[str, Any]] = None
    
    # Execution results
    successful_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)

    def add_sql_attempt(
        self,
        sql: str,
        state: str,
        success: bool = False,
        error: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        confidence: Optional[float] = None
    ):
        attempt = SQLAttempt(
            sql=sql,
            timestamp=datetime.now(),
            state=state,
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
    
    def get_failed_attempts(self) -> List[SQLAttempt]: # all failed SQL attempts
        return [attempt for attempt in self.sql_attempts if not attempt.success]
    
    def get_successful_attempts(self) -> List[SQLAttempt]: # all successful SQL attempts
        return [attempt for attempt in self.sql_attempts if attempt.success]
    
    def get_error_types(self) -> List[str]: # list of unique error types encountered
        return list(set([
            err.get("error_type", "unknown") 
            for err in self.error_history
        ]))
    
    def to_context_string(self) -> str:
        """
        Convert memory to a formatted string for LLM context.
        """
        context_parts = []
        
        # Schema
        if self.schema_summary:
            context_parts.append(f"Database Schema:\n{self.schema_summary}")
        
        # Examples
        if self.examples:
            last_strategy = self.few_shot_history[-1]
            context_parts.append(f"""\nFew-shot:\n
                                 Strategy: {last_strategy["strategy"]}, k: {last_strategy["k"]}\n
                                 Examples:\n{self.examples()}"""
                                 )
        
        # Previous attempts
        if self.sql_attempts:
            recent_num = 3
            recent_attempts = self.sql_attempts[-recent_num:]
            formatted = self._format_attempts(recent_attempts)
            context_parts.append(f"\n{recent_num} Recent SQL Attempts:\n{formatted}")
        
        return "\n".join(context_parts)
    
    def _format_attempts(self, attempts: Optional[List[SQLAttempt]] = None) -> str:
        """
        Format SQLAttempt list for LLM context
        Args:
            attempts: List of attempts to format. If None, get all attempts
        """
        attempts_to_format = attempts if attempts is not None else self.sql_attempts

        if not attempts_to_format:
            return "No previous attempts."
        
        formatted = []
        for idx, attempt in enumerate(attempts_to_format, 1):
            status = "Success" if attempt.success else "Failed"
            parts = [f"Attempt #{idx} [{status}]", f"SQL: {attempt.sql}"]
            if attempt.error:
                error_message = attempt.error.get('error_message', 'Unknown error message')
                error_type = attempt.error.get('error_type', 'Unknown error type')
                parts.append(f"Error type: {error_type}, Error message: {error_message}")
            
            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)
