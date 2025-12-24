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
    
    # SQL generation history
    sql_attempts: List[SQLAttempt] = field(default_factory=list)
    
    # Error tracking
    last_error: Optional[Dict[str, Any]] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Question analysis (cached)
    question_analysis: Optional[Dict[str, Any]] = None
    
    # Execution results
    successful_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)

    def has_schema(self):
        return True if self.schema else False
    
    def has_examples(self):
        return True if self.examples else False

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
        Includes only the most relevant information.
        """
        context_parts = []
        
        # Schema
        if self.schema:
            context_parts.append(f"Database Schema:\n{self._format_schema()}")
        
        # Examples
        if self.examples:
            context_parts.append(f"\nFew-shot Examples:\n{self._format_examples()}")
        
        # Previous attempts
        if self.sql_attempts:
            context_parts.append(f"\nPrevious SQL Attempts:\n{self._format_attempts()}")
        
        # Last error
        if self.last_error:
            context_parts.append(f"\nLast Error:\n{self._format_error(self.last_error)}")
        
        return "\n".join(context_parts)
