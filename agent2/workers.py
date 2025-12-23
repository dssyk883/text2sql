from typing import Dict, Any, List, Optional
import json
from utils.jaccard import retrieve_jaccard_examples
from utils.intent_clustering import retrieve_intent_based_examples
from utils.random_examples import create_random_examples
from agent2.memory import AgentMemory


class AgentWorker:

    def __init__(self, db_id: str = None, db_path: str = None):
        self.db_id = db_id
        self.db_path = db_path
        
    def get_db_schema(self):
        from utils.RAG_setup import summarize_schema, get_schema_safe

        schema = get_schema_safe(self.db_id)
        summary_schema = summarize_schema(schema)

        return {
            "structured": schema,
            "summary": summary_schema
        }
    
    def search_similar_examples(self,
                                question: str,
                                k: int,
                                strategy: str) -> List[Dict[str, Any]]:
        examples = []
        if strategy == "random":
            examples = create_random_examples(k)
        
        elif strategy == "intent clustering":
            examples = retrieve_intent_based_examples(question=question, k=k)
        
        elif strategy == "jaccard": 
            examples = retrieve_jaccard_examples(question=question, k=k)

        else:
            raise ValueError(
                f"Unknown Few-Shot strategy: '{strategy}'"
                f"Valid strategies: ['random', 'intent clustering, 'jaccard']"
            )
        return examples
    
    def validate_sql_syntax(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL syntax without executing.
        
        Args:
            sql: SQL query string
            
        Returns:
            Dict with "valid" (bool) and "errors" (list) keys
        """
        errors = []
        
        # Basic syntax checks
        sql_lower = sql.lower().strip()
        
        if not sql_lower.startswith("select"):
            errors.append("Query must start with SELECT")
        
        if "from" not in sql_lower:
            errors.append("Query must contain FROM clause")
        
        # Check for common syntax issues
        if sql_lower.count("(") != sql_lower.count(")"):
            errors.append("Unmatched parentheses")
        
        if sql_lower.count("'") % 2 != 0:
            errors.append("Unmatched quotes")
        
        # Check for dangerous operations (optional safety check)
        dangerous_keywords = ["drop", "delete", "truncate", "update"]
        if any(kw in sql_lower for kw in dangerous_keywords):
            errors.append("Query contains potentially dangerous operations")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }