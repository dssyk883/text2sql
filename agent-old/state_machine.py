"""
NL2SQL Agent State Machine
"""

import logging
from typing import Dict, Any

from langchain_ollama import OllamaLLM

from agent.states import AgentState
from agent.memory import AgentMemory
from agent.tools import AgentTools
from agent.prompts import PromptBuilder
from models import extract_sql


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentConfig:
    """Configuration for agent behavior"""
    
    def __init__(
        self,
        max_iterations: int = 10,
        max_refinements: int = 3,
        validate_before_execute: bool = True,
        verbose: bool = True
    ):
        self.max_iterations = max_iterations
        self.max_refinements = max_refinements
        self.validate_before_execute = validate_before_execute
        self.verbose = verbose


class NL2SQLAgent:
    def __init__(
        self,
        model_type: str = "qwen",
        k: int = 3,
        config: AgentConfig = None,
    ):
      """
      Initialize the agent

      Args:
            model_type: "qwen" or "sonnet"
            k: the number of examples
            config: Configuration for agent behavior
      """
      self.config = config
      self.model_type = model_type
      self.k = k
      self.max_iterations = config.max_iterations
      self.max_refinements = config.max_refinements
      self.verbose = config.verbose

      self.prompt_builder = PromptBuilder(config=self.config)

      self.current_state = AgentState.ANALYZE
      self.iteration_count = 0
      self.refinement_count = 0

    def run(self, question: str, db_id: str, db_path: str) -> Dict[str, Any]:
        """
        Run the agent on a natural language question
        
        :param question: User's NL question

        :return Dict - SQLAttempt and trace:
            sql: str - Final SQL query
            result: Query execution result
            state: str
            success: bool
            error: Error msg when failed
        """
        self.tools = AgentTools(model_type=self.model_type, db_id=db_id, db_path=db_path)
        memory = AgentMemory(question=question)

        trace = []

        self.current_state = AgentState.ANALYZE
        self.iteration_count = 0
        self.refinement_count = 0

        logger.info(f"Starting NL2SQL agent for question: {question}")

        # main loop
        while self.current_state not in [AgentState.DONE, AgentState.FAIL]:
            self.iteration_count += 1

            # Max iteration check
            if self.iteration_count > self.max_iterations:
                logger.warning(f"Max iterations ({self.max_iterations}) reached")
                self.current_state = AgentState.FAIL
                break

            if self.verbose:
                logger.info(f"Iteration {self.iteration_count}: State = {self.current_state.value}")

            try:
                step_result = self._execute_state(self.current_state, memory)
                trace.append(step_result)

                self.current_state = step_result["next_state"]

            except Exception as e:
                logger.error(f"Error in state {self.current_state.value}: {str(e)}")
                trace.append({
                    "state": self.current_state.value,
                    "error": str(e),
                    "next_state": AgentState.FAIL
                })
                self.current_state = AgentState.FAIL
        # while ends

        # compile final result
        final_result = self._compile_result(memory, trace)

        if self.verbose:
            logger.info(f"Agent finished: {final_result['success']}")
        
        return final_result
    
    def _execute_state(
        self, 
        state: AgentState, 
        memory: AgentMemory
    ) -> Dict[str, Any]:
        """
        Execute logic for a specific state.
        
        Args:
            state: Current state
            memory: Agent memory
            
        Returns:
            Dict with state execution results and next_state
        """
        if state == AgentState.ANALYZE:
            return self._handle_analyze(memory)
        
        elif state == AgentState.GATHER_INFO:
            return self._handle_gather_info(memory)
        
        elif state == AgentState.GENERATE_SQL:
            return self._handle_generate_sql(memory)
        
        elif state == AgentState.VALIDATE_SQL:
            return self._handle_validate_sql(memory)
        
        elif state == AgentState.EXECUTE_SQL:
            return self._handle_execute_sql(memory)
        
        elif state == AgentState.REFINE_SQL:
            return self._handle_refine_sql(memory)
        
        else:
            return {
                "state": state.value,
                "action": "unknown",
                "next_state": AgentState.FAIL
            }
        
    def _handle_analyze(self, memory: AgentMemory) -> Dict[str, Any]:
        """Handle ANALYZE state - decide what information needed"""
        
        prompt = self.prompt_builder.build_prompt(AgentState.ANALYZE, memory)
        action = self._get_llm_action(prompt)
        
        if action == "get_db_schema":
            next_state = AgentState.GATHER_INFO
            
        elif action == "search_similar_examples":
            examples = self.tools.search_similar_examples(memory.question)
            memory.examples = examples
            next_state = AgentState.GATHER_INFO
            
        elif action == "generate_sql":
            next_state = AgentState.GENERATE_SQL
            
        else:
            # Invalid action, retry or fail
            next_state = AgentState.GATHER_INFO
        
        return {
            "state": AgentState.ANALYZE.value,
            "action": action,
            "next_state": next_state
        }
    
    def _handle_gather_info(self, memory: AgentMemory) -> Dict[str, Any]:
        """Handle GATHER_INFO state - collect schema/examples if needed"""
        needs_schema = not memory.has_schema()
        needs_examples = not memory.has_examples()
        actions = []

        if needs_schema:
            schema = self.tools.get_db_schema()
            memory.schema = schema['structured']
            memory.schema_summary = schema['summary']
            actions.append("get_db_schema")

        if needs_examples:
            examples = self.tools.search_similar_examples(memory.question, k=self.k)
            memory.examples = examples
            actions.append("search_similar_examples")

        next_state = AgentState.GENERATE_SQL if memory.has_examples() and memory.has_schema() else AgentState.GATHER_INFO

        return {
            "state": AgentState.GATHER_INFO.value,
            "action": ", ".join(actions) if actions else "ready",
            "next_state": next_state
        }
    
    def _handle_generate_sql(self, memory: AgentMemory) -> Dict[str, Any]:
        """Handle GENERATE_SQL state - generate SQL query"""
        prompt = self.prompt_builder.build_prompt(AgentState.GENERATE_SQL, memory)
        response = self._get_llm_response(prompt)
        sql = extract_sql(response.strip())
        memory.add_sql_attempt(
            sql=sql,
            state=AgentState.GENERATE_SQL.value,
            success=False
        )
        
        return {
            "state": AgentState.GENERATE_SQL.value,
            "sql": sql,
            "next_state": AgentState.VALIDATE_SQL
        }
    
    def _handle_validate_sql(self, memory: AgentMemory) -> Dict[str, Any]:
        """Handle VALIDATE_SQL state - check syntax before execution"""

        last_sql = memory.get_last_sql()
        validation = self.tools.validate_sql_syntax(last_sql)
        if validation["valid"]:
            return {
                "state": AgentState.VALIDATE_SQL.value,
                "validation": "passed",
                "next_state": AgentState.EXECUTE_SQL
            }
        else: # Syntax error
            memory.last_error = {
                "message" : "; ".join(validation["errors"]),
                "error_type": "syntax"
            }
            return {
                "state": AgentState.VALIDATE_SQL.value,
                "validation": "failed",
                "errors": validation["errors"],
                "next_state": AgentState.REFINE_SQL
            }
    
    def _handle_execute_sql(self, memory: AgentMemory) -> Dict[str, Any]:
        """Handle EXECUTE_SQL state - run the query"""

        last_sql = memory.get_last_sql()
        result = self.tools.execute_sql(last_sql)

        if result["success"]:
            # Success! Update memory and finish
            memory.add_sql_attempt(
                sql=last_sql,
                state=AgentState.EXECUTE_SQL.value,
                success=True,
                result=result["result"]
            )
            
            return {
                "state": AgentState.EXECUTE_SQL.value,
                "success": True,
                "result": result["result"],
                "next_state": AgentState.DONE
            }
        else:
            # Execution failed, need refinement
            error_info = {
                "error": result["error"],
                "error_type": result["error_type"]
            }

            memory.add_sql_attempt(
                sql=last_sql,
                state=AgentState.EXECUTE_SQL.value,
                error=error_info,
                success=False
            )
            
            # Check if we can still refine
            if self.refinement_count >= self.max_refinements:
                logger.warning(f"Max refinements ({self.max_refinements}) reached")
                return {
                    "state": AgentState.EXECUTE_SQL.value,
                    "success": False,
                    "error": result["error"],
                    "next_state": AgentState.FAIL
                }
            
            self.refinement_count += 1
            
            return {
                "state": AgentState.EXECUTE_SQL.value,
                "success": False,
                "error": error_info,
                "error_type": error_info["error_type"],
                "next_state": AgentState.REFINE_SQL
            }
        
    def _handle_refine_sql(self, memory: AgentMemory) -> Dict[str, Any]:
        """Handle REFINE_SQL state - fix errors"""
        
        prompt = self.prompt_builder.build_prompt(AgentState.REFINE_SQL, memory)
        
        response = self._get_llm_response(prompt)
        refined_sql = extract_sql(response.strip())
        
        # Record refinement attempt
        memory.add_sql_attempt(
            sql=refined_sql,
            state=AgentState.REFINE_SQL.value,
            success=False  # Not executed yet
        )
        
        return {
            "state": AgentState.REFINE_SQL.value,
            "refined_sql": refined_sql,
            "refinement_count": self.refinement_count,
            "next_state": AgentState.VALIDATE_SQL
        }
    
    def _get_llm(self):
        if self.model_type == "qwen":
            return OllamaLLM(
                        model="qwen2.5-coder:7b",
                        temperature=0, 
                        streaming=False, 
                        verbose=True)
        
        if self.model_type == "mistral":
            return OllamaLLM(
                        model="mistral:7b-instruct-q5_K_M",
                        temperature=0,
                        streaming=False,
                        verbose=True)

        if self.model_type == "sonnet":
            from claude_integration import get_claude_client
            return get_claude_client()

    def _get_llm_action(self, prompt: str) -> str:
        """Get and extract action from LLM response"""
        model = self._get_llm()
        response = ""
        valid_actions = ["get_db_schema", "search_similar_examples", "generate_sql"]
        try:
            if self.model_type in ["qwen", "mistral"]:
                response = model.invoke(prompt)
            if self.model_type == "sonnet":
                message = model.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=2048,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response = message.content[0].text
            
        
        except Exception as e:
            logger.error(f"LLM({self.model_type}) get_action error: {e}")
        
        response = response.strip().lower()
        for action in valid_actions:
            if action in response:
                return action
        
        logger.warning(f"Could not parse action from: {response}")
        return "generate_sql" # default       

    def _get_llm_response(self, prompt: str) -> str:
        """Get a full LLM response"""
        model = self._get_llm()
        try:
            if self.model_type in ["qwen", "mistral"]:
                response = model.invoke(prompt)
            if self.model_type == "sonnet":
                message = model.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=2048,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                response = message.content[0].text
            return response.strip()
        
        except Exception as e:
            logger.error(f"LLM({self.model_type}) get_response error: {e}")
            # Fallback SQL 
            return "SELECT * LIMIT 1"
        
    # When validation prompt needed in the future
    # def _get_llm_validation(self, prompt: str) -> str:
    
    def _compile_result(
        self, 
        memory: AgentMemory, 
        trace: list
    ) -> Dict[str, Any]:
        """
        Compile final result from memory and trace.
        
        Args:
            memory: Agent memory
            trace: Execution trace
            
        Returns:
            Final result dictionary
        """
        successful_attempts = memory.get_successful_attempts()
        
        if successful_attempts:
            last_success = successful_attempts[-1]
            return {
                "success": True,
                "sql": last_success.sql,
                "schema": memory.schema_summary,
                "result": last_success.result,
                "iterations": self.iteration_count,
                "refinements": self.refinement_count,
                "trace": trace if self.verbose else None,
                "error": memory.last_error.get("error") if memory.last_error else None,
                "error_type": memory.last_error.get("error_type") if memory.last_error else None,
            }
        else:
            return {
                "success": False,
                "schema": memory.schema_summary,
                "error": memory.last_error.get("error") if memory.last_error else "Unknown error",
                "error_type": memory.last_error.get("error_type") if memory.last_error else "unknown",
                "sql": [attempt.sql for attempt in memory.sql_attempts],
                "iterations": self.iteration_count,
                "refinements": self.refinement_count,
                "trace": trace if self.verbose else None
            }