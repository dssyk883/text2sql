"""
NL2SQL Agent State Machine
"""

import logging
from typing import Dict, Any

from langchain_ollama import OllamaLLM

from agent2.states2 import AgentState
from agent.memory import AgentMemory
from agent.tools import AgentTools
from agent2.prompts2 import PromptBuilder
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

      self.current_state = AgentState.PLAN
      self.iteration_count = 0
      self.refinement_count = 0

    def run(self, query: str) -> Dict[str, Any]:
        """메인 실행 루프"""
        self.memory = AgentMemory(query=query)
        self.state = AgentState.GENERATE_SQL
        
        while self.state not in [AgentState.DONE, AgentState.FAIL]:
            print(f"Current State: {self.state.value}")
            
            if self.state == AgentState.GENERATE_SQL:
                self._handle_generate_state()
            elif self.state == AgentState.VALIDATE_SQL:
                self._handle_validate_state()
            elif self.state == AgentState.ANALYZE_ERROR:
                self._handle_analyze_error()
            elif self.state == AgentState.ANALYZE_ISSUE:
                self._handle_analyze_issue()
            elif self.state == AgentState.EXECUTE_SQL:
                self._handle_execute_state()
            elif self.state == AgentState.ANALYZE:
                self._handle_analyze_state()
            elif self.state == AgentState.REFINE_SQL:
                self._handle_refine_state()
                
        return self._get_result()
    
    def _handle_generate_state():
        