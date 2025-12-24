"""
NL2SQL Agent State Machine
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from langchain_ollama import OllamaLLM

from agent2.states import AgentState, ActionType, classify_error
from agent2.memory import AgentMemory
from agent2.prompts import PromptBuilder
from agent2.workers import AgentWorker
from models import extract_sql


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentConfig:
    """Configuration for agent behavior"""
    
    def __init__(
        self,
        max_iterations: int = 10,
        max_refinements: int = 3,
        verbose: bool = True
    ):
        self.max_iterations = max_iterations
        self.max_refinements = max_refinements
        self.verbose = verbose

@dataclass
class Decision:
    action: ActionType
    params: Optional[Dict[str, str]] = None
    confidence: Optional[float] = None

    def is_structually_valid(self):
        if self.action not in ActionType:
            return False
        if self.action == ActionType.FEW_SHOT_SELECT:
            return self.params is not None
        return True



class NL2SQLAgent:
    def __init__(
        self,
        config: AgentConfig = None,
    ):
      """
      Initialize the agent

      """
      self.config = config
      self.max_iterations = config.max_iterations
      self.max_refinements = config.max_refinements
      self.verbose = config.verbose

      self.prompt_builder = PromptBuilder()
      self.worker = AgentWorker()

      self.current_state = AgentState.SQL_STATE
      self.llm = self._load_model()
        
    def run(self, question: str, db_id: str, db_path: str) -> Dict[str, Any]:
        """메인 실행 루프"""
        self.memory = AgentMemory(question=question)
        self.state = AgentState.GENERATE_SQL
        self.worker.db_id = db_id
        self.worker.db_path = db_path
        trace = []
        
        while True:
            decision = self.decide(self.state, self.memory)
        

    def _load_model(self):
        return OllamaLLM(model="qwen2.5-coder:7b",
                         temperature=0, 
                         streaming=False, 
                         verbose=True)

    def execute_sql(self, sql: str):
        from models import run_db
        try:
            result = run_db(sql, f"sqlite:///{self.db_path}")

            return {
                "result": result,
                "success": True
            }

        except Exception as e:
            error_msg = str(e)
            error_type = classify_error(error_msg)

            return {
                "error_message": error_msg,
                "error_type": error_type.value,
                "success": False
            }