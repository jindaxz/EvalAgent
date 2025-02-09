from __future__ import annotations # for pervious python version e.g. 3.9
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .prompt_manager import EvaluationType

class RAGEvaluator(ABC):
    """Base class for evaluating RAG outputs using LLM-as-a-judge pattern."""
    
    def __init__(self, llm: Any, prompt_manager: Any):
        """
        Initialize the evaluator with an LLM instance and prompt manager.
        
        Args:
            llm: Initialized LLM instance (e.g., OpenAI, Anthropic, etc.)
            prompt_manager: Prompt management system for constructing evaluation prompts
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
    
    @abstractmethod
    def pre_process(self, question: str|List[str], context: str|List[str], answer: str|List[str], **kwargs: Any ) -> Any:
        """
        Prepare and format the evaluation input.
        
        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            kwargs: Additional template parameters (golded_answer)
            
        Returns:
            Processed data ready for LLM evaluation
        """
        pass
    
    @abstractmethod
    def call_llm(self, processed_data: Any) -> str:
        """
        Execute the LLM call with the processed evaluation prompt.
        
        Args:
            processed_data: Formatted evaluation prompt from pre_process
            
        Returns:
            Raw LLM response string
        """
        pass
    
    @abstractmethod
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """
        Convert LLM response into evaluation scores.
        
        Args:
            llm_response: Raw response string from LLM
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        pass
    
    def evaluate(self, answer: str, question: str = "", context: str = "",  **kwargs) -> Dict[str, float]:
        """
        Main evaluation workflow.
        
        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            kwargs: Additional template parameters (golded_answer)
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        processed_data = self.pre_process(question, context, answer, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(llm_response)