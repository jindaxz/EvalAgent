import json
from typing import Dict, List
from evaluator.base_evaluator import RAGEvaluator
from evaluator.prompt_manager import EvaluationType


class LLMEquivalenceEvaluator(RAGEvaluator):
    def pre_process(
        self,
        question: str|List[str],
        context: str|List[str],
        answer: str|List[str]
    ) -> str:
        assert len(answer) == 2
        two_line_answer = f"    1. {answer[0]}\n    2. {answer[1]}"
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=two_line_answer,
            eval_type=EvaluationType.ANSWER_EQUIVALENCE
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "Q1": 1 if result['Q1'] == 'yes' else 0,
                "Q2": 1 if result['Q2'] == 'yes' else 0,
                "Q3": 1 if result['Q3'] == 'yes' else 0,
                "Q4": 1 if result['Q4'] == 'yes' else 0,
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0,
                'error': str(e)
            }
            
class RefusalAccuracyEvaluator(RAGEvaluator):
    
    def pre_process(self, question, context, answer):
        return super().pre_process(question, context, answer)
    
    def call_llm(self, processed_data):
        return super().call_llm(processed_data)
    
    
    def post_process(self, llm_response):
        return super().post_process(llm_response)