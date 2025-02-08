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
        two_line_answer = f"    1. {answer[0]}\n    2. {answer[1]}" # answer[0] should be ground truth , answer[1] should be candidate answer
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
                "Q1": -1, "Q2": -1, "Q3": -1, "Q4": -1,
                'error': str(e)
            }
            
class RefusalAccuracyEvaluator(RAGEvaluator):
    
    def pre_process(self, question, context, answer):
        pass
    
    def call_llm(self, processed_data):
        pass
    
    def post_process(self, llm_response):
        pass
    

    def evaluate(self, question, context, answer):
        prompt1 = self.prompt_manager.build_prompt(
            question = question,
            context = context,
            answer = answer,
            eval_type = EvaluationType.REFUSAL
        )
        
        resp1 = self.llm.generate(prompt1)
        
        prompt2 = self.prompt_manager.build_prompt(
            question = question,
            context = context,
            answer = answer,
            eval_type = EvaluationType.UNDERSPECIFIED_CHECK
        )
        
        resp2 = self.llm.generate(prompt2)
        
        try:
            response_text = resp1.strip().replace('```json', '').replace('```', '')
            result1 = json.loads(response_text)
            
            score1 = {
                "refusal": result1['refusal'],
                "reason": result1['reason']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {'refusal': 0xffffffff, "error": str(e)}

        try:
            response_text = resp2.strip().replace('```json', '').replace('```', '')
            result2= json.loads(response_text)
            
            score2 = {
                "underspecifie_check": result2['underspecifie_check'],
                "reason": result2['reason']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {'underspecifie_check': 0, "error": str(e)}
            
        return {'refusal_result': score1, "underspecifie_check_score": score2}
            
        