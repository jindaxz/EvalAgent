from __future__ import annotations # for pervious python version e.g. 3.9
import json
from typing import List, Dict, Union
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
            
        

class LearningFacilitationEvaluator(RAGEvaluator):
    def pre_process(
        self,
        question: str|List[str],
        context: str|List[str],
        answer: str|List[str]
    ) -> str:
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.LEARNING_FACILITATION
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            print(f"Raw LLM response: {llm_response}")
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "learning_facilitation_score": result['learning_facilitation_score'],
                "educational_strengths": result['educational_strengths'],
                "areas_for_improvement": result['areas_for_improvement'],
                "confidence": result['confidence']
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "learning_facilitation_score": -1,
                "educational_strengths": [],
                "areas_for_improvement": [],
                "confidence": -1,
                'error': str(e)
            }


class EngagementEvaluator(RAGEvaluator):
    def pre_process(
        self,
        question: Union[str, List[str]],
        context: Union[str, List[str]],
        answer: Union[str, List[str]]
    ) -> str:
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ENGAGEMENT_INDEX
        )
        
    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response: str) -> Dict[str, Union[float, List[str]]]:
        """Parse JSON response into scores dictionary"""
        try:
            print(f"Raw LLM response: {llm_response}")
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "engagement_score": result.get('engagement_score', -1),
                "engaging_elements": result.get('engaging_elements', []),
                "suggestions_for_improvement": result.get('suggestions_for_improvement', []),
                "confidence": result.get('confidence', -1)
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "engagement_score": -1,
                "engaging_elements": [],
                "suggestions_for_improvement": [],
                "confidence": -1,
                'error': str(e)
            }
        
class FactualCorrectnessEvaluator(RAGEvaluator):
    def pre_process(
        self, question: str|List[str], context: str|List[str], answer: str|List[str], **kwargs
    ) -> str:
        if "golden_answer" not in kwargs:
            raise KeyError("Missing required key: golden_answer")
        golden_answer = kwargs.get("golden_answer")
        return self.prompt_manager.build_prompt(
            answer=answer,
            eval_type=EvaluationType.FACTUAL_CORRECTNESS,
            golden_answer=golden_answer
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
                "TP": result['TP'],
                "FP": result['FP'],
                "FN": result['FN'],
                "F1_score": 0 if (result['TP'] + result['FP'] + result['FN']) == 0 else result['TP'] / (result['TP'] + result['FP'] + result['FN']),
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "TP": -1, "FP": -1, "FN": -1, "F1_SCORE": -1,
                'error': str(e)
            }
            
class KeyPointEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2408.01262, using extracted key points generate from ground truth answer to check with generated answer,
    using the categorized key_points count to calculate generation scores. 
    It can provide completeness, hallucination and irrelevance score. 
    """
    num_key_points = 0
    def pre_process(self, question, context, answer, **kwargs):
        if "key_points" not in kwargs:
            raise KeyError("Missing required input: key_points")
        key_points = kwargs.get("golden_answer")
        
        if not isinstance(key_points, List):
            raise ValueError("key_points is type of List[str]")
        
        if len(key_points) == 0:
            raise ValueError("key_points is an empty List, which is invalid")
        
        self.num_key_points = len(key_points)
        formatted_key_points = "\n".join([f"{i + 1}. {kp}" for i, kp in enumerate(key_points)])
        
        return self.prompt_manager.build_prompt(
            question = question,
            answer = answer,
            eval_type = EvaluationType.KEY_POINT,
            key_points = formatted_key_points,
        )
        
    def call_llm(self, processed_data):
        return self.llm.generate(processed_data)
    
    def post_process(self, llm_response):
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            
            scores = {
                "completeness_score": len(result['complete_ids']) / self.num_key_points,
                "irrelevant_score": len(result['irrelevant_ids']) / self.num_key_points,
                "hallucination_score": len(result['hallucinate_ids']) / self.num_key_points,
                "raw_output" : result
            }
            
            return scores
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "completeness_score": -1,
                "irrelevant_score": -1,
                "hallucination_score": -1,
                "raw_output" : response_text,
                "error": str(e),
            }
        