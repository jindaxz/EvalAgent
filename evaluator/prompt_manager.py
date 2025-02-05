from enum import Enum, auto
from typing import Dict, Any

class BasePrompt(Enum):
    """Base class for prompt enums with template and output formatting"""
    @property
    def template(self) -> str:
        return self.value['template']
    
    @property
    def criteria(self) -> str:
        return self.value.get('criteria', '')
    
    @property
    def formatter(self) -> str:
        return self.value['formatter']
    
    @classmethod
    def get_prompt_type(cls, name: str) -> 'BasePrompt':
        return cls[name.upper()]

class EvaluationType(BasePrompt):
    """Enumeration of different evaluation prompt types with JSON formatting"""
    RELEVANCE = {
        'template': (
            "Evaluate the relevance of the answer to the question and context.\n"
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Does the answer directly address the question?\n"
            "2. Is the answer supported by the provided context?\n"
            "3. Does the answer stay focused on the key points?"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- relevance_score (float between 0-1)\n"
            "- reasons (array of 3 short strings)\n"
            "- confidence (float between 0-1)\n"
            "Example:\n"
            "```json\n"
            '{"relevance_score": 0.85, "reasons": ["Directly addresses question", '
            '"Uses context effectively", "Stays focused"], "confidence": 0.92}\n'
            "```"
        )
    }
    
    COHERENCE = {
        'template': (
            "Assess the coherence and clarity of the answer.\n"
            "Question: {question}\nAnswer: {answer}\n"
            "Consider these aspects: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Logical flow of ideas\n2. Grammatical correctness\n"
            "3. Readability and structure\n4. Consistency within the answer"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- coherence_score (float between 0-1)\n"
            "- strengths (array of 2 short strings)\n"
            "- weaknesses (array of 2 short strings)\n"
            "Example:\n"
            "```json\n"
            '{"coherence_score": 0.78, "strengths": ["Clear structure", "Good transitions"], '
            '"weaknesses": ["Some run-on sentences", "Abrupt conclusion"], "confidence": 0.88}\n'
            "```"
        )
    }
    
    FACTUAL_ACCURACY = {
        'template': (
            "Evaluate the factual accuracy based on the provided context.\n"
            "Context: {context}\nAnswer: {answer}\n"
            "Accuracy criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Alignment with contextual facts\n2. Absence of contradictions\n"
            "3. Support from authoritative sources (if applicable)"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- accuracy_score (float between 0-1)\n"
            "- supported_claims (array of strings)\n"
            "- unsupported_claims (array of strings)\n"
            "Example:\n"
            "```json\n"
            '{"accuracy_score": 0.92, "supported_claims": ["Climate change drivers", '
            '"CO2 impact"], "unsupported_claims": ["Mention of solar flares"], '
            '"confidence": 0.95}\n'
            "```"
        )
    }

class PromptManager:
    """Manages prompt construction with JSON output formatting"""
    
    def __init__(self, default_type: EvaluationType = EvaluationType.RELEVANCE):
        self.default_type = default_type
        self.custom_prompts: Dict[str, Dict[str, str]] = {}
    
    def build_prompt(
        self,
        question: str,
        context: str,
        answer: str,
        eval_type: EvaluationType = None,
        **kwargs: Any
    ) -> str:
        """
        Construct an evaluation prompt with JSON formatting instructions
        
        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            eval_type: Type of evaluation to perform
            kwargs: Additional template parameters
            
        Returns:
            Formatted evaluation prompt with JSON instructions
        """
        eval_type = eval_type or self.default_type
        
        return eval_type.template.format(
            question=question,
            context=context,
            answer=answer,
            criteria=eval_type.criteria,
            formatter=eval_type.formatter,
            **kwargs
        )
    
    def add_custom_prompt(
        self,
        name: str,
        template: str,
        criteria: str = "",
        formatter: str = "",
        override: bool = False
    ) -> None:
        """
        Register a new custom prompt type with JSON formatting
        
        Args:
            name: Unique name for the prompt type
            template: Prompt template string with {formatter} placeholder
            criteria: Evaluation criteria description
            formatter: JSON formatting instructions with example
            override: Whether to overwrite existing prompt
        """
        if not override and name in self.custom_prompts:
            raise ValueError(f"Prompt '{name}' already exists")
            
        self.custom_prompts[name] = {
            'template': template,
            'criteria': criteria,
            'formatter': formatter
        }
    
    def get_custom_prompt(self, name: str) -> Dict[str, str]:
        """Retrieve a registered custom prompt with formatting"""
        if name not in self.custom_prompts:
            raise ValueError(f"Custom prompt '{name}' not found")
        return self.custom_prompts[name]

# Example usage
if __name__ == "__main__":
    # Create prompt manager with default evaluation type
    pm = PromptManager(default_type=EvaluationType.RELEVANCE)
    
    # Build a relevance evaluation prompt
    question = "What causes climate change?"
    context = "Scientific consensus attributes climate change to human activities..."
    answer = "Burning fossil fuels releases greenhouse gases that trap heat."
    
    prompt = pm.build_prompt(
        question=question,
        context=context,
        answer=answer,
        eval_type=EvaluationType.RELEVANCE
    )
    
    print("Relevance Evaluation Prompt:")
    print(prompt)
    
    # Add and use a custom prompt
    pm.add_custom_prompt(
        name="completeness",
        template=(
            "Evaluate answer completeness:\n"
            "Question: {question}\nAnswer: {answer}\n"
            "Required aspects: {required_aspects}\nEvaluation:"
        ),
        criteria="Coverage of all question aspects"
    )
    
    custom_prompt = pm.build_prompt(
        question="Explain machine learning",
        context="",
        answer="ML is about algorithms learning from data",
        eval_type=EvaluationType.FACTUAL_ACCURACY,
        required_aspects="Definition, examples, applications"
    )
    
    print("\nCustom Prompt:")
    print(custom_prompt)