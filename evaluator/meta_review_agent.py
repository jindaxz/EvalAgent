from llm import OpenAIClientLLM
from prompt_manager import PromptManager, EvaluationType
import json

class MetaReviewerLearningFacilitationAgent:
    def __init__(self, llm, prompt_manager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    def build_meta_prompt(self, question, context, answer, score1, reason1, score2, reason2):
        return self.prompt_manager.build_prompt(
            question=question,
            context=context,
            answer=answer,
            score1=score1,
            reason1=reason1,
            score2=score2,
            reason2=reason2,
            eval_type=EvaluationType.META_REVIEW_LEARNING_FACILITATION
        )

    def get_meta_scores(self, question, context, answer, score1, reason1, score2, reason2):
        prompt = self.build_meta_prompt(question, context, answer, score1, reason1, score2, reason2)
        response = self.llm.generate(prompt)
        return self.parse_response(response)

    def parse_response(self, response):
        response_text = response.strip().replace('```json', '').replace('```', '')
        result = json.loads(response_text)
        return {
            "meta_score": result['meta_score'],
            "reasons": result['reasons']
        }


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    llm = OpenAIClientLLM(
        model="gpt-4o",
        base_url='https://api.openai.com/v1/'
    )
    prompt_manager = PromptManager(
        default_type=EvaluationType.META_REVIEW_LEARNING_FACILITATION)
    meta_reviewer = MetaReviewerLearningFacilitationAgent(llm, prompt_manager)

    question = "What is the capital of France?"
    context = "The capital of France is Paris. It is known for its art, fashion, and culture."
    answer = "The capital of France is Paris."

    score1 = 0.8
    reason1 = "Clear explanations and good examples."
    score2 = 0.7
    reason2 = "Needs more depth and visual aids."

    meta_scores = meta_reviewer.get_meta_scores(
        question, context, answer, score1, reason1, score2, reason2)
    print(meta_scores)