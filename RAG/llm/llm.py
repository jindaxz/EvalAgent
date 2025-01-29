# llm.py
from abc import ABC, abstractmethod
import torch
import openai

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    GenerationConfig = None

class BaseLLM(ABC):
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class HuggingFaceLLM(BaseLLM):
    """Base class for all Hugging Face models with automatic device detection"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.generation_config = None

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.generation_config.pad_token_id = self.generation_config.eos_token_id

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            input_tensor,
            generation_config=self.generation_config,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class DeepSeekLLM(HuggingFaceLLM):
    """Specialized implementation for DeepSeek models"""
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        super().__init__(model_path)

class QwenLLM(HuggingFaceLLM):
    """Specialized implementation for Qwen models"""
    def __init__(self, model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model_path)

class OpenAILLM(BaseLLM):
    """Implementation for OpenAI API models"""
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def load_model(self) -> None:
        pass  # No explicit loading needed for OpenAI

    def generate(self, prompt: str, **kwargs) -> str:
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content