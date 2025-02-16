from abc import ABC, abstractmethod
import os
from openai import OpenAI
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from huggingface_hub import login
import torch
import time
from vllm import LLM

class LLMClient(ABC):
    """Base class for LLM clients with standardized invocation interface"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Execute LLM call with given prompt and return response text
        
        Args:
            prompt: Input text/prompt for the LLM
            
        Returns:
            Generated text response from LLM
        """
        pass

class OpenAIClientLLM(LLMClient):
    """Concrete implementation using OpenAI-compatible client"""
    
    def __init__(self, 
                 model: str = "meta-llama/Llama-3.3-70B-Instruct",
                 system_message: str = "You are a helpful assistant",
                 base_url: str = "https://api-eu.centml.com/openai/v1",
                 **kwargs):
        """
        Initialize OpenAI-style client
        
        Args:
            model: Model identifier string
            system_message: System prompt for conversation context
            base_url: API endpoint URL
            kwargs: Additional parameters for completions
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_message = system_message
        self.params = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0.5,
            **kwargs
        }

    def generate(self, prompt: str) -> str:
        """Execute synchronous LLM call"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.params
        )
        
        return completion.choices[0].message.content

class HTTPLLM(LLMClient):
    """Concrete implementation using generic HTTP API endpoint"""
    
    def __init__(self,
                 model: str = "deepseek_r1",
                 base_url: str = "https://cloud.luchentech.com/api/maas/chat/completions",
                 system_message: str = "You are a helpful and harmless assistant. You should think step-by-step.",
                 **kwargs):
        """
        Initialize HTTP client
        
        Args:
            model: Model identifier string
            base_url: API endpoint URL
            system_message: System prompt for conversation context  
            kwargs: Additional parameters for completions
        """
        api_key = os.getenv("MAAS_API_KEY")
        if not api_key:
            raise ValueError("MAAS_API_KEY environment variable required")
            
        self.model = model
        self.base_url = base_url
        self.system_message = system_message
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.params = {
            "stream": False,
            "max_tokens": 32000,
            **kwargs
        }

    def generate(self, prompt: str) -> str:
        """Execute synchronous HTTP request"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            **self.params
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

class HFClient(LLMClient):
    """Concrete implementation for local Hugging Face models (GPU-only)"""

    def __init__(self,
                 model_path: str,
                 system_message: str = "You are a helpful assistant",
                 **kwargs):
        """
        Initialize local Hugging Face client

        Args:
            model_path: Path or name of Hugging Face model
            system_message: System prompt for conversation context
            kwargs: Additional parameters
        """
        # Retrieve Hugging Face token from environment variable
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")

        # Authenticate with Hugging Face
        login(token=hf_token)
        
        self.model_path = model_path
        self.system_message = system_message
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, token=hf_token).to(self.device)
        
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        # Set pad_token_id based on the tokenizer's eos_token_id.
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        self.generation_config.pad_token_id = eos_id

    def generate(self, prompt: str, **kwargs) -> str:
        # Determine how to format the prompt
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            # Generate input tensor using the chat template
            input_tensor = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
            # Decode the formatted prompt to know its text length
            formatted_prompt = self.tokenizer.decode(input_tensor[0], skip_special_tokens=True)
        else:
            # Fallback: manual formatting of the prompt
            full_prompt = f"{self.system_message}\n\nUser: {prompt}\n\nAssistant:"
            input_tensor = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
            formatted_prompt = full_prompt  # Use the full prompt as formatted prompt

        # Set max_new_tokens if not provided
        max_new_tokens = kwargs.pop('max_new_tokens', 1000)
        
        # Start time counting before generation
        start_time = time.time()
        
        outputs = self.model.generate(
            input_tensor,
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # End time counting after generation
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Inference time: {elapsed_time:.2f} seconds")
        
        # Decode the complete generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt portion to get only the assistant's response
        assistant_response = generated_text[len(formatted_prompt):]
        
        # Additional filtering for DeepSeek: remove everything before and including '</think>'
        if "</think>" in assistant_response:
            idx = assistant_response.find("</think>")
            assistant_response = assistant_response[idx + len("</think>"):].strip()
        
        return assistant_response.strip()



class VLLMClient(LLMClient):
    """Optimized implementation using vLLM inference engine"""
    
    def __init__(self,
                 model_path: str = "deepseek-ai/deepseek-r1",
                 system_message: str = "You are a helpful and harmless assistant. Think step-by-step.",
                 quant_method: str = "awq",
                 **kwargs):
        """
        Initialize vLLM client with optimized configurations
        
        Args:
            model_path: Local path or HF repo ID
            system_message: System prompt for context
            quant_method: Quantization method (awq, gptq, none)
            kwargs: Additional engine parameters
        """
        self.model_path = model_path
        self.system_message = system_message
        self.quant_config = self._get_quant_config(quant_method)
        
        # Initialize vLLM engine with optimized params
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=EngineArgs(
                model=self.model_path,
                quantization=self.quant_config,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=16384,
                enforce_eager=True,  # Bypass graph capture for dynamic inputs
                **kwargs
            )
        )
        
        self.generation_config = {
            "max_tokens": 32000,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0.5
        }

    def _get_quant_config(self, method: str):
        """Configure quantization method"""
        if method == "awq":
            return AWQConfig(quant_method="awq")
        elif method == "gptq":
            return GPTQConfig(bits=4, group_size=128)
        return None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Execute optimized generation with vLLM"""
        sampling_params = SamplingParams(
            **{**self.generation_config, **kwargs}
        )
        
        # Format messages with system prompt
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Use vLLM's optimized tokenizer
        tokenizer = self.engine.engine.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            formatted_prompt = f"{self.system_message}\n\nUser: {prompt}\nAssistant:"

        # Async streaming generation
        results_generator = self.engine.generate(
            formatted_prompt, 
            sampling_params,
            request_id=uuid.uuid4().hex
        )

        # Collect output
        async for request_output in results_generator:
            return request_output.outputs[0].text

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Optimized batch processing with continuous batching"""
        sampling_params = SamplingParams(
            **{**self.generation_config, **kwargs}
        )
        
        outputs = self.engine.generate_batch(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        return [output.outputs[0].text for output in outputs]



# Example usage
if __name__ == "__main__":
    # Set environment variables first
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    os.environ["MAAS_API_KEY"] = "your-api-key-here"

    # OpenAI client example
    openai_llm = OpenAIClientLLM()
    print("OpenAI response:", openai_llm.generate("Hello world!"))
    
    # HTTP client example
    http_llm = HTTPLLM()
    print("HTTP response:", http_llm.generate("Explain quantum computing in 3 sentences"))