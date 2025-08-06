"""Ollama LLM provider for local models."""

import json
import time
from typing import Dict, List, Optional

import requests
from loguru import logger

from config import settings
from .llm_provider import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 model_name: str = "llama2",
                 timeout: int = 120,
                 max_retries: int = 3):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama server URL
            model_name: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger.bind(component="OllamaProvider")
        
        self.logger.info(f"Initialized Ollama provider with model: {self.model_name} at {self.base_url}")
    
    def generate_answer(self, 
                       query: str, 
                       context: List[str], 
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7) -> Dict[str, any]:
        """
        Generate an answer using Ollama.
        
        Args:
            query: User question
            context: List of relevant text chunks
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Create prompt
            prompt = self.create_prompt(query, context)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens or 1000,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Generate response with retries
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    generation_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'response' in result:
                            return {
                                'answer': result['response'].strip(),
                                'model': self.model_name,
                                'generation_time': generation_time,
                                'prompt_tokens': result.get('prompt_eval_count', len(prompt.split())),
                                'completion_tokens': result.get('eval_count', len(result['response'].split())),
                                'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0),
                                'temperature': temperature,
                                'success': True,
                                'eval_duration': result.get('eval_duration', 0),
                                'load_duration': result.get('load_duration', 0)
                            }
                        else:
                            return {
                                'answer': "I couldn't generate a response. Please try again.",
                                'model': self.model_name,
                                'generation_time': generation_time,
                                'success': False,
                                'error': "No response in result"
                            }
                    else:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        if attempt < self.max_retries - 1:
                            self.logger.warning(f"Generation attempt {attempt + 1} failed: {error_msg}, retrying...")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            return {
                                'answer': f"I encountered a server error: {error_msg}",
                                'model': self.model_name,
                                'success': False,
                                'error': error_msg
                            }
                            
                except requests.exceptions.Timeout:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Request timeout on attempt {attempt + 1}, retrying...")
                        time.sleep(2 ** attempt)
                    else:
                        return {
                            'answer': "The request timed out. The model might be taking too long to respond.",
                            'model': self.model_name,
                            'success': False,
                            'error': "Request timeout"
                        }
                        
                except requests.exceptions.ConnectionError:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                        time.sleep(2 ** attempt)
                    else:
                        return {
                            'answer': "Could not connect to Ollama server. Please check if Ollama is running.",
                            'model': self.model_name,
                            'success': False,
                            'error': "Connection error"
                        }
                        
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}, retrying...")
                        time.sleep(2 ** attempt)
                    else:
                        return {
                            'answer': f"I encountered an error: {str(e)}",
                            'model': self.model_name,
                            'success': False,
                            'error': str(e)
                        }
            
        except Exception as e:
            self.logger.error(f"Error in generate_answer: {str(e)}")
            return {
                'answer': f"I encountered an error: {str(e)}",
                'model': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def get_model_name(self) -> str:
        """Get the name of the LLM model."""
        return self.model_name
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama availability check failed: {str(e)}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                self.logger.error(f"Failed to list models: HTTP {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model to Ollama."""
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes for model download
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
    
    def create_prompt(self, query: str, context: List[str]) -> str:
        """Create a specialized prompt for Ollama."""
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a helpful assistant that answers questions based on provided context. Please follow these instructions:

1. Use only the information provided in the context to answer the question
2. If the answer is not in the context, say "I don't have enough information to answer this question"
3. Be accurate and concise
4. If you reference information, mention which source it comes from

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
