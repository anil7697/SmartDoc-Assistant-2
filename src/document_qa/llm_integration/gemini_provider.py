"""Google Gemini LLM provider."""

import time
from typing import Dict, List, Optional

import google.generativeai as genai
from loguru import logger

from config import settings
from .llm_provider import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash",
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key (uses settings if not provided)
            model_name: Gemini model to use
            max_retries: Maximum number of retries
            retry_delay: Delay between retries
        """
        google_api_key = "AIzaSyBYbzeGj9cE70YXax5-_FnRzskJyEeWYxA"
        self.api_key = api_key or google_api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger.bind(component="GeminiProvider")
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        try:
            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"Initialized Gemini provider with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise
    
    def generate_answer(self, 
                       query: str, 
                       context: List[str], 
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7) -> Dict[str, any]:
        """
        Generate an answer using Gemini.
        
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
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 1000,
                top_p=0.9,
                top_k=40
            )
            
            # Generate response with retries
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    generation_time = time.time() - start_time
                    
                    if response.text:
                        return {
                            'answer': response.text.strip(),
                            'model': self.model_name,
                            'generation_time': generation_time,
                            'prompt_tokens': len(prompt.split()),  # Approximate
                            'completion_tokens': len(response.text.split()),  # Approximate
                            'total_tokens': len(prompt.split()) + len(response.text.split()),
                            'temperature': temperature,
                            'success': True
                        }
                    else:
                        return {
                            'answer': "I couldn't generate a response. Please try rephrasing your question.",
                            'model': self.model_name,
                            'generation_time': generation_time,
                            'success': False,
                            'error': "Empty response from model"
                        }
                        
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}, retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"All generation attempts failed: {str(e)}")
                        return {
                            'answer': f"I encountered an error while generating the response: {str(e)}",
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
        """Check if Gemini is available."""
        try:
            # Try a simple generation to test availability
            test_response = self.model.generate_content(
                "Test",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.1
                )
            )
            return bool(test_response.text)
        except Exception as e:
            self.logger.warning(f"Gemini availability check failed: {str(e)}")
            return False
    
    def create_prompt(self, query: str, context: List[str]) -> str:
        """Create a specialized prompt for Gemini."""
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context. Please follow these guidelines:

1. Answer the question based only on the information provided in the context
2. If the answer cannot be found in the context, clearly state that the information is not available
3. Be concise but comprehensive in your response
4. If relevant, cite which source document contains the information
5. If the question is ambiguous, ask for clarification

Context Information:
{context_text}

Question: {query}

Please provide a clear and helpful answer:"""
        
        return prompt
