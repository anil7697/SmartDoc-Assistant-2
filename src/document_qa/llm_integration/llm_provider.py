"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_answer(self, 
                       query: str, 
                       context: List[str], 
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7) -> Dict[str, any]:
        """
        Generate an answer based on query and context.
        
        Args:
            query: User question
            context: List of relevant text chunks
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the LLM model.
        
        Returns:
            Model name
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available.
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    def format_context(self, context_chunks: List[Dict[str, any]]) -> List[str]:
        """
        Format context chunks for the LLM.
        
        Args:
            context_chunks: List of chunk dictionaries
            
        Returns:
            List of formatted context strings
        """
        formatted_context = []
        
        for i, chunk in enumerate(context_chunks):
            content = chunk.get('content', '')
            source = chunk.get('source_document', 'Unknown')
            
            # Format with source information
            formatted = f"[Source {i+1}: {source}]\n{content}"
            formatted_context.append(formatted)
        
        return formatted_context
    
    def create_prompt(self, query: str, context: List[str]) -> str:
        """
        Create a prompt for the LLM.
        
        Args:
            query: User question
            context: List of context strings
            
        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join(context)
        
        prompt = f"""Based on the following context information, please answer the question. If the answer cannot be found in the context, please say so clearly.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
