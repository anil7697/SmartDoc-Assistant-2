"""Answer generation orchestrator for document Q&A."""

from typing import Dict, List, Optional

from loguru import logger

from config import settings
from .gemini_provider import GeminiProvider
from .llm_provider import LLMProvider
from .ollama_provider import OllamaProvider


class AnswerGenerator:
    """Orchestrates answer generation using multiple LLM providers."""
    
    def __init__(self, default_provider: str = "gemini"):
        """
        Initialize answer generator.
        
        Args:
            default_provider: Default LLM provider to use
        """
        self.default_provider = default_provider
        self.providers: Dict[str, LLMProvider] = {}
        self.logger = logger.bind(component="AnswerGenerator")
        
        # Initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        # Initialize Gemini if API key is available
        try:
            google_api_key = "AIzaSyBYbzeGj9cE70YXax5-_FnRzskJyEeWYxA"
            if google_api_key:
                self.providers['gemini'] = GeminiProvider()
                self.logger.info("Initialized Gemini provider")
            else:
                self.logger.warning("Google API key not found - Gemini provider not available")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini provider: {str(e)}")

        # Initialize Ollama if available
        try:
            ollama_provider = OllamaProvider()
            if ollama_provider.is_available():
                self.providers['ollama'] = ollama_provider
                self.logger.info("Initialized Ollama provider")
            else:
                self.logger.warning("Ollama server not available")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama provider: {str(e)}")

        if not self.providers:
            self.logger.error("No LLM providers available! Please configure Google API key or start Ollama server.")
        else:
            self.logger.info(f"Available providers: {list(self.providers.keys())}")
    
    def generate_answer(self, 
                       query: str,
                       search_results: List[Dict[str, any]],
                       provider: Optional[str] = None,
                       max_context_length: int = 4000,
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7) -> Dict[str, any]:
        """
        Generate an answer based on query and search results.
        
        Args:
            query: User question
            search_results: List of relevant document chunks
            provider: LLM provider to use (uses default if None)
            max_context_length: Maximum context length in characters
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        # Select provider
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            available_providers = list(self.providers.keys())
            if available_providers:
                provider_name = available_providers[0]
                self.logger.warning(f"Requested provider '{provider}' not available, using '{provider_name}'")
            else:
                return {
                    'answer': "No LLM providers are currently available. Please check your configuration.",
                    'success': False,
                    'error': "No providers available"
                }
        
        llm_provider = self.providers[provider_name]
        
        try:
            # Prepare context from search results
            context = self._prepare_context(search_results, max_context_length)
            
            if not context:
                return {
                    'answer': "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if documents have been uploaded.",
                    'query': query,
                    'provider': provider_name,
                    'context_chunks': 0,
                    'success': False,
                    'error': "No context available"
                }
            
            # Generate answer
            result = llm_provider.generate_answer(
                query=query,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Add metadata
            result.update({
                'query': query,
                'provider': provider_name,
                'context_chunks': len(search_results),
                'context_length': sum(len(c) for c in context),
                'sources': self._extract_sources(search_results)
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': f"I encountered an error while generating the answer: {str(e)}",
                'query': query,
                'provider': provider_name,
                'success': False,
                'error': str(e)
            }
    
    def _prepare_context(self, search_results: List[Dict[str, any]], max_length: int) -> List[str]:
        """
        Prepare context from search results.
        
        Args:
            search_results: List of search result chunks
            max_length: Maximum total context length
            
        Returns:
            List of formatted context strings
        """
        if not search_results:
            return []
        
        context = []
        current_length = 0
        
        # Sort by similarity score (highest first)
        sorted_results = sorted(
            search_results, 
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )
        
        for i, result in enumerate(sorted_results):
            content = result.get('content', '')
            source = result.get('source_document', f'Document {i+1}')
            similarity = result.get('similarity_score', 0)
            
            # Format context with metadata
            formatted_context = f"[Source: {source} (Relevance: {similarity:.2f})]\n{content}"
            
            # Check if adding this context would exceed the limit
            if current_length + len(formatted_context) > max_length and context:
                break
            
            context.append(formatted_context)
            current_length += len(formatted_context)
        
        return context
    
    def _extract_sources(self, search_results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Extract source information from search results."""
        sources = []
        seen_sources = set()
        
        for result in search_results:
            source_doc = result.get('source_document', 'Unknown')
            if source_doc not in seen_sources:
                sources.append({
                    'document': source_doc,
                    'similarity_score': result.get('similarity_score', 0),
                    'chunk_id': result.get('chunk_id', 0)
                })
                seen_sources.add(source_doc)
        
        return sources
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return list(self.providers.keys())
    
    def set_default_provider(self, provider: str):
        """Set the default LLM provider."""
        if provider in self.providers:
            self.default_provider = provider
            self.logger.info(f"Set default provider to: {provider}")
        else:
            raise ValueError(f"Provider '{provider}' not available. Available: {list(self.providers.keys())}")
    
    def test_provider(self, provider: str) -> Dict[str, any]:
        """Test a specific provider."""
        if provider not in self.providers:
            return {
                'provider': provider,
                'available': False,
                'error': 'Provider not initialized'
            }
        
        try:
            llm_provider = self.providers[provider]
            is_available = llm_provider.is_available()
            
            if is_available:
                # Test with a simple query
                test_result = llm_provider.generate_answer(
                    query="What is AI?",
                    context=["Artificial Intelligence (AI) is a field of computer science."],
                    max_tokens=50,
                    temperature=0.1
                )
                
                return {
                    'provider': provider,
                    'available': True,
                    'model': llm_provider.get_model_name(),
                    'test_successful': test_result.get('success', False),
                    'test_response_length': len(test_result.get('answer', '')),
                }
            else:
                return {
                    'provider': provider,
                    'available': False,
                    'error': 'Provider availability check failed'
                }
                
        except Exception as e:
            return {
                'provider': provider,
                'available': False,
                'error': str(e)
            }
    
    def get_provider_stats(self) -> Dict[str, any]:
        """Get statistics about all providers."""
        stats = {
            'total_providers': len(self.providers),
            'available_providers': list(self.providers.keys()),
            'default_provider': self.default_provider,
            'provider_details': {}
        }
        
        for name, provider in self.providers.items():
            stats['provider_details'][name] = {
                'model': provider.get_model_name(),
                'available': provider.is_available()
            }
        
        return stats
