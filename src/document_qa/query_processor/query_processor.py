"""Query preprocessing and optimization."""

import re
from typing import Dict, List, Optional

from loguru import logger


class QueryProcessor:
    """Process and optimize queries for better retrieval."""
    
    def __init__(self, 
                 expand_queries: bool = True,
                 remove_stop_words: bool = False,
                 min_query_length: int = 3):
        """
        Initialize query processor.
        
        Args:
            expand_queries: Whether to expand queries with synonyms/variations
            remove_stop_words: Whether to remove common stop words
            min_query_length: Minimum query length to process
        """
        self.expand_queries = expand_queries
        self.remove_stop_words = remove_stop_words
        self.min_query_length = min_query_length
        self.logger = logger.bind(component="QueryProcessor")
        
        # Common stop words (basic set)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
        }
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a query and return processed versions.
        
        Args:
            query: Raw query string
            
        Returns:
            Dictionary with processed query information
        """
        if not query or len(query.strip()) < self.min_query_length:
            return {
                'original_query': query,
                'processed_query': query,
                'expanded_queries': [query],
                'is_valid': False,
                'reason': 'Query too short or empty'
            }
        
        # Clean and normalize the query
        processed_query = self._clean_query(query)
        
        # Generate query variations
        expanded_queries = self._expand_query(processed_query) if self.expand_queries else [processed_query]
        
        return {
            'original_query': query,
            'processed_query': processed_query,
            'expanded_queries': expanded_queries,
            'is_valid': True,
            'query_type': self._classify_query(processed_query),
            'keywords': self._extract_keywords(processed_query),
        }
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep basic punctuation
        cleaned = re.sub(r'[^\w\s\?\!\.\,\-]', ' ', cleaned)
        
        # Remove stop words if enabled
        if self.remove_stop_words:
            words = cleaned.split()
            words = [word for word in words if word not in self.stop_words]
            cleaned = ' '.join(words)
        
        return cleaned.strip()
    
    def _expand_query(self, query: str) -> List[str]:
        """Generate expanded versions of the query."""
        expanded = [query]
        
        # Add question variations
        if not query.endswith('?'):
            expanded.append(query + '?')
        
        # Add "what is" variation for definition-like queries
        if not query.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            expanded.append(f"what is {query}")
            expanded.append(f"explain {query}")
        
        # Add "how to" variation for procedural queries
        if not query.startswith('how'):
            expanded.append(f"how to {query}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for q in expanded:
            if q not in seen:
                seen.add(q)
                unique_expanded.append(q)
        
        return unique_expanded
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Question words
        if any(query_lower.startswith(word) for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            if query_lower.startswith('what'):
                return 'definition'
            elif query_lower.startswith('how'):
                return 'procedural'
            elif query_lower.startswith('why'):
                return 'explanatory'
            elif query_lower.startswith(('when', 'where')):
                return 'factual'
            elif query_lower.startswith('who'):
                return 'person'
        
        # Command-like queries
        if any(query_lower.startswith(word) for word in ['explain', 'describe', 'tell me', 'show me']):
            return 'explanatory'
        
        # Comparison queries
        if any(word in query_lower for word in ['vs', 'versus', 'compare', 'difference', 'better']):
            return 'comparison'
        
        # List queries
        if any(word in query_lower for word in ['list', 'types of', 'kinds of', 'examples']):
            return 'list'
        
        return 'general'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Simple keyword extraction
        words = query.split()
        
        # Remove very short words and stop words
        keywords = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                # Remove punctuation
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    keywords.append(clean_word)
        
        return keywords
    
    def optimize_for_retrieval(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Optimize query specifically for retrieval.
        
        Args:
            query: Original query
            context: Optional context information
            
        Returns:
            Optimized query for retrieval
        """
        processed = self.process_query(query)
        
        if not processed['is_valid']:
            return query
        
        optimized_query = processed['processed_query']
        
        # Add context if available
        if context:
            if 'domain' in context:
                optimized_query = f"{context['domain']} {optimized_query}"
            
            if 'previous_queries' in context:
                # Could implement query history optimization here
                pass
        
        return optimized_query
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Generate query suggestions based on partial input.
        
        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        
        # Common question starters
        question_starters = [
            "What is",
            "How to",
            "Why does",
            "When did",
            "Where is",
            "Who was",
            "Explain",
            "Describe",
            "Compare"
        ]
        
        for starter in question_starters:
            if starter.lower().startswith(partial_query.lower()):
                suggestions.append(f"{starter} ")
            elif partial_query.lower() not in starter.lower():
                suggestions.append(f"{starter} {partial_query}")
        
        # Limit suggestions
        return suggestions[:max_suggestions]
