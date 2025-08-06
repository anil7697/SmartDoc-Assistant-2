"""Semantic search engine combining query processing, embeddings, and vector search."""

from typing import Dict, List, Optional

from loguru import logger

from ..embeddings.embedding_generator import EmbeddingGenerator
from ..vector_store.vector_store import VectorStore
from .query_processor import QueryProcessor


class SemanticSearchEngine:
    """Semantic search engine for document Q&A."""
    
    def __init__(self,
                 embedding_generator: EmbeddingGenerator,
                 vector_store: VectorStore,
                 query_processor: Optional[QueryProcessor] = None):
        """
        Initialize semantic search engine.
        
        Args:
            embedding_generator: Embedding generator for queries
            vector_store: Vector store for similarity search
            query_processor: Optional query processor for optimization
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.query_processor = query_processor or QueryProcessor()
        self.logger = logger.bind(component="SemanticSearchEngine")
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               min_similarity: float = 0.0,
               rerank: bool = True) -> Dict[str, any]:
        """
        Perform semantic search for a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            rerank: Whether to rerank results
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(f"Searching for query: '{query}' (top_k={top_k})")
        
        # Process the query
        processed_query = self.query_processor.process_query(query)
        
        if not processed_query['is_valid']:
            return {
                'query': query,
                'processed_query': processed_query,
                'results': [],
                'total_results': 0,
                'search_time': 0.0,
                'error': processed_query.get('reason', 'Invalid query')
            }
        
        try:
            import time
            start_time = time.time()
            
            # Generate embedding for the main processed query
            main_query = processed_query['processed_query']
            query_embedding = self.embedding_generator.generate_embeddings(main_query)
            
            # Search vector store
            search_results = self.vector_store.search_chunks(query_embedding, top_k * 2)  # Get more for reranking
            
            # Filter by minimum similarity
            filtered_results = [
                result for result in search_results 
                if result.get('similarity_score', 0) >= min_similarity
            ]
            
            # Rerank if enabled and we have expanded queries
            if rerank and len(processed_query['expanded_queries']) > 1:
                filtered_results = self._rerank_results(
                    filtered_results, 
                    processed_query['expanded_queries'],
                    top_k
                )
            
            # Limit to top_k
            final_results = filtered_results[:top_k]
            
            search_time = time.time() - start_time
            
            # Add additional metadata to results
            for result in final_results:
                result['query_type'] = processed_query['query_type']
                result['matched_keywords'] = self._find_matching_keywords(
                    result.get('content', ''), 
                    processed_query['keywords']
                )
            
            self.logger.info(f"Found {len(final_results)} results in {search_time:.3f}s")
            
            return {
                'query': query,
                'processed_query': processed_query,
                'results': final_results,
                'total_results': len(filtered_results),
                'search_time': search_time,
                'embedding_model': self.embedding_generator.get_model_name(),
            }
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return {
                'query': query,
                'processed_query': processed_query,
                'results': [],
                'total_results': 0,
                'search_time': 0.0,
                'error': str(e)
            }
    
    def _rerank_results(self, 
                       results: List[Dict], 
                       expanded_queries: List[str], 
                       top_k: int) -> List[Dict]:
        """
        Rerank results using multiple query variations.
        
        Args:
            results: Initial search results
            expanded_queries: List of query variations
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if len(expanded_queries) <= 1:
            return results
        
        try:
            # Generate embeddings for all expanded queries
            expanded_embeddings = self.embedding_generator.generate_embeddings_batch(expanded_queries[1:])
            
            # Calculate additional similarity scores
            for result in results:
                additional_scores = []
                
                # Get the original embedding (we'd need to store this or regenerate)
                # For now, we'll use a simpler text-based reranking
                content = result.get('content', '').lower()
                
                for query in expanded_queries[1:]:
                    # Simple keyword matching score
                    query_words = set(query.lower().split())
                    content_words = set(content.split())
                    overlap = len(query_words.intersection(content_words))
                    score = overlap / len(query_words) if query_words else 0
                    additional_scores.append(score)
                
                # Combine scores (weighted average)
                original_score = result.get('similarity_score', 0)
                avg_additional_score = sum(additional_scores) / len(additional_scores) if additional_scores else 0
                
                # Weight: 70% original, 30% additional
                result['reranked_score'] = 0.7 * original_score + 0.3 * avg_additional_score
                result['additional_scores'] = additional_scores
            
            # Sort by reranked score
            results.sort(key=lambda x: x.get('reranked_score', x.get('similarity_score', 0)), reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error during reranking: {str(e)}, using original ranking")
        
        return results
    
    def _find_matching_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """Find which keywords appear in the content."""
        content_lower = content.lower()
        matching = []
        
        for keyword in keywords:
            if keyword.lower() in content_lower:
                matching.append(keyword)
        
        return matching
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions for a partial query.
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested queries
        """
        return self.query_processor.get_query_suggestions(partial_query)
    
    def get_related_chunks(self, chunk_id: int, top_k: int = 3) -> List[Dict]:
        """
        Find chunks related to a specific chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of related chunks to return
            
        Returns:
            List of related chunks
        """
        # This would require storing chunk embeddings or regenerating them
        # For now, return empty list
        self.logger.warning("get_related_chunks not fully implemented")
        return []
    
    def explain_search_results(self, search_results: Dict) -> Dict[str, any]:
        """
        Provide explanation for search results.
        
        Args:
            search_results: Results from search method
            
        Returns:
            Dictionary with explanations
        """
        if not search_results.get('results'):
            return {
                'explanation': 'No results found for the query.',
                'suggestions': [
                    'Try using different keywords',
                    'Make your query more specific',
                    'Check for spelling errors'
                ]
            }
        
        results = search_results['results']
        query_type = search_results['processed_query'].get('query_type', 'general')
        
        explanation = f"Found {len(results)} relevant documents for your {query_type} query. "
        
        # Analyze result quality
        avg_similarity = sum(r.get('similarity_score', 0) for r in results) / len(results)
        
        if avg_similarity > 0.8:
            explanation += "The results are highly relevant to your query."
        elif avg_similarity > 0.6:
            explanation += "The results are moderately relevant to your query."
        else:
            explanation += "The results have lower relevance. Consider refining your query."
        
        # Provide suggestions based on query type
        suggestions = []
        if query_type == 'definition':
            suggestions.append("Look for results that contain definitions or explanations")
        elif query_type == 'procedural':
            suggestions.append("Focus on step-by-step instructions or procedures")
        elif query_type == 'comparison':
            suggestions.append("Look for results that discuss differences or similarities")
        
        return {
            'explanation': explanation,
            'query_type': query_type,
            'average_similarity': avg_similarity,
            'suggestions': suggestions
        }
