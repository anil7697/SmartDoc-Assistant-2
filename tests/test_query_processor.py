"""Tests for query processing components."""

import pytest
from unittest.mock import Mock

from src.document_qa.query_processor import QueryProcessor, SemanticSearchEngine


class TestQueryProcessor:
    """Test cases for QueryProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = QueryProcessor()
    
    def test_process_valid_query(self):
        """Test processing a valid query."""
        query = "What is artificial intelligence?"
        result = self.processor.process_query(query)
        
        assert result['is_valid'] is True
        assert result['original_query'] == query
        assert 'processed_query' in result
        assert 'expanded_queries' in result
        assert 'query_type' in result
        assert 'keywords' in result
    
    def test_process_short_query(self):
        """Test processing a query that's too short."""
        query = "AI"
        result = self.processor.process_query(query)
        
        assert result['is_valid'] is False
        assert 'reason' in result
    
    def test_process_empty_query(self):
        """Test processing an empty query."""
        query = ""
        result = self.processor.process_query(query)
        
        assert result['is_valid'] is False
    
    def test_query_classification(self):
        """Test query type classification."""
        test_cases = [
            ("What is machine learning?", "definition"),
            ("How to train a model?", "procedural"),
            ("Why does this happen?", "explanatory"),
            ("When was this invented?", "factual"),
            ("Who created this?", "person"),
            ("Compare A and B", "comparison"),
            ("List all types", "list"),
            ("Random query", "general")
        ]
        
        for query, expected_type in test_cases:
            result = self.processor.process_query(query)
            assert result['query_type'] == expected_type
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        query = "What is machine learning and artificial intelligence?"
        result = self.processor.process_query(query)
        
        keywords = result['keywords']
        assert 'machine' in keywords
        assert 'learning' in keywords
        assert 'artificial' in keywords
        assert 'intelligence' in keywords
        # Stop words should be filtered out
        assert 'what' not in keywords
        assert 'is' not in keywords
        assert 'and' not in keywords
    
    def test_query_expansion(self):
        """Test query expansion."""
        processor = QueryProcessor(expand_queries=True)
        query = "machine learning"
        result = processor.process_query(query)
        
        expanded = result['expanded_queries']
        assert len(expanded) > 1
        assert query in expanded  # Original should be included
    
    def test_query_expansion_disabled(self):
        """Test with query expansion disabled."""
        processor = QueryProcessor(expand_queries=False)
        query = "machine learning"
        result = processor.process_query(query)
        
        expanded = result['expanded_queries']
        assert len(expanded) == 1
        assert expanded[0] == query.lower()  # Only processed version
    
    def test_optimize_for_retrieval(self):
        """Test query optimization for retrieval."""
        query = "What is machine learning?"
        optimized = self.processor.optimize_for_retrieval(query)
        
        assert isinstance(optimized, str)
        assert len(optimized) > 0
    
    def test_get_query_suggestions(self):
        """Test query suggestions."""
        partial_query = "what"
        suggestions = self.processor.get_query_suggestions(partial_query)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        
        # Should include common question starters
        suggestion_text = ' '.join(suggestions).lower()
        assert 'what' in suggestion_text
    
    def test_stop_word_removal(self):
        """Test stop word removal."""
        processor = QueryProcessor(remove_stop_words=True)
        query = "What is the best way to learn machine learning?"
        result = processor.process_query(query)
        
        processed = result['processed_query']
        # Common stop words should be removed
        assert 'the' not in processed
        assert 'is' not in processed
        # Important words should remain
        assert 'best' in processed
        assert 'learn' in processed
        assert 'machine' in processed


class TestSemanticSearchEngine:
    """Test cases for SemanticSearchEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_embedding_generator = Mock()
        self.mock_vector_store = Mock()
        self.mock_query_processor = Mock()
        
        # Set up mock returns
        self.mock_embedding_generator.generate_embeddings.return_value = [0.1, 0.2, 0.3]
        self.mock_embedding_generator.get_model_name.return_value = "test-model"
        
        self.mock_vector_store.search_chunks.return_value = [
            {
                'content': 'Test content 1',
                'similarity_score': 0.9,
                'source_document': 'doc1.txt'
            },
            {
                'content': 'Test content 2',
                'similarity_score': 0.8,
                'source_document': 'doc2.txt'
            }
        ]
        
        self.mock_query_processor.process_query.return_value = {
            'is_valid': True,
            'processed_query': 'test query',
            'expanded_queries': ['test query', 'test question'],
            'query_type': 'general',
            'keywords': ['test', 'query']
        }
        
        # Create search engine
        self.search_engine = SemanticSearchEngine(
            embedding_generator=self.mock_embedding_generator,
            vector_store=self.mock_vector_store,
            query_processor=self.mock_query_processor
        )
    
    def test_search_valid_query(self):
        """Test searching with a valid query."""
        query = "What is machine learning?"
        result = self.search_engine.search(query, top_k=2)
        
        assert 'query' in result
        assert 'results' in result
        assert 'total_results' in result
        assert 'search_time' in result
        assert result['query'] == query
        assert len(result['results']) == 2
        
        # Verify mocks were called
        self.mock_query_processor.process_query.assert_called_once_with(query)
        self.mock_embedding_generator.generate_embeddings.assert_called_once()
        self.mock_vector_store.search_chunks.assert_called_once()
    
    def test_search_invalid_query(self):
        """Test searching with an invalid query."""
        # Mock invalid query
        self.mock_query_processor.process_query.return_value = {
            'is_valid': False,
            'reason': 'Query too short'
        }
        
        query = "AI"
        result = self.search_engine.search(query)
        
        assert 'error' in result
        assert result['total_results'] == 0
        assert len(result['results']) == 0
    
    def test_search_with_minimum_similarity(self):
        """Test searching with minimum similarity threshold."""
        # Mock results with different similarity scores
        self.mock_vector_store.search_chunks.return_value = [
            {'content': 'High similarity', 'similarity_score': 0.9},
            {'content': 'Medium similarity', 'similarity_score': 0.6},
            {'content': 'Low similarity', 'similarity_score': 0.3}
        ]
        
        query = "test query"
        result = self.search_engine.search(query, min_similarity=0.7)
        
        # Should only return results above threshold
        assert len(result['results']) == 1
        assert result['results'][0]['similarity_score'] == 0.9
    
    def test_search_suggestions(self):
        """Test getting search suggestions."""
        partial_query = "machine"
        suggestions = self.search_engine.get_search_suggestions(partial_query)
        
        assert isinstance(suggestions, list)
        # Should delegate to query processor
        self.mock_query_processor.get_query_suggestions.assert_called_once_with(partial_query)
    
    def test_explain_search_results(self):
        """Test explaining search results."""
        # Mock search results
        search_results = {
            'results': [
                {'similarity_score': 0.9},
                {'similarity_score': 0.8}
            ],
            'processed_query': {'query_type': 'definition'}
        }
        
        explanation = self.search_engine.explain_search_results(search_results)
        
        assert 'explanation' in explanation
        assert 'query_type' in explanation
        assert 'average_similarity' in explanation
        assert 'suggestions' in explanation
    
    def test_explain_empty_results(self):
        """Test explaining empty search results."""
        search_results = {'results': []}
        explanation = self.search_engine.explain_search_results(search_results)
        
        assert 'No results found' in explanation['explanation']
        assert 'suggestions' in explanation
