"""Tests for vector store components."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.document_qa.vector_store import FAISSVectorStore


class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dimension = 128
        self.store = FAISSVectorStore(dimension=self.dimension)
    
    def test_initialization(self):
        """Test vector store initialization."""
        assert self.store.dimension == self.dimension
        assert self.store.get_size() == 0
        assert self.store.index is not None
    
    def test_add_vectors(self):
        """Test adding vectors to the store."""
        # Create test vectors
        vectors = [np.random.rand(self.dimension).astype(np.float32) for _ in range(5)]
        metadata = [{'id': i, 'content': f'Document {i}'} for i in range(5)]
        
        self.store.add_vectors(vectors, metadata)
        
        assert self.store.get_size() == 5
        assert len(self.store.metadata) == 5
    
    def test_search_vectors(self):
        """Test searching for similar vectors."""
        # Add some vectors
        vectors = [np.random.rand(self.dimension).astype(np.float32) for _ in range(10)]
        metadata = [{'id': i, 'content': f'Document {i}'} for i in range(10)]
        
        self.store.add_vectors(vectors, metadata)
        
        # Search with one of the added vectors
        query_vector = vectors[0]
        results = self.store.search(query_vector, k=3)
        
        assert len(results) == 3
        assert all(len(result) == 2 for result in results)  # (score, metadata) tuples
        
        # First result should be the exact match (highest similarity)
        best_score, best_metadata = results[0]
        assert best_metadata['id'] == 0
    
    def test_search_empty_store(self):
        """Test searching in empty store."""
        query_vector = np.random.rand(self.dimension).astype(np.float32)
        results = self.store.search(query_vector, k=5)
        
        assert results == []
    
    def test_add_chunks(self):
        """Test adding chunks with embeddings."""
        chunks = []
        for i in range(3):
            chunk = {
                'content': f'This is document {i}',
                'chunk_id': i,
                'embedding': np.random.rand(self.dimension).astype(np.float32)
            }
            chunks.append(chunk)
        
        self.store.add_chunks(chunks)
        
        assert self.store.get_size() == 3
        
        # Metadata should not contain embeddings
        for metadata in self.store.metadata:
            assert 'embedding' not in metadata
            assert 'content' in metadata
    
    def test_search_chunks(self):
        """Test searching chunks."""
        # Add chunks
        chunks = []
        for i in range(5):
            chunk = {
                'content': f'This is document {i}',
                'chunk_id': i,
                'embedding': np.random.rand(self.dimension).astype(np.float32)
            }
            chunks.append(chunk)
        
        self.store.add_chunks(chunks)
        
        # Search
        query_vector = chunks[0]['embedding']
        results = self.store.search_chunks(query_vector, k=3)
        
        assert len(results) == 3
        assert all('similarity_score' in result for result in results)
        assert all('content' in result for result in results)
    
    def test_save_and_load(self):
        """Test saving and loading vector store."""
        # Add some data
        vectors = [np.random.rand(self.dimension).astype(np.float32) for _ in range(5)]
        metadata = [{'id': i, 'content': f'Document {i}'} for i in range(5)]
        
        self.store.add_vectors(vectors, metadata)
        original_size = self.store.get_size()
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_store"
            self.store.save(str(save_path))
            
            # Create new store and load
            new_store = FAISSVectorStore(dimension=self.dimension)
            new_store.load(str(save_path))
            
            assert new_store.get_size() == original_size
            assert len(new_store.metadata) == len(self.store.metadata)
    
    def test_clear(self):
        """Test clearing the vector store."""
        # Add some data
        vectors = [np.random.rand(self.dimension).astype(np.float32) for _ in range(3)]
        metadata = [{'id': i} for i in range(3)]
        
        self.store.add_vectors(vectors, metadata)
        assert self.store.get_size() == 3
        
        # Clear
        self.store.clear()
        assert self.store.get_size() == 0
        assert len(self.store.metadata) == 0
    
    def test_get_stats(self):
        """Test getting vector store statistics."""
        stats = self.store.get_stats()
        
        assert 'size' in stats
        assert 'dimension' in stats
        assert 'index_type' in stats
        assert 'metric' in stats
        assert stats['dimension'] == self.dimension
    
    def test_invalid_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        wrong_dimension_vector = np.random.rand(64).astype(np.float32)  # Wrong dimension
        
        with pytest.raises(Exception):  # FAISS will raise an exception
            self.store.add_vectors([wrong_dimension_vector], [{'id': 0}])
    
    def test_mismatched_vectors_metadata_length(self):
        """Test error handling for mismatched vectors and metadata lengths."""
        vectors = [np.random.rand(self.dimension).astype(np.float32) for _ in range(3)]
        metadata = [{'id': i} for i in range(2)]  # One less metadata entry
        
        with pytest.raises(ValueError, match="Number of vectors must match"):
            self.store.add_vectors(vectors, metadata)
