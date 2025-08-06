"""Base class for vector stores."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, any]]) -> None:
        """
        Add vectors with metadata to the store.
        
        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries corresponding to each vector
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (similarity_score, metadata)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the store
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load the store from
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            Number of vectors
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pass
    
    def add_chunks(self, chunks: List[Dict[str, any]]) -> None:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        vectors = []
        metadata = []
        
        for chunk in chunks:
            if 'embedding' not in chunk:
                raise ValueError("Chunk missing embedding field")
            
            vectors.append(chunk['embedding'])
            
            # Create metadata without the embedding (to save space)
            chunk_metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
            metadata.append(chunk_metadata)
        
        self.add_vectors(vectors, metadata)
    
    def search_chunks(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar chunks and return full chunk information.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        results = self.search(query_vector, k)
        
        chunks_with_scores = []
        for score, metadata in results:
            chunk_with_score = metadata.copy()
            chunk_with_score['similarity_score'] = score
            chunks_with_scores.append(chunk_with_score)
        
        return chunks_with_scores
