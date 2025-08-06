"""Base class for embedding generators."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""
    
    @abstractmethod
    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Single embedding array or list of embedding arrays
        """
        pass
    
    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts with batching support.
        
        Args:
            texts: List of text strings
            batch_size: Size of each batch for processing
            
        Returns:
            List of embedding arrays
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this generator.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name
        """
        pass
    
    def embed_chunks(self, chunks: List[Dict[str, any]], batch_size: Optional[int] = None) -> List[Dict[str, any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' field
            batch_size: Size of each batch for processing
            
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []
        
        # Extract text content from chunks
        texts = [chunk.get('content', '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, batch_size)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embedding
            embedded_chunk['embedding_model'] = self.get_model_name()
            embedded_chunk['embedding_dimension'] = len(embedding)
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
