"""FAISS-based vector store implementation."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from config import settings
from .vector_store import VectorStore


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with persistent storage."""
    
    def __init__(self, 
                 dimension: int,
                 index_type: str = "flat",
                 metric: str = "cosine"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2", "ip")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.logger = logger.bind(component="FAISSVectorStore")
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Store metadata separately
        self.metadata: List[Dict[str, any]] = []
        
        self.logger.info(f"Initialized FAISS vector store with dimension {dimension}, type {index_type}, metric {metric}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.metric == "cosine":
            # For cosine similarity, we'll use inner product with normalized vectors
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                # IVF with 100 clusters (adjust based on data size)
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(1, self.dimension // 10)))
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        elif self.metric == "l2":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(1, self.dimension // 10)))
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        elif self.metric == "ip":  # Inner product
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, max(1, self.dimension // 10)))
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return index
    
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, any]]) -> None:
        """
        Add vectors with metadata to the store.
        
        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries corresponding to each vector
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if not vectors:
            return
        
        # Convert to numpy array
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            # Normalize vectors to unit length
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vectors_array = vectors_array / norms
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if vectors_array.shape[0] >= self.index.nlist:
                self.logger.info("Training IVF index...")
                self.index.train(vectors_array)
            else:
                self.logger.warning(f"Not enough vectors to train IVF index (need {self.index.nlist}, have {vectors_array.shape[0]})")
        
        # Add vectors to index
        self.index.add(vectors_array)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        self.logger.info(f"Added {len(vectors)} vectors to the store. Total size: {self.get_size()}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (similarity_score, metadata)
        """
        if self.get_size() == 0:
            return []
        
        # Ensure query vector is the right shape and type
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # Normalize query vector for cosine similarity
        if self.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        # Perform search
        k = min(k, self.get_size())  # Don't search for more than available
        
        try:
            distances, indices = self.index.search(query_vector, k)
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []
        
        # Convert results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            # Convert distance to similarity score
            if self.metric == "cosine":
                # For cosine similarity with normalized vectors, distance is actually the inner product
                similarity = float(distance)
            elif self.metric == "l2":
                # Convert L2 distance to similarity (higher is better)
                similarity = 1.0 / (1.0 + float(distance))
            elif self.metric == "ip":
                # Inner product (higher is better)
                similarity = float(distance)
            else:
                similarity = float(distance)
            
            if idx < len(self.metadata):
                results.append((similarity, self.metadata[idx]))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the store
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'size': self.get_size(),
        }
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load the store from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store path does not exist: {path}")
        
        # Load configuration
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify configuration matches
        if config['dimension'] != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {config['dimension']}")
        
        # Load FAISS index
        index_path = path / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.logger.info(f"Loaded vector store from {path} with {self.get_size()} vectors")
    
    def get_size(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            Number of vectors
        """
        return self.index.ntotal
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.index = self._create_index()
        self.metadata = []
        self.logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            'size': self.get_size(),
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': getattr(self.index, 'is_trained', True),
            'memory_usage_mb': self.index.ntotal * self.dimension * 4 / (1024 * 1024),  # Approximate
        }
