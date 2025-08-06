"""Embedding generation using Sentence Transformers"""

import numpy as np
from typing import List, Union
import streamlit as st


class EmbeddingGenerator:
    """Generates embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """Load the sentence transformer model (cached)"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(_self.model_name)
            return model
        except ImportError:
            st.error("sentence-transformers not installed. Please run: pip install sentence-transformers")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            st.stop()
    
    def _load_model(self):
        """Load the model"""
        self.model = self._load_model()
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Single embedding array or list of embedding arrays
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10
            )
            
            # Convert to float32 for memory efficiency
            embeddings = embeddings.astype(np.float32)
            
            if single_text:
                return embeddings[0]
            else:
                return list(embeddings)
                
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return np.zeros((384,), dtype=np.float32) if single_text else [np.zeros((384,), dtype=np.float32) for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if self.model is None:
            return 384  # Default for all-MiniLM-L6-v2
        
        try:
            return self.model.get_sentence_embedding_dimension()
        except:
            return 384
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'model_loaded': self.model is not None
        }
