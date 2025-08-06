"""Sentence Transformers embeddings implementation."""

import time
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import settings
from ..utils import chunk_list
from .embedding_generator import EmbeddingGenerator


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """Sentence Transformers embedding generator for local embeddings."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize Sentence Transformer embedding generator.
        
        Args:
            model_name: Sentence transformer model to use
            batch_size: Default batch size for processing
            device: Device to use ('cpu', 'cuda', etc.). Auto-detect if None
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.logger = logger.bind(component="SentenceTransformerEmbeddingGenerator")
        
        try:
            # Initialize the model
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Initialized Sentence Transformer with model: {model_name}")
            self.logger.info(f"Embedding dimension: {self.dimension}")
            self.logger.info(f"Device: {self.model.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sentence Transformer model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Single embedding array or list of embedding arrays
        """
        if isinstance(texts, str):
            return self._generate_single_embedding(texts)
        else:
            return self.generate_embeddings_batch(texts)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts with batching support.
        
        Args:
            texts: List of text strings
            batch_size: Size of each batch for processing
            
        Returns:
            List of embedding arrays
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        # Split texts into batches
        text_batches = chunk_list(texts, batch_size)
        
        self.logger.info(f"Processing {len(texts)} texts in {len(text_batches)} batches")
        
        for batch_idx, batch_texts in enumerate(tqdm(text_batches, desc="Generating embeddings")):
            try:
                batch_embeddings = self._generate_batch_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                raise
        
        self.logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            # Clean the text
            text = text.strip()
            if not text:
                # Return zero vector for empty text
                return np.zeros(self.dimension, dtype=np.float32)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating single embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                cleaned_text = text.strip()
                if not cleaned_text:
                    cleaned_text = " "  # Use space for empty texts
                cleaned_texts.append(cleaned_text)
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                batch_size=len(cleaned_texts),
                show_progress_bar=False
            )
            
            # Convert to list of arrays
            embedding_list = []
            for embedding in embeddings:
                embedding_list.append(embedding.astype(np.float32))
            
            return embedding_list
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this generator.
        
        Returns:
            Embedding dimension
        """
        return self.dimension
    
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name
        """
        return self.model_name
    
    def get_available_models(self) -> List[str]:
        """
        Get list of popular sentence transformer models.
        
        Returns:
            List of model names
        """
        return [
            "all-MiniLM-L6-v2",  # Fast and good quality
            "all-mpnet-base-v2",  # Best quality
            "all-MiniLM-L12-v2",  # Balance of speed and quality
            "paraphrase-MiniLM-L6-v2",  # Good for paraphrase detection
            "multi-qa-MiniLM-L6-cos-v1",  # Optimized for Q&A
            "msmarco-distilbert-base-v4",  # Good for search
        ]
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.dimension,
            'device': str(self.model.device),
            'batch_size': self.batch_size,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
        }
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
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
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
