"""OpenAI embeddings implementation."""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import openai
from loguru import logger
from tqdm import tqdm

from config import settings
from ..utils import chunk_list
from .embedding_generator import EmbeddingGenerator


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """OpenAI embeddings generator with rate limiting and error handling."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-3-small",
                 batch_size: int = 100,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize OpenAI embedding generator.
        
        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: OpenAI embedding model to use
            batch_size: Default batch size for processing
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger.bind(component="OpenAIEmbeddingGenerator")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model dimensions mapping
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        self.logger.info(f"Initialized OpenAI embedding generator with model: {self.model}")
    
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
                
                # Add small delay to avoid rate limiting
                if batch_idx < len(text_batches) - 1:  # Don't delay after the last batch
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                raise
        
        self.logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding
                
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries reached for rate limit error")
                    raise
                    
            except openai.APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"API error: {str(e)}, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries reached for API error: {str(e)}")
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error generating embedding: {str(e)}")
                raise
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                embeddings = []
                for data_point in response.data:
                    embedding = np.array(data_point.embedding, dtype=np.float32)
                    embeddings.append(embedding)
                
                return embeddings
                
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries reached for rate limit error")
                    raise
                    
            except openai.APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"API error: {str(e)}, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries reached for API error: {str(e)}")
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error generating batch embeddings: {str(e)}")
                raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this generator.
        
        Returns:
            Embedding dimension
        """
        return self.model_dimensions.get(self.model, 1536)  # Default to 1536 if unknown
    
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model.
        
        Returns:
            Model name
        """
        return self.model
    
    def estimate_cost(self, num_tokens: int) -> float:
        """
        Estimate the cost for generating embeddings.
        
        Args:
            num_tokens: Number of tokens to process
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (may need updates)
        pricing = {
            "text-embedding-3-small": 0.00002 / 1000,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013 / 1000,  # $0.00013 per 1K tokens
            "text-embedding-ada-002": 0.0001 / 1000,   # $0.0001 per 1K tokens
        }
        
        rate = pricing.get(self.model, 0.0001 / 1000)  # Default rate
        return num_tokens * rate
    
    def get_usage_stats(self) -> Dict[str, any]:
        """
        Get usage statistics for the embedding generator.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'model': self.model,
            'embedding_dimension': self.get_embedding_dimension(),
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
        }
