"""Configuration management for Document Q&A application."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    google_api_key: Optional[str] = Field(default="AIzaSyBYbzeGj9cE70YXax5-_FnRzskJyEeWYxA", env="GOOGLE_API_KEY")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    
    # Document Processing
    default_chunk_size: int = Field(default=1000, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=200, env="DEFAULT_CHUNK_OVERLAP")
    
    # Query Processing
    default_top_k: int = Field(default=5, env="DEFAULT_TOP_K")
    default_llm_provider: str = Field(default="gemini", env="DEFAULT_LLM_PROVIDER")
    
    # Storage Paths
    vector_store_path: str = Field(default="./data/vector_store", env="VECTOR_STORE_PATH")
    documents_path: str = Field(default="./data/documents", env="DOCUMENTS_PATH")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # Embedding Configuration (using sentence-transformers)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            Path(self.vector_store_path).parent,
            Path(self.documents_path),
            Path(self.log_file).parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
