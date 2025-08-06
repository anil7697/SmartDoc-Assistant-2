"""Main application orchestrator for Document Q&A system."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from config import settings
from .document_processor import DocumentLoader, TextChunker, TextPreprocessor
from .embeddings import SentenceTransformerEmbeddingGenerator
from .llm_integration import AnswerGenerator
from .query_processor import SemanticSearchEngine, QueryProcessor
from .vector_store import FAISSVectorStore


class DocumentQAApp:
    """Main application class for Document Q&A system."""
    
    def __init__(self):
        """Initialize the Document Q&A application."""
        self.logger = logger.bind(component="DocumentQAApp")
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_chunker = TextChunker()
        self.text_preprocessor = TextPreprocessor()
        
        # Initialize embedding generator
        try:
            self.embedding_generator = SentenceTransformerEmbeddingGenerator(
                model_name=settings.embedding_model
            )
            embedding_dim = self.embedding_generator.get_embedding_dimension()
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding generator: {str(e)}")
            raise
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore(dimension=embedding_dim)
        
        # Initialize query processor and search engine
        self.query_processor = QueryProcessor()
        self.search_engine = SemanticSearchEngine(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            query_processor=self.query_processor
        )
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator()
        
        # State tracking
        self.documents_loaded = False
        self.vector_store_path = Path(settings.vector_store_path)
        
        self.logger.info("Document Q&A application initialized successfully")
    
    def load_documents(self, file_paths: List[Union[str, Path]]) -> Dict[str, any]:
        """
        Load and process documents.
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Loading {len(file_paths)} documents")
        
        results = {
            'total_files': len(file_paths),
            'successful_loads': 0,
            'failed_loads': 0,
            'total_chunks': 0,
            'documents': [],
            'errors': []
        }
        
        try:
            # Load documents
            documents = []
            for file_path in file_paths:
                try:
                    doc = self.document_loader.load_document(file_path)
                    documents.append(doc)
                    results['successful_loads'] += 1
                    self.logger.info(f"Loaded: {doc['filename']}")
                except Exception as e:
                    error_msg = f"Failed to load {file_path}: {str(e)}"
                    results['errors'].append(error_msg)
                    results['failed_loads'] += 1
                    self.logger.error(error_msg)
            
            if not documents:
                return results
            
            # Chunk documents
            chunks = self.text_chunker.chunk_documents(documents)
            
            # Preprocess chunks
            processed_chunks = self.text_preprocessor.preprocess_chunks(chunks)
            
            # Generate embeddings
            embedded_chunks = self.embedding_generator.embed_chunks(processed_chunks)
            
            # Add to vector store
            self.vector_store.add_chunks(embedded_chunks)
            
            results['total_chunks'] = len(embedded_chunks)
            results['documents'] = [doc['filename'] for doc in documents]
            
            self.documents_loaded = True
            self.logger.info(f"Successfully processed {len(documents)} documents into {len(embedded_chunks)} chunks")
            
        except Exception as e:
            error_msg = f"Error during document processing: {str(e)}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return results
    
    def load_documents_from_bytes(self, files_data: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Load documents from bytes data (for Streamlit file uploads).
        
        Args:
            files_data: List of dictionaries with 'name' and 'data' keys
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Loading {len(files_data)} documents from bytes")
        
        results = {
            'total_files': len(files_data),
            'successful_loads': 0,
            'failed_loads': 0,
            'total_chunks': 0,
            'documents': [],
            'errors': []
        }
        
        try:
            documents = []
            for file_data in files_data:
                try:
                    doc = self.document_loader.load_document(
                        file_data['data'], 
                        filename=file_data['name']
                    )
                    documents.append(doc)
                    results['successful_loads'] += 1
                    self.logger.info(f"Loaded: {doc['filename']}")
                except Exception as e:
                    error_msg = f"Failed to load {file_data['name']}: {str(e)}"
                    results['errors'].append(error_msg)
                    results['failed_loads'] += 1
                    self.logger.error(error_msg)
            
            if not documents:
                return results
            
            # Process documents (same as load_documents)
            chunks = self.text_chunker.chunk_documents(documents)
            processed_chunks = self.text_preprocessor.preprocess_chunks(chunks)
            embedded_chunks = self.embedding_generator.embed_chunks(processed_chunks)
            self.vector_store.add_chunks(embedded_chunks)
            
            results['total_chunks'] = len(embedded_chunks)
            results['documents'] = [doc['filename'] for doc in documents]
            
            self.documents_loaded = True
            self.logger.info(f"Successfully processed {len(documents)} documents into {len(embedded_chunks)} chunks")
            
        except Exception as e:
            error_msg = f"Error during document processing: {str(e)}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return results
    
    def ask_question(self, 
                    question: str,
                    top_k: int = 5,
                    llm_provider: Optional[str] = None,
                    temperature: float = 0.7) -> Dict[str, any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            llm_provider: LLM provider to use
            temperature: LLM temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.documents_loaded:
            return {
                'answer': "No documents have been loaded yet. Please upload some documents first.",
                'success': False,
                'error': 'No documents loaded'
            }
        
        try:
            # Search for relevant chunks
            search_results = self.search_engine.search(question, top_k=top_k)
            
            if not search_results['results']:
                return {
                    'answer': "I couldn't find any relevant information to answer your question. Please try rephrasing your query.",
                    'question': question,
                    'search_results': search_results,
                    'success': False,
                    'error': 'No relevant results found'
                }
            
            # Generate answer
            answer_result = self.answer_generator.generate_answer(
                query=question,
                search_results=search_results['results'],
                provider=llm_provider,
                temperature=temperature
            )
            
            # Combine results
            answer_result['search_results'] = search_results
            
            return answer_result
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.logger.error(error_msg)
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'question': question,
                'success': False,
                'error': str(e)
            }
    
    def save_vector_store(self, path: Optional[str] = None):
        """Save the vector store to disk."""
        save_path = path or self.vector_store_path
        self.vector_store.save(save_path)
        self.logger.info(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, path: Optional[str] = None) -> bool:
        """Load the vector store from disk."""
        load_path = path or self.vector_store_path
        try:
            self.vector_store.load(load_path)
            self.documents_loaded = True
            self.logger.info(f"Vector store loaded from {load_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def clear_documents(self):
        """Clear all loaded documents."""
        self.vector_store.clear()
        self.documents_loaded = False
        self.logger.info("Cleared all documents")
    
    def get_status(self) -> Dict[str, any]:
        """Get application status."""
        return {
            'documents_loaded': self.documents_loaded,
            'vector_store_size': self.vector_store.get_size(),
            'available_llm_providers': self.answer_generator.get_available_providers(),
            'embedding_model': self.embedding_generator.get_model_name(),
            'vector_store_stats': self.vector_store.get_stats(),
        }
    
    def update_settings(self, **kwargs):
        """Update application settings."""
        if 'chunk_size' in kwargs or 'chunk_overlap' in kwargs:
            self.text_chunker.update_chunk_size(
                kwargs.get('chunk_size', self.text_chunker.chunk_size),
                kwargs.get('chunk_overlap', self.text_chunker.chunk_overlap)
            )
        
        if 'default_llm_provider' in kwargs:
            self.answer_generator.set_default_provider(kwargs['default_llm_provider'])
        
        self.logger.info(f"Updated settings: {kwargs}")
