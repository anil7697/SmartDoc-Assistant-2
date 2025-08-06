"""Text chunking functionality for document processing."""

import re
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings


class TextChunker:
    """Split text into chunks for processing."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.default_chunk_overlap
        self.logger = logger.bind(component="TextChunker")
        
        # Default separators for recursive splitting
        self.separators = separators or [
            "\n\n",  # Double newlines (paragraphs)
            "\n",    # Single newlines
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Characters
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def chunk_text(self, text: str, document_metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text content to chunk
            document_metadata: Optional metadata about the source document
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for chunking")
            return []
        
        self.logger.info(f"Chunking text of length {len(text)} characters")
        
        try:
            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk_content in enumerate(chunks):
                chunk_obj = {
                    'content': chunk_content,
                    'chunk_id': i,
                    'chunk_size': len(chunk_content),
                    'word_count': len(chunk_content.split()),
                    'start_char': self._find_chunk_start_position(text, chunk_content, i),
                }
                
                # Add document metadata if provided
                if document_metadata:
                    chunk_obj.update({
                        'source_document': document_metadata.get('filename', 'unknown'),
                        'document_size': document_metadata.get('file_size', 0),
                        'document_extension': document_metadata.get('extension', ''),
                    })
                
                chunk_objects.append(chunk_obj)
            
            self.logger.info(f"Created {len(chunk_objects)} chunks")
            return chunk_objects
            
        except Exception as e:
            self.logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            if 'content' not in doc:
                self.logger.warning(f"Document missing content: {doc.get('filename', 'unknown')}")
                continue
            
            doc_chunks = self.chunk_text(doc['content'], doc)
            all_chunks.extend(doc_chunks)
        
        self.logger.info(f"Total chunks created from {len(documents)} documents: {len(all_chunks)}")
        return all_chunks
    
    def _find_chunk_start_position(self, full_text: str, chunk_content: str, chunk_index: int) -> int:
        """
        Find the approximate start position of a chunk in the full text.
        
        Args:
            full_text: The complete text
            chunk_content: The chunk content
            chunk_index: Index of the chunk
            
        Returns:
            Approximate character position where the chunk starts
        """
        try:
            # For the first chunk, start position is 0
            if chunk_index == 0:
                return 0
            
            # Try to find the chunk content in the full text
            # Take first few words to avoid issues with overlapping content
            chunk_words = chunk_content.split()[:5]
            if not chunk_words:
                return chunk_index * (self.chunk_size - self.chunk_overlap)
            
            search_phrase = ' '.join(chunk_words)
            position = full_text.find(search_phrase)
            
            if position != -1:
                return position
            
            # Fallback: estimate based on chunk size and overlap
            return chunk_index * (self.chunk_size - self.chunk_overlap)
            
        except Exception:
            # Fallback calculation
            return chunk_index * (self.chunk_size - self.chunk_overlap)
    
    def get_chunk_statistics(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'total_words': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
            }
        
        chunk_sizes = [chunk['chunk_size'] for chunk in chunks]
        total_chars = sum(chunk_sizes)
        total_words = sum(chunk.get('word_count', 0) for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_chunk_size': total_chars / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'configured_chunk_size': self.chunk_size,
            'configured_overlap': self.chunk_overlap,
        }
    
    def update_chunk_size(self, new_chunk_size: int, new_overlap: Optional[int] = None):
        """
        Update chunk size and overlap settings.
        
        Args:
            new_chunk_size: New chunk size
            new_overlap: New overlap size (optional)
        """
        self.chunk_size = new_chunk_size
        if new_overlap is not None:
            self.chunk_overlap = new_overlap
        
        # Recreate the text splitter with new settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        self.logger.info(f"Updated chunk size to {self.chunk_size}, overlap to {self.chunk_overlap}")
