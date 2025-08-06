"""Text chunking logic for document processing"""

import re
from typing import List, Dict, Any


class TextChunker:
    """Handles text chunking with configurable parameters"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, filename: str = "unknown") -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            filename: Name of the source file
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'id': chunk_id,
                    'content': chunk_text,
                    'filename': filename,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(chunk_text)
                }
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start <= chunks[-1]['start_pos'] if chunks else False:
                start = end
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        # Clean up multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within a range"""
        # Look for sentence endings
        sentence_endings = ['.', '!', '?']
        
        # Search backwards from end position
        for i in range(end - 1, start - 1, -1):
            if text[i] in sentence_endings:
                # Make sure it's not an abbreviation (simple check)
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # If no sentence boundary found, look for paragraph breaks
        for i in range(end - 1, start - 1, -1):
            if text[i] == '\n':
                return i + 1
        
        # If no good boundary found, return original end
        return end
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [chunk['length'] for chunk in chunks]
        total_chars = sum(chunk_sizes)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'avg_chunk_size': total_chars / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'configured_chunk_size': self.chunk_size,
            'configured_overlap': self.chunk_overlap
        }
    
    def update_settings(self, chunk_size: int = None, chunk_overlap: int = None):
        """Update chunking settings"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
