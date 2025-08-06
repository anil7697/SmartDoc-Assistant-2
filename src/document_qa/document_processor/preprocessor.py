"""Text preprocessing utilities for document processing."""

import re
import unicodedata
from typing import Dict, List, Optional

from loguru import logger


class TextPreprocessor:
    """Preprocess text content for better embedding and retrieval."""
    
    def __init__(self, 
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_special_chars: bool = False,
                 min_chunk_length: int = 50):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_extra_whitespace: Remove extra whitespace and normalize spacing
            normalize_unicode: Normalize unicode characters
            remove_special_chars: Remove special characters (be careful with this)
            min_chunk_length: Minimum length for chunks to be considered valid
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.min_chunk_length = min_chunk_length
        self.logger = logger.bind(component="TextPreprocessor")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        processed_text = text
        
        # Normalize unicode characters
        if self.normalize_unicode:
            processed_text = self._normalize_unicode(processed_text)
        
        # Remove or normalize special characters
        if self.remove_special_chars:
            processed_text = self._remove_special_chars(processed_text)
        else:
            processed_text = self._normalize_special_chars(processed_text)
        
        # Clean up whitespace
        if self.remove_extra_whitespace:
            processed_text = self._clean_whitespace(processed_text)
        
        # Remove empty lines and normalize line breaks
        processed_text = self._normalize_line_breaks(processed_text)
        
        return processed_text.strip()
    
    def preprocess_chunks(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Preprocess a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' field
            
        Returns:
            List of preprocessed chunks (may be filtered)
        """
        processed_chunks = []
        filtered_count = 0
        
        for chunk in chunks:
            if 'content' not in chunk:
                continue
            
            original_content = chunk['content']
            processed_content = self.preprocess_text(original_content)
            
            # Filter out chunks that are too short after preprocessing
            if len(processed_content) < self.min_chunk_length:
                filtered_count += 1
                continue
            
            # Update the chunk with processed content
            processed_chunk = chunk.copy()
            processed_chunk['content'] = processed_content
            processed_chunk['original_content'] = original_content
            processed_chunk['preprocessing_applied'] = True
            processed_chunk['chunk_size'] = len(processed_content)
            processed_chunk['word_count'] = len(processed_content.split())
            
            processed_chunks.append(processed_chunk)
        
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} chunks that were too short after preprocessing")
        
        self.logger.info(f"Preprocessed {len(processed_chunks)} chunks")
        return processed_chunks
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        return unicodedata.normalize('NFC', text)
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and basic punctuation."""
        # Keep letters, numbers, and basic punctuation
        pattern = r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]'
        return re.sub(pattern, ' ', text)
    
    def _normalize_special_chars(self, text: str) -> str:
        """Normalize special characters without removing them."""
        # Replace various dash types with standard hyphen
        text = re.sub(r'[–—―]', '-', text)
        
        # Replace various quote types with standard quotes
        text = re.sub(r'[""''`´]', '"', text)
        
        # Replace various apostrophe types
        text = re.sub(r'[''`´]', "'", text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        # Replace multiple consecutive punctuation marks
        text = re.sub(r'([!?]){2,}', r'\1', text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace issues."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove spaces at the beginning and end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        return text
    
    def _normalize_line_breaks(self, text: str) -> str:
        """Normalize line breaks and remove empty lines."""
        # Split into lines and remove empty ones
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Join with single newlines
        return '\n'.join(non_empty_lines)
    
    def get_preprocessing_stats(self, original_chunks: List[Dict], processed_chunks: List[Dict]) -> Dict[str, any]:
        """
        Get statistics about the preprocessing operation.
        
        Args:
            original_chunks: Original chunks before preprocessing
            processed_chunks: Chunks after preprocessing
            
        Returns:
            Dictionary with preprocessing statistics
        """
        original_count = len(original_chunks)
        processed_count = len(processed_chunks)
        filtered_count = original_count - processed_count
        
        if original_chunks:
            original_total_chars = sum(len(chunk.get('content', '')) for chunk in original_chunks)
            original_avg_length = original_total_chars / original_count
        else:
            original_total_chars = 0
            original_avg_length = 0
        
        if processed_chunks:
            processed_total_chars = sum(len(chunk.get('content', '')) for chunk in processed_chunks)
            processed_avg_length = processed_total_chars / processed_count
        else:
            processed_total_chars = 0
            processed_avg_length = 0
        
        return {
            'original_chunk_count': original_count,
            'processed_chunk_count': processed_count,
            'filtered_chunk_count': filtered_count,
            'original_total_characters': original_total_chars,
            'processed_total_characters': processed_total_chars,
            'original_avg_length': original_avg_length,
            'processed_avg_length': processed_avg_length,
            'character_reduction_ratio': (original_total_chars - processed_total_chars) / original_total_chars if original_total_chars > 0 else 0,
            'settings': {
                'remove_extra_whitespace': self.remove_extra_whitespace,
                'normalize_unicode': self.normalize_unicode,
                'remove_special_chars': self.remove_special_chars,
                'min_chunk_length': self.min_chunk_length,
            }
        }
