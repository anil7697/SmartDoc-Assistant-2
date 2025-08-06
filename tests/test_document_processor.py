"""Tests for document processing components."""

import io
import pytest
from unittest.mock import Mock, patch

from src.document_qa.document_processor import DocumentLoader, TextChunker, TextPreprocessor


class TestDocumentLoader:
    """Test cases for DocumentLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
    
    def test_supported_extensions(self):
        """Test that supported extensions are correctly defined."""
        extensions = self.loader.get_supported_extensions()
        assert '.pdf' in extensions
        assert '.txt' in extensions
        assert '.docx' in extensions
    
    def test_load_text_from_bytes(self):
        """Test loading text from BytesIO."""
        text_content = "This is a test document."
        text_bytes = io.BytesIO(text_content.encode('utf-8'))
        
        result = self.loader.load_document(text_bytes, filename="test.txt")
        
        assert result['filename'] == "test.txt"
        assert result['content'] == text_content
        assert result['extension'] == '.txt'
        assert result['char_count'] == len(text_content)
    
    def test_unsupported_file_format(self):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.loader.load_document(io.BytesIO(b"test"), filename="test.xyz")
    
    def test_missing_filename_for_bytes(self):
        """Test that filename is required for BytesIO."""
        with pytest.raises(ValueError, match="Filename is required"):
            self.loader.load_document(io.BytesIO(b"test"))


class TestTextChunker:
    """Test cases for TextChunker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test document. " * 10  # Create longer text
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('chunk_id' in chunk for chunk in chunks)
        assert all('chunk_size' in chunk for chunk in chunks)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = self.chunker.chunk_text("")
        assert chunks == []
    
    def test_chunk_with_metadata(self):
        """Test chunking with document metadata."""
        text = "This is a test document. " * 10
        metadata = {'filename': 'test.txt', 'file_size': 1000}
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        assert chunks[0]['source_document'] == 'test.txt'
        assert chunks[0]['document_size'] == 1000
    
    def test_update_chunk_size(self):
        """Test updating chunk size."""
        original_size = self.chunker.chunk_size
        new_size = 200
        
        self.chunker.update_chunk_size(new_size)
        assert self.chunker.chunk_size == new_size
        assert self.chunker.chunk_size != original_size
    
    def test_get_chunk_statistics(self):
        """Test chunk statistics calculation."""
        text = "This is a test document. " * 10
        chunks = self.chunker.chunk_text(text)
        stats = self.chunker.get_chunk_statistics(chunks)
        
        assert 'total_chunks' in stats
        assert 'total_characters' in stats
        assert 'avg_chunk_size' in stats
        assert stats['total_chunks'] == len(chunks)


class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "  This is a TEST document.  \n\n  With extra spaces.  "
        processed = self.preprocessor.preprocess_text(text)
        
        assert processed.strip() == processed  # No leading/trailing whitespace
        assert "  " not in processed  # No double spaces
    
    def test_preprocess_empty_text(self):
        """Test preprocessing empty text."""
        assert self.preprocessor.preprocess_text("") == ""
        assert self.preprocessor.preprocess_text(None) == ""
    
    def test_preprocess_chunks(self):
        """Test preprocessing chunks."""
        chunks = [
            {'content': 'This is a test document.', 'chunk_id': 0},
            {'content': '  Another document with spaces.  ', 'chunk_id': 1},
            {'content': 'Short', 'chunk_id': 2}  # Too short, should be filtered
        ]
        
        processed = self.preprocessor.preprocess_chunks(chunks)
        
        # Should filter out the short chunk
        assert len(processed) == 2
        assert all('preprocessing_applied' in chunk for chunk in processed)
        assert all('original_content' in chunk for chunk in processed)
    
    def test_unicode_normalization(self):
        """Test unicode normalization."""
        text = "Café naïve résumé"  # Text with accented characters
        processed = self.preprocessor.preprocess_text(text)
        
        # Should not crash and should return valid text
        assert isinstance(processed, str)
        assert len(processed) > 0
    
    def test_get_preprocessing_stats(self):
        """Test preprocessing statistics."""
        original_chunks = [
            {'content': 'This is a test document.'},
            {'content': '  Another document.  '}
        ]
        processed_chunks = [
            {'content': 'This is a test document.'},
            {'content': 'Another document.'}
        ]
        
        stats = self.preprocessor.get_preprocessing_stats(original_chunks, processed_chunks)
        
        assert 'original_chunk_count' in stats
        assert 'processed_chunk_count' in stats
        assert 'settings' in stats
        assert stats['original_chunk_count'] == 2
        assert stats['processed_chunk_count'] == 2
