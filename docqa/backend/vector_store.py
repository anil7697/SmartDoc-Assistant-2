"""Vector database implementation using FAISS"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import streamlit as st


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.index = None
        self.documents = []  # Store document metadata
        self.embeddings = []  # Store embeddings for similarity calculation
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index"""
        try:
            import faiss
            # Use cosine similarity (inner product with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            st.error("FAISS not installed. Please run: pip install faiss-cpu")
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize FAISS index: {str(e)}")
            st.stop()
    
    def add_documents(self, chunks: List[Dict[str, Any]], embedder):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks with metadata
            embedder: Embedding generator instance
        """
        if not chunks:
            return
        
        try:
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            with st.spinner(f"Generating embeddings for {len(texts)} chunks..."):
                embeddings = embedder.generate_embeddings(texts)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store documents and embeddings
            self.documents.extend(chunks)
            self.embeddings.extend(embeddings_array)
            
            st.success(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (similarity_score, document_metadata)
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32)
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            top_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    results.append((float(similarity), self.documents[idx]))
            
            return results
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def clear(self):
        """Clear all documents from the vector store"""
        self._initialize_index()
        self.documents = []
        self.embeddings = []
    
    def get_size(self) -> int:
        """Get the number of documents in the store"""
        return len(self.documents)
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        try:
            import faiss
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings,
                    'dimension': self.dimension
                }, f)
            
            st.success(f"Vector store saved to {filepath}")
            
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
    
    def load(self, filepath: str):
        """Load the vector store from disk"""
        try:
            import faiss
            
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.dimension = data['dimension']
            
            st.success(f"Vector store loaded from {filepath}")
            
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'index_size': self.index.ntotal if self.index else 0
        }
