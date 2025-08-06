"""Query processing and answer generation logic"""

import os
from typing import Dict, Any, List
import streamlit as st


class QueryEngine:
    """Handles query processing and answer generation"""
    
    def __init__(self, embedder, vector_store):
        """
        Initialize the query engine
        
        Args:
            embedder: Embedding generator instance
            vector_store: Vector store instance
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_provider = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM provider (Gemini Flash 1.5)"""
        #google_api_key = os.getenv('GOOGLE_API_KEY')

        google_api_key = "AIzaSyBYbzeGj9cE70YXax5-_FnRzskJyEeWYxA"
        
        if not google_api_key:
            st.warning("Google API key not found. LLM functionality will be limited.")
            return None
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model
        except ImportError:
            st.error("google-generativeai not installed. Please run: pip install google-generativeai")
            return None
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {str(e)}")
            return None
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a query and return an answer
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            # Generate embedding for the question
            question_embedding = self.embedder.generate_embeddings(question)
            
            # Search for relevant documents
            search_results = self.vector_store.search(question_embedding, top_k)
            
            if not search_results:
                return {
                    "success": False,
                    "answer": "No relevant documents found for your question.",
                    "sources": []
                }
            
            # Extract relevant context
            context_chunks = []
            sources = []
            
            for similarity, doc in search_results:
                context_chunks.append(doc['content'])
                sources.append({
                    'filename': doc['filename'],
                    'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    'similarity': similarity
                })
            
            # Generate answer using LLM
            if self.llm_provider:
                answer = self._generate_llm_answer(question, context_chunks)
            else:
                # Fallback: return most relevant chunk
                answer = f"Based on the most relevant document chunk:\n\n{context_chunks[0][:500]}..."
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": len(context_chunks)
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }
    
    def _generate_llm_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate an answer using the LLM
        
        Args:
            question: User's question
            context_chunks: Relevant document chunks
            
        Returns:
            Generated answer
        """
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
            
            # Create prompt
            prompt = f"""Based on the following context information, please answer the question. 
If the answer cannot be found in the context, please say so clearly.

Context:
{context}

Question: {question}

Please provide a clear and helpful answer based on the context:"""
            
            # Generate response
            response = self.llm_provider.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested queries
        """
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        
        # Common question starters
        starters = [
            "What is",
            "How does",
            "Why is",
            "When was",
            "Where is",
            "Who was",
            "Explain",
            "Describe",
            "Compare"
        ]
        
        for starter in starters:
            if starter.lower().startswith(partial_query.lower()):
                suggestions.append(f"{starter} ")
            elif partial_query.lower() not in starter.lower():
                suggestions.append(f"{starter} {partial_query}")
        
        return suggestions[:5]  # Return top 5 suggestions
