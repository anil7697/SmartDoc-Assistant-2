"""Streamlit Document Q&A Application"""

import io
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import backend modules
from backend.chunker import TextChunker
from backend.embedder import EmbeddingGenerator
from backend.vector_store import VectorStore
from backend.query_engine import QueryEngine

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class DocumentQAApp:
    """Main Document Q&A Application Class"""
    
    def __init__(self):
        """Initialize the application components"""
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.query_engine = QueryEngine(
            embedder=self.embedder,
            vector_store=self.vector_store
        )
        
        # Initialize session state
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'vector_store_size' not in st.session_state:
            st.session_state.vector_store_size = 0
    
    def load_documents(self, uploaded_files):
        """Load and process uploaded documents"""
        if not uploaded_files:
            return {"success": False, "message": "No files uploaded"}
        
        try:
            all_chunks = []
            processed_files = []
            
            for uploaded_file in uploaded_files:
                # Read file content
                file_content = uploaded_file.read()
                
                # Process based on file type
                if uploaded_file.name.endswith('.txt'):
                    text = file_content.decode('utf-8')
                elif uploaded_file.name.endswith('.pdf'):
                    # For now, treat as text (you can enhance this with PDF parsing)
                    text = file_content.decode('utf-8', errors='ignore')
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                # Chunk the text
                chunks = self.chunker.chunk_text(text, uploaded_file.name)
                all_chunks.extend(chunks)
                processed_files.append(uploaded_file.name)
            
            if all_chunks:
                # Generate embeddings and store in vector database
                self.vector_store.add_documents(all_chunks, self.embedder)
                
                # Update session state
                st.session_state.documents_loaded = True
                st.session_state.vector_store_size = len(all_chunks)
                
                return {
                    "success": True,
                    "message": f"Successfully processed {len(processed_files)} files with {len(all_chunks)} chunks",
                    "files": processed_files,
                    "chunks": len(all_chunks)
                }
            else:
                return {"success": False, "message": "No valid content found in uploaded files"}
                
        except Exception as e:
            return {"success": False, "message": f"Error processing documents: {str(e)}"}
    
    def ask_question(self, question, top_k=5):
        """Process a question and return an answer"""
        if not st.session_state.documents_loaded:
            return {
                "success": False,
                "answer": "Please upload and process documents first.",
                "sources": []
            }
        
        try:
            # Use query engine to get answer
            result = self.query_engine.query(question, top_k=top_k)
            return result
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }


@st.cache_resource
def initialize_app():
    """Initialize the Document Q&A application"""
    try:
        return DocumentQAApp()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()


def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">📚 Document Q&A System</h1>', unsafe_allow_html=True)
    
    # Initialize app
    app = initialize_app()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        st.subheader("API Keys")
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key:
            st.success("✅ Google API Key configured")
        else:
            st.error("❌ Google API Key missing")
            st.info("Set GOOGLE_API_KEY environment variable")
        
        # Settings
        st.subheader("Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        top_k = st.slider("Number of Results", 1, 10, 5)
        
        # System status
        st.subheader("System Status")
        st.metric("Documents Loaded", "Yes" if st.session_state.documents_loaded else "No")
        st.metric("Vector Store Size", st.session_state.vector_store_size)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "❓ Ask Questions", "📊 System Info"])
    
    with tab1:
        st.header("Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    result = app.load_documents(uploaded_files)
                    
                    if result["success"]:
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ {result['message']}<br>
                            📚 Files: {', '.join(result['files'])}<br>
                            📊 Chunks: {result['chunks']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            ❌ {result['message']}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Document management
        st.subheader("Document Management")
        if st.button("Clear All Documents"):
            app.vector_store.clear()
            st.session_state.documents_loaded = False
            st.session_state.vector_store_size = 0
            st.success("All documents cleared!")
            st.rerun()
    
    with tab2:
        st.header("Ask Questions")
        
        if not st.session_state.documents_loaded:
            st.warning("Please upload and process documents first!")
        else:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about your documents?",
                help="Ask any question about the uploaded documents"
            )
            
            if question:
                if st.button("Get Answer", type="primary"):
                    with st.spinner("Searching documents and generating answer..."):
                        start_time = time.time()
                        
                        # Get answer
                        result = app.ask_question(question, top_k=top_k)
                        processing_time = time.time() - start_time
                        
                        # Display answer
                        if result.get('success', True):
                            st.subheader("📝 Answer")
                            st.write(result['answer'])
                            
                            # Display metadata
                            with st.expander("📊 Answer Details"):
                                st.metric("Processing Time", f"{processing_time:.2f}s")
                                st.metric("Sources Used", len(result.get('sources', [])))
                            
                            # Display sources
                            if result.get('sources'):
                                st.subheader("📚 Sources")
                                for i, source in enumerate(result['sources']):
                                    st.write(f"{i+1}. **{source.get('filename', 'Unknown')}**")
                                    st.write(f"   {source.get('content', '')[:200]}...")
                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                ❌ {result['answer']}
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab3:
        st.header("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Application Stats")
            st.metric("Documents Loaded", "Yes" if st.session_state.documents_loaded else "No")
            st.metric("Vector Store Size", st.session_state.vector_store_size)
            st.metric("Embedding Model", app.embedder.model_name)
        
        with col2:
            st.subheader("🔧 Configuration")
            st.write(f"**Chunk Size**: {chunk_size}")
            st.write(f"**Chunk Overlap**: {chunk_overlap}")
            st.write(f"**Top K Results**: {top_k}")


if __name__ == "__main__":
    main()
