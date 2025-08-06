"""Streamlit web application for Document Q&A system."""

import io
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our application
from src.document_qa.app import DocumentQAApp
from src.document_qa.utils import format_file_size

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


@st.cache_resource
def initialize_app():
    """Initialize the Document Q&A application."""
    try:
        return DocumentQAApp()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📚 Document Q&A System</h1>', unsafe_allow_html=True)
    
    # Initialize app
    app = initialize_app()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        st.subheader("API Keys")
        status = app.get_status()

        # Check Google API key
        if hasattr(app.answer_generator.providers.get('gemini'), 'api_key'):
            st.success("✅ Google API Key configured")
        else:
            st.error("❌ Google API Key missing")
            st.info("Set GOOGLE_API_KEY environment variable")
        
        # LLM Provider selection
        st.subheader("LLM Provider")
        available_providers = status['available_llm_providers']
        
        if available_providers:
            selected_provider = st.selectbox(
                "Select LLM Provider",
                available_providers,
                index=0
            )
        else:
            st.error("No LLM providers available")
            selected_provider = None
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
            top_k = st.slider("Number of Results", 1, 10, 5)
            temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1)
            
            if st.button("Update Settings"):
                app.update_settings(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                st.success("Settings updated!")
        
        # System status
        st.subheader("System Status")
        st.metric("Documents Loaded", "Yes" if status['documents_loaded'] else "No")
        st.metric("Vector Store Size", status['vector_store_size'])
        st.metric("Embedding Model", status['embedding_model'])
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "❓ Ask Questions", "📊 System Info"])
    
    with tab1:
        st.header("Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            
            total_size = 0
            for file in uploaded_files:
                file_size = len(file.getvalue())
                total_size += file_size
                st.write(f"- {file.name} ({format_file_size(file_size)})")
            
            st.write(f"Total size: {format_file_size(total_size)}")
            
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Prepare file data
                    files_data = []
                    for file in uploaded_files:
                        files_data.append({
                            'name': file.name,
                            'data': io.BytesIO(file.getvalue())
                        })
                    
                    # Process documents
                    results = app.load_documents_from_bytes(files_data)
                    
                    # Display results
                    if results['successful_loads'] > 0:
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ Successfully processed {results['successful_loads']} documents<br>
                            📊 Created {results['total_chunks']} text chunks<br>
                            📚 Documents: {', '.join(results['documents'])}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Auto-save vector store
                        app.save_vector_store()
                        st.info("Vector store saved automatically")
                    
                    if results['failed_loads'] > 0:
                        st.markdown(f"""
                        <div class="error-box">
                            ❌ Failed to process {results['failed_loads']} documents<br>
                            Errors: {'; '.join(results['errors'])}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Document management
        st.subheader("Document Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear All Documents"):
                app.clear_documents()
                st.success("All documents cleared!")
                st.rerun()
        
        with col2:
            if st.button("Save Vector Store"):
                app.save_vector_store()
                st.success("Vector store saved!")
    
    with tab2:
        st.header("Ask Questions")
        
        if not status['documents_loaded']:
            st.warning("Please upload and process documents first!")
        else:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about your documents?",
                help="Ask any question about the uploaded documents"
            )
            
            # Query suggestions
            if question and len(question) > 2:
                suggestions = app.search_engine.get_search_suggestions(question)
                if suggestions:
                    st.write("💡 Suggestions:")
                    for suggestion in suggestions[:3]:
                        if st.button(suggestion, key=f"suggestion_{suggestion}"):
                            question = suggestion
                            st.rerun()
            
            if question:
                if st.button("Get Answer", type="primary"):
                    with st.spinner("Searching documents and generating answer..."):
                        start_time = time.time()
                        
                        # Get answer
                        result = app.ask_question(
                            question=question,
                            top_k=top_k,
                            llm_provider=selected_provider,
                            temperature=temperature
                        )
                        
                        processing_time = time.time() - start_time
                        
                        # Display answer
                        if result.get('success', True):
                            st.subheader("📝 Answer")
                            st.write(result['answer'])
                            
                            # Display metadata
                            with st.expander("📊 Answer Details"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Processing Time", f"{processing_time:.2f}s")
                                    st.metric("LLM Provider", result.get('provider', 'Unknown'))
                                    st.metric("Context Chunks", result.get('context_chunks', 0))
                                
                                with col2:
                                    if 'generation_time' in result:
                                        st.metric("Generation Time", f"{result['generation_time']:.2f}s")
                                    if 'total_tokens' in result:
                                        st.metric("Total Tokens", result['total_tokens'])
                            
                            # Display sources
                            if 'sources' in result and result['sources']:
                                st.subheader("📚 Sources")
                                for i, source in enumerate(result['sources']):
                                    st.write(f"{i+1}. **{source['document']}** (Relevance: {source['similarity_score']:.2f})")
                            
                            # Display search results
                            if 'search_results' in result and result['search_results']['results']:
                                with st.expander("🔍 Search Results"):
                                    for i, chunk in enumerate(result['search_results']['results']):
                                        st.write(f"**Result {i+1}** (Score: {chunk.get('similarity_score', 0):.3f})")
                                        st.write(chunk.get('content', '')[:300] + "...")
                                        st.write(f"*Source: {chunk.get('source_document', 'Unknown')}*")
                                        st.divider()
                        
                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                ❌ {result['answer']}
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab3:
        st.header("System Information")
        
        # Application status
        status = app.get_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Vector Store Stats")
            vector_stats = status['vector_store_stats']
            st.metric("Total Vectors", vector_stats['size'])
            st.metric("Dimension", vector_stats['dimension'])
            st.metric("Index Type", vector_stats['index_type'])
            st.metric("Memory Usage", f"{vector_stats['memory_usage_mb']:.1f} MB")
        
        with col2:
            st.subheader("🤖 LLM Providers")
            provider_stats = app.answer_generator.get_provider_stats()
            
            for provider_name in provider_stats['available_providers']:
                provider_info = provider_stats['provider_details'][provider_name]
                status_icon = "✅" if provider_info['available'] else "❌"
                st.write(f"{status_icon} **{provider_name}**: {provider_info['model']}")
        
        # Test providers
        st.subheader("🧪 Test Providers")
        for provider in status['available_llm_providers']:
            if st.button(f"Test {provider}", key=f"test_{provider}"):
                with st.spinner(f"Testing {provider}..."):
                    test_result = app.answer_generator.test_provider(provider)
                    
                    if test_result['available']:
                        st.success(f"✅ {provider} is working correctly")
                        st.json(test_result)
                    else:
                        st.error(f"❌ {provider} test failed: {test_result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
