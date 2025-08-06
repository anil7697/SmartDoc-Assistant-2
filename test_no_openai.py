#!/usr/bin/env python
"""Test script to verify the system works without OpenAI."""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_embeddings():
    """Test the sentence transformer embeddings."""
    print("🧪 Testing Sentence Transformer Embeddings...")
    
    try:
        from src.document_qa.embeddings import SentenceTransformerEmbeddingGenerator
        
        # Initialize embedding generator
        embedder = SentenceTransformerEmbeddingGenerator()
        print(f"✅ Initialized embedder: {embedder.get_model_name()}")
        print(f"📊 Embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Test single embedding
        test_text = "This is a test sentence for embedding generation."
        embedding = embedder.generate_embeddings(test_text)
        print(f"✅ Generated single embedding: shape {embedding.shape}")
        
        # Test batch embeddings
        test_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedder.generate_embeddings_batch(test_texts)
        print(f"✅ Generated batch embeddings: {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {str(e)}")
        return False


def test_llm_providers():
    """Test LLM providers (Gemini and Ollama)."""
    print("\n🤖 Testing LLM Providers...")
    
    try:
        from src.document_qa.llm_integration import AnswerGenerator
        
        # Initialize answer generator
        answer_gen = AnswerGenerator()
        providers = answer_gen.get_available_providers()
        print(f"✅ Available LLM providers: {providers}")
        
        if not providers:
            print("⚠️ No LLM providers available. Please configure Google API key or start Ollama.")
            return False
        
        # Test each provider
        for provider in providers:
            test_result = answer_gen.test_provider(provider)
            if test_result['available']:
                print(f"✅ {provider} provider is working")
            else:
                print(f"❌ {provider} provider failed: {test_result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM provider test failed: {str(e)}")
        return False


def test_document_processing():
    """Test document processing without OpenAI."""
    print("\n📄 Testing Document Processing...")
    
    try:
        from src.document_qa.app import DocumentQAApp
        
        # Create a test document
        temp_dir = Path(tempfile.mkdtemp())
        test_doc = temp_dir / "test.txt"
        test_content = """
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) is a field of computer science that aims to create 
        intelligent machines. Machine Learning (ML) is a subset of AI that enables 
        computers to learn from data without being explicitly programmed.
        
        Key concepts in ML include:
        - Supervised Learning: Learning with labeled data
        - Unsupervised Learning: Finding patterns in unlabeled data  
        - Deep Learning: Using neural networks with multiple layers
        """
        
        test_doc.write_text(test_content)
        print(f"✅ Created test document: {test_doc}")
        
        # Initialize app
        app = DocumentQAApp()
        print("✅ Initialized DocumentQAApp")
        
        # Load document
        result = app.load_documents([str(test_doc)])
        print(f"✅ Loaded document: {result['successful_loads']} files, {result['total_chunks']} chunks")
        
        # Test question answering
        if result['successful_loads'] > 0:
            answer = app.ask_question("What is machine learning?", top_k=2)
            if answer.get('success', True):
                print(f"✅ Generated answer: {answer['answer'][:100]}...")
                print(f"📊 Used {answer.get('context_chunks', 0)} context chunks")
                print(f"🤖 Provider: {answer.get('provider', 'Unknown')}")
            else:
                print(f"❌ Question answering failed: {answer.get('error', 'Unknown error')}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print("✅ Cleaned up test files")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Document Q&A System (No OpenAI)")
    print("=" * 50)
    
    # Check environment
    google_key = os.getenv('GOOGLE_API_KEY')
    if google_key:
        print(f"✅ Google API Key configured: {google_key[:10]}...")
    else:
        print("⚠️ Google API Key not found - Gemini provider will not be available")
    
    print()
    
    # Run tests
    tests = [
        ("Embeddings", test_embeddings),
        ("LLM Providers", test_llm_providers),
        ("Document Processing", test_document_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is working without OpenAI.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    print("\n💡 Next steps:")
    print("1. Run: python -m streamlit run streamlit_app.py")
    print("2. Open browser to http://localhost:8501")
    print("3. Upload documents and start asking questions!")


if __name__ == "__main__":
    main()
