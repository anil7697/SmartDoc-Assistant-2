#!/usr/bin/env python
"""Test script for the Document Q&A system"""

import os
from pathlib import Path

# Test imports
def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from backend.chunker import TextChunker
        print("✅ TextChunker imported")
        
        from backend.embedder import EmbeddingGenerator
        print("✅ EmbeddingGenerator imported")
        
        from backend.vector_store import VectorStore
        print("✅ VectorStore imported")
        
        from backend.query_engine import QueryEngine
        print("✅ QueryEngine imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_chunker():
    """Test the text chunker"""
    print("\n📄 Testing text chunker...")
    
    try:
        from backend.chunker import TextChunker
        
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        test_text = "This is a test document. " * 20  # Create longer text
        chunks = chunker.chunk_text(test_text, "test.txt")
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"✅ First chunk: {chunks[0]['content'][:50]}...")
        
        stats = chunker.get_chunk_stats(chunks)
        print(f"✅ Stats: {stats['total_chunks']} chunks, avg size: {stats['avg_chunk_size']:.1f}")
        
        return True
    except Exception as e:
        print(f"❌ Chunker test failed: {str(e)}")
        return False

def test_embedder():
    """Test the embedding generator"""
    print("\n🔗 Testing embedding generator...")
    
    try:
        from backend.embedder import EmbeddingGenerator
        
        embedder = EmbeddingGenerator()
        print(f"✅ Initialized embedder: {embedder.model_name}")
        
        # Test single embedding
        test_text = "This is a test sentence."
        embedding = embedder.generate_embeddings(test_text)
        print(f"✅ Generated embedding: shape {embedding.shape}")
        
        # Test batch embeddings
        test_texts = ["First sentence.", "Second sentence."]
        embeddings = embedder.generate_embeddings(test_texts)
        print(f"✅ Generated batch embeddings: {len(embeddings)} embeddings")
        
        return True
    except Exception as e:
        print(f"❌ Embedder test failed: {str(e)}")
        return False

def test_vector_store():
    """Test the vector store"""
    print("\n🗄️ Testing vector store...")
    
    try:
        from backend.embedder import EmbeddingGenerator
        from backend.vector_store import VectorStore
        
        embedder = EmbeddingGenerator()
        vector_store = VectorStore(dimension=embedder.get_embedding_dimension())
        
        # Create test documents
        test_docs = [
            {"id": 0, "content": "Artificial intelligence is a field of computer science.", "filename": "ai.txt"},
            {"id": 1, "content": "Machine learning is a subset of artificial intelligence.", "filename": "ml.txt"},
            {"id": 2, "content": "Deep learning uses neural networks with multiple layers.", "filename": "dl.txt"}
        ]
        
        # Add documents
        vector_store.add_documents(test_docs, embedder)
        print(f"✅ Added {len(test_docs)} documents to vector store")
        
        # Test search
        query_embedding = embedder.generate_embeddings("What is AI?")
        results = vector_store.search(query_embedding, top_k=2)
        print(f"✅ Search returned {len(results)} results")
        
        if results:
            print(f"✅ Top result: {results[0][1]['content'][:50]}... (score: {results[0][0]:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {str(e)}")
        return False

def test_query_engine():
    """Test the query engine"""
    print("\n🔍 Testing query engine...")
    
    try:
        from backend.chunker import TextChunker
        from backend.embedder import EmbeddingGenerator
        from backend.vector_store import VectorStore
        from backend.query_engine import QueryEngine
        
        # Initialize components
        chunker = TextChunker()
        embedder = EmbeddingGenerator()
        vector_store = VectorStore(dimension=embedder.get_embedding_dimension())
        query_engine = QueryEngine(embedder, vector_store)
        
        # Create and add test document
        test_text = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that can perform tasks that typically require human intelligence. 
        These tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine Learning (ML) is a subset of AI that enables computers to learn and improve 
        from experience without being explicitly programmed. ML algorithms build mathematical 
        models based on training data to make predictions or decisions.
        """
        
        chunks = chunker.chunk_text(test_text, "ai_overview.txt")
        vector_store.add_documents(chunks, embedder)
        
        print(f"✅ Added {len(chunks)} chunks to knowledge base")
        
        # Test query
        question = "What is artificial intelligence?"
        result = query_engine.query(question, top_k=2)
        
        if result['success']:
            print(f"✅ Query successful")
            print(f"✅ Answer: {result['answer'][:100]}...")
            print(f"✅ Sources: {len(result['sources'])} documents")
        else:
            print(f"❌ Query failed: {result['answer']}")
        
        return result['success']
    except Exception as e:
        print(f"❌ Query engine test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Document Q&A System")
    print("=" * 50)
    
    # Check environment
    google_key = os.getenv('GOOGLE_API_KEY')
    if google_key:
        print(f"✅ Google API Key configured: {google_key[:10]}...")
    else:
        print("⚠️ Google API Key not found")
    
    print()
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Text Chunker", test_chunker),
        ("Embedding Generator", test_embedder),
        ("Vector Store", test_vector_store),
        ("Query Engine", test_query_engine),
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
        print("🎉 All tests passed! The system is working correctly.")
        print("\n💡 Next steps:")
        print("1. Run: python -m streamlit run app.py")
        print("2. Open browser to http://localhost:8501")
        print("3. Upload documents and start asking questions!")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
