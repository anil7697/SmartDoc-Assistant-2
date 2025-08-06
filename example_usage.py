#!/usr/bin/env python
"""Example usage of the Document Q&A system."""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.document_qa.app import DocumentQAApp


def create_sample_documents():
    """Create sample documents for testing."""
    documents = []
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Sample document 1: AI Overview
    doc1_content = """
    Artificial Intelligence (AI) Overview
    
    Artificial Intelligence is a branch of computer science that aims to create intelligent machines 
    that can perform tasks that typically require human intelligence. These tasks include learning, 
    reasoning, problem-solving, perception, and language understanding.
    
    Types of AI:
    1. Narrow AI (Weak AI): Designed for specific tasks
    2. General AI (Strong AI): Human-level intelligence across all domains
    3. Superintelligence: AI that surpasses human intelligence
    
    Applications of AI:
    - Healthcare: Medical diagnosis and drug discovery
    - Transportation: Autonomous vehicles
    - Finance: Fraud detection and algorithmic trading
    - Entertainment: Recommendation systems
    - Education: Personalized learning platforms
    """
    
    doc1_path = temp_dir / "ai_overview.txt"
    doc1_path.write_text(doc1_content)
    documents.append(doc1_path)
    
    # Sample document 2: Machine Learning
    doc2_content = """
    Machine Learning Fundamentals
    
    Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn 
    and improve from experience without being explicitly programmed. ML algorithms build mathematical 
    models based on training data to make predictions or decisions.
    
    Types of Machine Learning:
    
    1. Supervised Learning:
       - Uses labeled training data
       - Examples: Classification, Regression
       - Algorithms: Linear Regression, Decision Trees, Neural Networks
    
    2. Unsupervised Learning:
       - Finds patterns in data without labels
       - Examples: Clustering, Dimensionality Reduction
       - Algorithms: K-Means, PCA, Autoencoders
    
    3. Reinforcement Learning:
       - Learns through interaction with environment
       - Uses rewards and penalties
       - Applications: Game playing, Robotics
    
    Popular ML Libraries:
    - Python: scikit-learn, TensorFlow, PyTorch
    - R: caret, randomForest
    - Java: Weka, MOA
    """
    
    doc2_path = temp_dir / "machine_learning.txt"
    doc2_path.write_text(doc2_content)
    documents.append(doc2_path)
    
    # Sample document 3: Deep Learning
    doc3_content = """
    Deep Learning and Neural Networks
    
    Deep Learning is a subset of machine learning that uses artificial neural networks with multiple 
    layers (hence "deep") to model and understand complex patterns in data. It's inspired by the 
    structure and function of the human brain.
    
    Key Concepts:
    
    Neural Networks:
    - Composed of interconnected nodes (neurons)
    - Each connection has a weight
    - Activation functions determine output
    
    Deep Neural Networks:
    - Multiple hidden layers
    - Can learn hierarchical representations
    - Automatic feature extraction
    
    Popular Architectures:
    
    1. Convolutional Neural Networks (CNNs):
       - Excellent for image processing
       - Use convolution and pooling layers
       - Applications: Computer vision, medical imaging
    
    2. Recurrent Neural Networks (RNNs):
       - Process sequential data
       - Have memory capabilities
       - Applications: Natural language processing, time series
    
    3. Transformers:
       - Attention-based architecture
       - Parallel processing capabilities
       - Applications: Language models, machine translation
    
    Training Process:
    - Forward propagation
    - Loss calculation
    - Backpropagation
    - Weight updates using optimizers (SGD, Adam)
    """
    
    doc3_path = temp_dir / "deep_learning.txt"
    doc3_path.write_text(doc3_content)
    documents.append(doc3_path)
    
    return documents, temp_dir


def main():
    """Main example function."""
    print("🚀 Document Q&A System - Example Usage")
    print("=" * 50)
    
    # Check if API keys are configured
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    try:
        # Initialize the application
        print("📚 Initializing Document Q&A System...")
        app = DocumentQAApp()
        print(f"✅ Application initialized successfully!")
        print(f"Available LLM providers: {app.answer_generator.get_available_providers()}")
        
        # Create sample documents
        print("\n📄 Creating sample documents...")
        documents, temp_dir = create_sample_documents()
        print(f"Created {len(documents)} sample documents in {temp_dir}")
        
        # Load documents
        print("\n🔄 Loading and processing documents...")
        load_result = app.load_documents([str(doc) for doc in documents])
        
        print(f"✅ Successfully loaded: {load_result['successful_loads']} documents")
        print(f"📊 Total chunks created: {load_result['total_chunks']}")
        
        if load_result['errors']:
            print(f"❌ Errors: {load_result['errors']}")
        
        # Example questions
        questions = [
            "What is artificial intelligence?",
            "What are the types of machine learning?",
            "Explain neural networks and deep learning",
            "What are the applications of AI?",
            "Compare supervised and unsupervised learning",
            "What are CNNs and RNNs?"
        ]
        
        print(f"\n❓ Asking {len(questions)} example questions...")
        print("-" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔍 Question {i}: {question}")
            
            # Get answer
            result = app.ask_question(
                question=question,
                top_k=3,
                temperature=0.7
            )
            
            if result.get('success', True):
                print(f"🤖 Answer: {result['answer'][:200]}...")
                print(f"📊 Context chunks: {result.get('context_chunks', 0)}")
                print(f"⚡ Provider: {result.get('provider', 'Unknown')}")
                
                # Show sources
                if 'sources' in result and result['sources']:
                    sources = [s['document'] for s in result['sources'][:2]]
                    print(f"📚 Sources: {', '.join(sources)}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Show system status
        print(f"\n📊 System Status:")
        status = app.get_status()
        print(f"Documents loaded: {status['documents_loaded']}")
        print(f"Vector store size: {status['vector_store_size']}")
        print(f"Embedding model: {status['embedding_model']}")
        
        # Save vector store for future use
        print(f"\n💾 Saving vector store...")
        app.save_vector_store()
        print(f"✅ Vector store saved to: {app.vector_store_path}")
        
        print(f"\n🎉 Example completed successfully!")
        print(f"You can now run 'streamlit run streamlit_app.py' to use the web interface")
        
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during example execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
