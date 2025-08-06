<<<<<<< HEAD
# SmartDoc-Assistant-2
An Open Source AI Assistant for Document Q&amp;A
=======
# Document Q&A System

A comprehensive document question-answering system with multiple LLM support, built with Python and Streamlit.

## 🚀 Features

- **Multi-format Document Support**: PDF, TXT, and DOCX files
- **Advanced Text Processing**: Intelligent chunking, preprocessing, and cleaning
- **Multiple LLM Providers**: Support for OpenAI GPT, Google Gemini, and Ollama
- **Vector Search**: FAISS-powered semantic search with configurable parameters
- **Interactive Web UI**: Beautiful Streamlit interface with real-time processing
- **Persistent Storage**: Save and load vector stores for quick startup
- **Comprehensive Logging**: Detailed logging and error handling
- **Extensible Architecture**: Modular design for easy customization

## 📋 Requirements

- Python 3.8+
- Google API key (for Gemini Flash 1.5 models)
- Ollama (optional, for local models)
- No OpenAI dependency - uses local sentence-transformers for embeddings

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd document-qa-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` file with your API keys:

```env
# Required for Gemini Flash 1.5
GOOGLE_API_KEY=your_google_api_key_here

# Optional for Ollama (if running locally)
OLLAMA_BASE_URL=http://localhost:11434
```

## 🚀 Quick Start

### 1. Start the Application

```bash
streamlit run streamlit_app.py
```

### 2. Upload Documents

1. Navigate to the "Document Upload" tab
2. Select PDF, TXT, or DOCX files
3. Click "Process Documents"
4. Wait for processing to complete

### 3. Ask Questions

1. Go to the "Ask Questions" tab
2. Enter your question about the uploaded documents
3. Click "Get Answer"
4. View the AI-generated response with source citations

## 📁 Project Structure

```
document-qa-system/
├── src/
│   └── document_qa/
│       ├── document_processor/     # Document loading and chunking
│       ├── embeddings/            # Embedding generation
│       ├── vector_store/          # Vector storage with FAISS
│       ├── query_processor/       # Query processing and search
│       ├── llm_integration/       # LLM providers
│       ├── app.py                 # Main application orchestrator
│       └── utils.py               # Utility functions
├── tests/                         # Unit tests
├── data/                          # Data storage
│   ├── documents/                 # Uploaded documents
│   └── vector_store/              # Vector store files
├── logs/                          # Application logs
├── config.py                      # Configuration management
├── streamlit_app.py              # Streamlit web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes | - |
| `GOOGLE_API_KEY` | Google API key for Gemini | No | - |
| `OLLAMA_BASE_URL` | Ollama server URL | No | `http://localhost:11434` |
| `DEFAULT_CHUNK_SIZE` | Text chunk size | No | `1000` |
| `DEFAULT_CHUNK_OVERLAP` | Chunk overlap size | No | `200` |
| `DEFAULT_TOP_K` | Default search results | No | `5` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

### Advanced Settings

You can modify settings in the Streamlit sidebar:

- **Chunk Size**: Controls how documents are split (500-2000 characters)
- **Chunk Overlap**: Overlap between chunks (50-500 characters)
- **Number of Results**: How many relevant chunks to retrieve (1-10)
- **LLM Temperature**: Controls response creativity (0.0-1.0)

## 🤖 LLM Providers

### OpenAI GPT (via embeddings)
- Used for generating embeddings
- Requires `OPENAI_API_KEY`

### Google Gemini
- Fast and efficient responses
- Requires `GOOGLE_API_KEY`
- Models: `gemini-1.5-flash`, `gemini-1.5-pro`

### Ollama (Local)
- Run models locally for privacy
- Requires Ollama installation
- Supports various open-source models

## 📊 Usage Examples

### Basic Document Q&A

1. Upload a PDF document about machine learning
2. Ask: "What is supervised learning?"
3. Get an AI-generated answer with source citations

### Multi-document Analysis

1. Upload multiple research papers
2. Ask: "Compare the methodologies discussed in these papers"
3. Get a comprehensive comparison across documents

### Technical Documentation

1. Upload API documentation
2. Ask: "How do I authenticate with the API?"
3. Get step-by-step instructions with code examples

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_document_processor.py
```

## 🔧 Development

### Adding New Document Types

1. Extend `DocumentLoader` in `src/document_qa/document_processor/document_loader.py`
2. Add the new file extension to `SUPPORTED_EXTENSIONS`
3. Implement the loading method

### Adding New LLM Providers

1. Create a new provider class inheriting from `LLMProvider`
2. Implement required methods: `generate_answer()`, `get_model_name()`, `is_available()`
3. Register the provider in `AnswerGenerator`

### Custom Preprocessing

Modify `TextPreprocessor` in `src/document_qa/document_processor/preprocessor.py` to add custom text cleaning logic.

## 📝 API Reference

### DocumentQAApp

Main application class that orchestrates all components.

```python
from src.document_qa.app import DocumentQAApp

app = DocumentQAApp()

# Load documents
result = app.load_documents(['path/to/document.pdf'])

# Ask questions
answer = app.ask_question("What is this document about?")
```

### Key Methods

- `load_documents(file_paths)`: Load documents from file paths
- `load_documents_from_bytes(files_data)`: Load from bytes (Streamlit uploads)
- `ask_question(question, top_k, llm_provider, temperature)`: Get answers
- `save_vector_store(path)`: Save vector store
- `load_vector_store(path)`: Load vector store
- `get_status()`: Get application status

## 🐛 Troubleshooting

### Common Issues

1. **"No LLM providers available"**
   - Check API keys in `.env` file
   - Verify internet connection
   - For Ollama: ensure server is running

2. **"Failed to load document"**
   - Check file format is supported
   - Verify file is not corrupted
   - Check file permissions

3. **"Vector store loading failed"**
   - Ensure vector store was saved properly
   - Check file permissions
   - Verify FAISS installation

4. **Poor search results**
   - Try different chunk sizes
   - Adjust similarity thresholds
   - Use more specific queries

### Performance Tips

- Use smaller chunk sizes for better precision
- Increase chunk overlap for better context
- Save vector store to avoid reprocessing
- Use local Ollama for faster responses

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the test files for usage examples

## 🔄 Changelog

### v1.0.0
- Initial release
- Multi-format document support
- Multiple LLM providers
- Streamlit web interface
- Comprehensive testing suite
>>>>>>> 398cdbb (Initial commit)
