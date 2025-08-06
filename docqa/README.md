# Document Q&A System

A streamlined document question-answering system using Gemini Flash 1.5 and local embeddings.

## 📁 Project Structure

```
docqa/
├── app.py                  # Streamlit UI
├── backend/
│   ├── chunker.py          # Text chunking logic
│   ├── embedder.py         # Local embeddings (Sentence Transformers)
│   ├── vector_store.py     # Vector DB logic (FAISS)
│   ├── query_engine.py     # Query processing logic
├── data/
│   └── documents/          # Upload your documents here
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd docqa
pip install -r requirements.txt
```

### 2. Configure API Key

Your Google API key is already configured in `.env`:
```
GOOGLE_API_KEY=AIzaSyBYbzeGj9cE70YXax5-_FnRzskJyEeWYxA
```

### 3. Run the Application

```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📚 How to Use

1. **Upload Documents**: Go to the "Document Upload" tab and upload PDF or TXT files
2. **Process Documents**: Click "Process Documents" to chunk and embed your files
3. **Ask Questions**: Go to "Ask Questions" tab and ask about your documents
4. **Get Answers**: Receive AI-generated answers with source citations

## 🔧 Features

- **Local Embeddings**: Uses Sentence Transformers (no OpenAI dependency)
- **Gemini Flash 1.5**: Fast, high-quality answer generation
- **FAISS Vector Store**: Efficient similarity search
- **Smart Chunking**: Intelligent text splitting with overlap
- **Source Citations**: See which documents were used for answers
- **Clean UI**: Simple, intuitive Streamlit interface

## ⚙️ Configuration

You can adjust settings in the sidebar:
- **Chunk Size**: How large each text chunk should be (500-2000 characters)
- **Chunk Overlap**: How much chunks should overlap (50-500 characters)  
- **Number of Results**: How many relevant chunks to use for answers (1-10)

## 🧪 Supported File Types

- **PDF**: Portable Document Format files
- **TXT**: Plain text files

## 📊 System Requirements

- Python 3.8+
- 4GB+ RAM (for embedding models)
- Google API key for Gemini Flash 1.5

## 🔍 How It Works

1. **Document Processing**: Files are loaded and split into overlapping chunks
2. **Embedding Generation**: Each chunk is converted to a vector using Sentence Transformers
3. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
4. **Query Processing**: User questions are embedded and matched against document chunks
5. **Answer Generation**: Relevant chunks are sent to Gemini Flash 1.5 for answer synthesis

## 🛠️ Troubleshooting

### Common Issues

1. **"streamlit not recognized"**: Use `python -m streamlit run app.py`
2. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
3. **No answers generated**: Check that your Google API key is valid
4. **Slow processing**: Large documents take time to process - this is normal

### Performance Tips

- Use smaller chunk sizes for more precise answers
- Increase chunk overlap for better context
- Upload documents in smaller batches
- Restart the app if it becomes slow

## 📝 Example Usage

1. Upload a research paper (PDF)
2. Ask: "What is the main conclusion of this study?"
3. Get an AI-generated answer with citations to specific sections

## 🔄 Updates

This simplified structure makes it easy to:
- Add new document types
- Modify chunking strategies  
- Integrate additional LLM providers
- Customize the UI

## 📞 Support

For issues:
1. Check this README
2. Verify your API key is correct
3. Ensure all dependencies are installed
4. Try restarting the application
