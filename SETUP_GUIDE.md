# Document Q&A System - Setup Guide

This guide will walk you through setting up the Document Q&A system step by step.

## 📋 Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher installed
- Git (for cloning the repository)
- At least 4GB of RAM
- Internet connection for API access

## 🔧 Step-by-Step Setup

### Step 1: Environment Setup

1. **Create a project directory:**
   ```bash
   mkdir document-qa-system
   cd document-qa-system
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Verify Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

### Step 2: Install Dependencies

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   pip list | grep streamlit
   pip list | grep openai
   pip list | grep faiss
   ```

### Step 3: API Keys Configuration

1. **Get OpenAI API Key:**
   - Go to [OpenAI Platform](https://platform.openai.com/)
   - Create an account or sign in
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Get Google API Key (Optional):**
   - Go to [Google AI Studio](https://makersuite.google.com/)
   - Create a project
   - Enable the Generative AI API
   - Create credentials (API Key)
   - Copy the key

3. **Configure environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit the .env file with your keys
   # On Windows: notepad .env
   # On macOS/Linux: nano .env
   ```

4. **Add your API keys to .env:**
   ```env
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   GOOGLE_API_KEY=your-google-api-key-here
   ```

### Step 4: Test Installation

1. **Run basic tests:**
   ```bash
   python run_tests.py
   ```

2. **Test API connectivity:**
   ```bash
   python -c "
   from src.document_qa.app import DocumentQAApp
   app = DocumentQAApp()
   print('✅ Application initialized successfully!')
   print(f'Available LLM providers: {app.answer_generator.get_available_providers()}')
   "
   ```

### Step 5: Launch the Application

1. **Start Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser:**
   - The app should automatically open at `http://localhost:8501`
   - If not, manually navigate to that URL

3. **Verify the interface:**
   - You should see the Document Q&A System homepage
   - Check that API keys show as configured in the sidebar

## 🧪 Testing Your Setup

### Test 1: Document Upload

1. Create a simple test document:
   ```bash
   echo "This is a test document about artificial intelligence. AI is a field of computer science." > test_doc.txt
   ```

2. In the Streamlit app:
   - Go to "Document Upload" tab
   - Upload `test_doc.txt`
   - Click "Process Documents"
   - Verify successful processing

### Test 2: Question Answering

1. In the "Ask Questions" tab:
   - Enter: "What is AI?"
   - Click "Get Answer"
   - Verify you get a relevant response

### Test 3: Provider Testing

1. In the "System Info" tab:
   - Click "Test" buttons for available providers
   - Verify all providers show as working

## 🔧 Optional: Ollama Setup (Local LLM)

If you want to use local models with Ollama:

1. **Install Ollama:**
   - Visit [Ollama website](https://ollama.ai/)
   - Download and install for your OS

2. **Pull a model:**
   ```bash
   ollama pull llama2
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Update .env:**
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: "ModuleNotFoundError"
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### Issue: "OpenAI API key not found"
**Solution:**
1. Check `.env` file exists and contains your key
2. Restart the Streamlit app
3. Verify key format (should start with `sk-`)

#### Issue: "FAISS installation failed"
**Solution:**
```bash
# Try installing FAISS separately
pip install faiss-cpu --no-cache-dir

# On M1 Mac, you might need:
conda install -c conda-forge faiss-cpu
```

#### Issue: "Streamlit won't start"
**Solution:**
```bash
# Check if port is in use
netstat -an | grep 8501

# Use different port
streamlit run streamlit_app.py --server.port 8502
```

#### Issue: "Vector store loading failed"
**Solution:**
1. Delete existing vector store: `rm -rf data/vector_store/*`
2. Re-upload and process documents
3. Check file permissions

### Performance Issues

#### Slow document processing:
- Reduce chunk size in settings
- Process fewer documents at once
- Check available RAM

#### Slow question answering:
- Reduce number of search results
- Use faster LLM provider (Gemini Flash)
- Check internet connection

## 📊 System Requirements

### Minimum Requirements:
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection

### Recommended Requirements:
- Python 3.10+
- 8GB RAM
- 5GB free disk space
- Fast internet connection

## 🔄 Updates and Maintenance

### Updating Dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Backing Up Data:
```bash
# Backup vector store
cp -r data/vector_store/ backup/

# Backup configuration
cp .env backup/
```

### Monitoring Logs:
```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log
```

## 🎯 Next Steps

After successful setup:

1. **Upload your documents** - Start with a few small files
2. **Experiment with settings** - Try different chunk sizes and LLM providers
3. **Test various question types** - Factual, analytical, comparative
4. **Monitor performance** - Check logs and system resources
5. **Customize as needed** - Modify code for your specific use case

## 📞 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review the main README.md
3. Check the test files for examples
4. Create an issue on GitHub with:
   - Your OS and Python version
   - Error messages
   - Steps to reproduce

## ✅ Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] API keys configured in .env
- [ ] Basic tests pass
- [ ] Streamlit app launches
- [ ] Document upload works
- [ ] Question answering works
- [ ] All LLM providers tested

Congratulations! Your Document Q&A system is ready to use! 🎉
