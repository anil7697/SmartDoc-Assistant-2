#!/usr/bin/env python
"""Simple script to run the Document Q&A application"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'sentence_transformers', 
        'faiss_cpu',
        'google.generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('_cpu', ''))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_api_key():
    """Check if Google API key is configured"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️ Google API key not found")
        print("💡 Make sure GOOGLE_API_KEY is set in your .env file")
        return False
    
    print(f"✅ Google API key configured: {api_key[:10]}...")
    return True

def main():
    """Main function to run the application"""
    print("🚀 Starting Document Q&A System")
    print("=" * 40)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All required packages are installed")
    
    # Check API key
    print("\n🔑 Checking API configuration...")
    check_api_key()  # Warning only, not blocking
    
    # Run the application
    print("\n🌐 Starting Streamlit application...")
    print("📍 The app will open at: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
