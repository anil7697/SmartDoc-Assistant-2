#!/usr/bin/env python
"""Test runner script for the Document Q&A system."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the test suite with coverage reporting."""
    print("🧪 Running Document Q&A System Tests")
    print("=" * 50)
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    
    try:
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=70",
            "-v"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except FileNotFoundError:
        print("❌ pytest not found. Please install requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
