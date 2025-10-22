"""
Setup Verification Script

This script verifies that the setup was completed successfully
by checking all required components.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, f"Python {version.major}.{version.minor}.{version.micro}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'langchain',
        'sentence_transformers',
        'chromadb',
        'pandas',
        'pydantic',
        'ollama'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, installed, missing


def check_data_files():
    """Check if required data files exist"""
    import config
    
    required_files = [
        (config.REFERENCE_DATA_PATH, "Reference activities CSV"),
        (config.SUSTAINABILITY_TIPS_PATH, "Sustainability tips"),
    ]
    
    missing = []
    found = []
    
    for filepath, description in required_files:
        if Path(filepath).exists():
            found.append((filepath, description))
        else:
            missing.append((filepath, description))
    
    return len(missing) == 0, found, missing


def check_vector_store():
    """Check if vector store is initialized"""
    try:
        from components.vector_store import VectorStore
        import config
        
        vector_store = VectorStore(
            collection_name="sustainability_tips",
            persist_directory=config.VECTOR_DB_PATH,
            embedding_model=config.EMBEDDING_MODEL
        )
        
        stats = vector_store.get_collection_stats()
        doc_count = stats['document_count']
        
        if doc_count > 0:
            return True, f"{doc_count} documents"
        else:
            return False, "No documents found"
            
    except Exception as e:
        return False, str(e)


def check_ollama():
    """Check if Ollama is available"""
    import subprocess
    
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version
        else:
            return False, "Ollama not responding"
            
    except FileNotFoundError:
        return False, "Ollama not installed"
    except subprocess.TimeoutExpired:
        return False, "Ollama timeout"
    except Exception as e:
        return False, str(e)


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("CO2 Reduction AI Agent - Setup Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Check Python version
    print("[1/5] Checking Python version...")
    passed, info = check_python_version()
    if passed:
        print(f"  ✓ {info}")
    else:
        print(f"  ✗ {info} (requires 3.9+)")
        all_passed = False
    print()
    
    # Check dependencies
    print("[2/5] Checking dependencies...")
    passed, installed, missing = check_dependencies()
    if passed:
        print(f"  ✓ All {len(installed)} required packages installed")
    else:
        print(f"  ✗ Missing packages: {', '.join(missing)}")
        all_passed = False
    print()
    
    # Check data files
    print("[3/5] Checking data files...")
    passed, found, missing = check_data_files()
    if passed:
        print(f"  ✓ All {len(found)} required data files found")
        for filepath, desc in found:
            print(f"    - {desc}: {filepath}")
    else:
        print(f"  ✗ Missing files:")
        for filepath, desc in missing:
            print(f"    - {desc}: {filepath}")
        all_passed = False
    print()
    
    # Check vector store
    print("[4/5] Checking vector store...")
    passed, info = check_vector_store()
    if passed:
        print(f"  ✓ Vector store initialized: {info}")
    else:
        print(f"  ✗ Vector store issue: {info}")
        all_passed = False
    print()
    
    # Check Ollama
    print("[5/5] Checking Ollama...")
    passed, info = check_ollama()
    if passed:
        print(f"  ✓ Ollama available: {info}")
    else:
        print(f"  ⚠ Ollama: {info}")
        print("    Note: Ollama is required to run the AI agent")
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ Setup verification PASSED")
        print("=" * 60)
        print()
        print("You can now run the application with:")
        print("  streamlit run app.py")
        return 0
    else:
        print("✗ Setup verification FAILED")
        print("=" * 60)
        print()
        print("Please run the setup script to fix issues:")
        print("  Windows: setup.bat")
        print("  Linux/Mac: ./setup.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
