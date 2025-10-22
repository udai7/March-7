# Scripts Directory

This directory contains utility scripts for setting up and managing the CO2 Reduction AI Agent.

## Available Scripts

### init_vector_store.py

Initializes the ChromaDB vector store with sustainability tips from the knowledge base.

**Usage:**

```bash
python scripts/init_vector_store.py
```

**What it does:**

- Parses sustainability tips from `data/sustainability_tips.txt`
- Generates embeddings using SentenceTransformers
- Populates ChromaDB with the embedded documents
- Verifies the initialization with a test search

**When to run:**

- During initial setup (automatically called by setup scripts)
- After updating the sustainability tips file
- If the vector store becomes corrupted

### verify_setup.py

Verifies that all components are properly installed and configured.

**Usage:**

```bash
python scripts/verify_setup.py
```

**What it checks:**

- Python version (3.9+)
- Required Python packages
- Data files (reference activities, sustainability tips)
- Vector store initialization
- Ollama installation (optional)

**When to run:**

- After running the setup script
- Before starting the application
- When troubleshooting issues

## Setup Scripts (Root Directory)

### setup.bat (Windows)

Automated setup script for Windows systems.

**Usage:**

```cmd
setup.bat
```

### setup.sh (Linux/Mac)

Automated setup script for Unix-based systems.

**Usage:**

```bash
chmod +x setup.sh
./setup.sh
```

**What the setup scripts do:**

1. Check Python version (3.9+)
2. Create virtual environment
3. Activate virtual environment
4. Upgrade pip
5. Install dependencies from requirements.txt
6. Initialize vector store
7. Check Ollama installation

## Troubleshooting

### Vector Store Issues

If the vector store fails to initialize:

1. Check that `data/sustainability_tips.txt` exists
2. Ensure ChromaDB is properly installed: `pip install chromadb`
3. Delete the `chroma_db` directory and re-run initialization

### Ollama Not Found

Ollama is required to run the AI agent with local LLMs:

1. Install Ollama from https://ollama.ai/
2. Pull a model: `ollama pull llama3` or `ollama pull mistral`
3. Verify installation: `ollama --version`

### Permission Issues (Linux/Mac)

If you get permission errors:

```bash
chmod +x setup.sh
chmod +x scripts/*.py
```

### Virtual Environment Issues

If the virtual environment fails to activate:

- Windows: `venv\Scripts\activate.bat`
- Linux/Mac: `source venv/bin/activate`

If it still doesn't work, recreate it:

```bash
rm -rf venv  # or rmdir /s /q venv on Windows
python -m venv venv
```
