# CO₂ Reduction AI Agent

An intelligent Retrieval-Augmented Generation (RAG) system that helps individuals and communities identify high CO₂ emission activities and provides actionable recommendations to reduce their carbon footprint.

## Features

- **Natural Language Queries**: Ask questions about reducing CO₂ emissions in plain English
- **Dataset Analysis**: Upload your activity data (CSV/Excel) for personalized carbon footprint analysis
- **Smart Recommendations**: Get AI-generated suggestions based on a curated knowledge base of sustainability tips
- **Quantitative Comparisons**: See emission reductions in kg CO₂/day and annual savings projections
- **Open-Source Stack**: Built entirely with open-source technologies (no proprietary APIs)
- **Interactive Web UI**: User-friendly Streamlit interface for easy interaction

## Technology Stack

- **LLM**: Groq (fastest, recommended), Ollama (local), or Hugging Face Inference
- **Agent Framework**: Custom RAG implementation with LangChain components
- **Vector Database**: ChromaDB with relevance filtering
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **UI**: Streamlit
- **Data Processing**: Pandas, Pydantic

## ⚡ Performance Improvements

This system now includes:

- **10x Faster Responses**: Using Groq API for sub-second LLM inference
- **Relevance Checking**: Automatically detects when queries are outside the knowledge base
- **Smart Fallbacks**: Returns honest "out of scope" messages instead of hallucinated answers
- **Optimized Vector Search**: Similarity scoring to filter irrelevant results

## LLM Provider Comparison

Choose the provider that best fits your needs:

| Provider           | Speed  | Setup  | Cost   | Best For                             |
| ------------------ | ------ | ------ | ------ | ------------------------------------ |
| **Groq** ⚡        | 0.5-2s | Easy   | Free\* | Production, demos, user-facing apps  |
| **Ollama** 🏠      | 2-10s  | Medium | Free   | Offline, unlimited requests, privacy |
| **HuggingFace** ☁️ | 5-20s  | Easy   | Free\* | Backup, specific models              |

\*Free tier with rate limits

### When to Use Each

**Groq (Recommended for most users)**

- ✅ Blazing fast responses (10x faster)
- ✅ No local setup required
- ✅ Free tier: 30 requests/min
- ❌ Requires internet connection
- ❌ Rate limits on free tier

**Ollama (Best for development/offline)**

- ✅ Unlimited requests (no rate limits)
- ✅ Works offline
- ✅ Complete privacy (data stays local)
- ❌ Slower responses
- ❌ Requires local setup & RAM

**HuggingFace (Backup option)**

- ✅ Many models available
- ✅ Easy setup
- ❌ Slowest responses
- ❌ Cold start delays

## Prerequisites

Before installing, ensure you have:

- **Python 3.9 or higher**
- **Git** (for cloning the repository)
- **LLM Provider** (choose based on table above):
  - **Groq API** (recommended) - Get free key at https://console.groq.com
  - **Ollama** (offline/unlimited) - Download from https://ollama.ai
  - **HuggingFace** (backup) - Get free key at https://huggingface.co
- **4GB+ RAM** (8GB recommended, especially for Ollama)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd co2-reduction-ai-agent
```

### Step 2: Create Virtual Environment

**On Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize Vector Store

```bash
python scripts/init_vector_store.py
```

This script loads sustainability tips into the ChromaDB vector database.

### Step 5: Verify Setup

```bash
python scripts/verify_setup.py
```

This checks that all components are properly configured.

### Step 6: Configure LLM Provider

**Option A: Groq (Recommended - Fastest)**

1. Get free API key from https://console.groq.com
2. Set environment variables:

```cmd
# Windows CMD
set GROQ_API_KEY=gsk_your_key_here
set LLM_PROVIDER=groq

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
$env:LLM_PROVIDER="groq"

# Linux/Mac
export GROQ_API_KEY=gsk_your_key_here
export LLM_PROVIDER=groq
```

3. Test setup: `python test_groq.py`

See [GROQ_SETUP.md](GROQ_SETUP.md) for detailed instructions.

**Option B: Ollama (Offline/Unlimited)**

1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama3`
3. Set environment variables:

```cmd
# Windows CMD
set LLM_PROVIDER=ollama
set LLM_MODEL=llama3

# Linux/Mac
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3
```

**Option C: HuggingFace (Backup)**

1. Get free API key from https://huggingface.co
2. Set environment variables:

```cmd
# Windows CMD
set HUGGINGFACE_API_KEY=hf_your_key_here
set LLM_PROVIDER=huggingface

# Linux/Mac
export HUGGINGFACE_API_KEY=hf_your_key_here
export LLM_PROVIDER=huggingface
```

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

**Note**: Make sure to set environment variables in the same terminal where you run the app!

## Quick Start with Setup Script

For convenience, use the provided setup script:

**On Windows:**

```cmd
setup.bat
```

**On Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### Asking Questions

1. Open the application in your browser
2. Type your question in the text input box
3. Click "Submit" or press Enter
4. View the AI-generated recommendations with emission comparisons

**Example queries:**

- "I drive 20 km daily using a petrol car. How can I reduce my carbon footprint?"
- "What's better for the environment: beef or chicken?"
- "What are the top 3 things I can do to reduce household emissions?"

See `data/example_queries.txt` for more examples.

### Uploading Activity Data

1. Prepare your data in CSV or Excel format with these columns:

   - `Activity`: Description of the activity (e.g., "Driving petrol car")
   - `Avg_CO2_Emission(kg/day)`: Daily CO₂ emission in kilograms
   - `Category`: One of Transport, Household, Food, or Lifestyle

2. Click the "Upload Dataset" section in the sidebar
3. Upload your file using the file uploader
4. View the analysis with:
   - Total daily and annual emissions
   - Top emission activities
   - Prioritized recommendations

**Example dataset format:**

```csv
Activity,Avg_CO2_Emission(kg/day),Category
Driving petrol car 20km,4.6,Transport
Eating beef,3.3,Food
Using electric heating,2.5,Household
```

### Interpreting Results

The agent provides:

- **Current Emission**: Your baseline CO₂ output
- **Recommendations**: Alternative actions ranked by impact
- **Emission Reduction**: Absolute (kg CO₂/day) and percentage savings
- **Annual Savings**: Projected yearly CO₂ reduction
- **Implementation Difficulty**: Easy, Medium, or Hard
- **Timeframe**: Immediate, Short-term, or Long-term

## Configuration

Edit `config.py` to customize settings:

### LLM Settings

```python
# Provider: "groq", "ollama", or "huggingface"
LLM_PROVIDER = "groq"

# Model names by provider:
# Groq: "llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"
# Ollama: "llama3", "mistral", "llama2"
# HuggingFace: "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MODEL = "llama-3.1-8b-instant"

# API Keys (or use environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Generation settings
LLM_TEMPERATURE = 0.3  # 0.0 (deterministic) to 1.0 (creative)
LLM_MAX_TOKENS = 300  # Maximum response length

# Relevance filtering
RELEVANCE_THRESHOLD = 0.5  # Minimum similarity score (0.0 to 1.0)
```

### Vector Store Settings

```python
VECTOR_DB_PATH = "./chroma_db"  # Database storage location
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
RETRIEVAL_TOP_K = 5  # Number of tips to retrieve per query
```

### Data Settings

```python
REFERENCE_DATA_PATH = "./data/reference_activities.csv"
SUSTAINABILITY_TIPS_PATH = "./data/sustainability_tips.txt"
```

### UI Settings

```python
PAGE_TITLE = "CO₂ Reduction AI Agent"
MAX_UPLOAD_SIZE_MB = 10  # Maximum file upload size
```

## Project Structure

```
co2-reduction-ai-agent/
├── app.py                      # Streamlit application entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── setup.bat / setup.sh        # Setup scripts
├── components/                 # Core application components
│   ├── agent.py               # Main agent orchestration
│   ├── llm_client.py          # LLM integration
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── embeddings.py          # Embedding generation
│   ├── query_processor.py     # Query parsing
│   ├── dataset_analyzer.py    # Dataset analysis
│   ├── recommendation_generator.py
│   ├── emission_calculator.py
│   ├── data_validator.py
│   ├── reference_data.py
│   ├── knowledge_loader.py
│   ├── prompt_templates.py
│   └── response_parser.py
├── models/                     # Data models
│   └── data_models.py         # Pydantic models
├── data/                       # Data files
│   ├── reference_activities.csv
│   ├── sustainability_tips.txt
│   └── example_queries.txt
├── scripts/                    # Utility scripts
│   ├── init_vector_store.py
│   └── verify_setup.py
├── utils/                      # Utility modules
│   ├── logger.py
│   └── error_handler.py
└── chroma_db/                  # Vector database storage
```

## Troubleshooting

### Issue: Slow responses (5+ seconds)

**Solution:**

1. **Switch to Groq** (fastest option):
   ```cmd
   set GROQ_API_KEY=your_key
   set LLM_PROVIDER=groq
   ```
2. Test speed: `python test_groq.py`
3. Expected: 0.5-2 second responses

### Issue: "Groq API key not provided"

**Solution:**

1. Get free key from https://console.groq.com
2. Set in same terminal where you run app:
   ```cmd
   set GROQ_API_KEY=gsk_your_key_here
   ```
3. Verify: `echo %GROQ_API_KEY%` (Windows) or `echo $GROQ_API_KEY` (Linux/Mac)

### Issue: Rate limit exceeded (Groq)

**Solution:**

1. Free tier: 30 requests/minute
2. Wait 60 seconds, or
3. Switch to Ollama for unlimited requests:
   ```cmd
   set LLM_PROVIDER=ollama
   ```

### Issue: Getting irrelevant answers

**Solution:**

1. System now detects irrelevant queries automatically
2. Adjust threshold in `config.py`:
   ```python
   RELEVANCE_THRESHOLD = 0.6  # Stricter (0.4 = more lenient)
   ```
3. Reinitialize vector store: `python scripts/init_vector_store.py`

### Issue: "Ollama service not available"

**Solution:**

1. Ensure Ollama is installed: `ollama --version`
2. Start Ollama service (it should auto-start, but you can restart it)
3. Verify model is downloaded: `ollama list`
4. If model is missing: `ollama pull llama3`
5. Test Ollama: `ollama run llama3 "Hello"`

### Issue: "Module not found" errors

**Solution:**

1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.9+)

### Issue: "ChromaDB collection not found"

**Solution:**

1. Run initialization script: `python scripts/init_vector_store.py`
2. Check that `chroma_db/` directory exists
3. Verify `data/sustainability_tips.txt` exists

### Issue: Want to switch providers

**Solution:**

Just change the environment variable:

```cmd
# Switch to Groq (fastest)
set LLM_PROVIDER=groq
set GROQ_API_KEY=your_key

# Switch to Ollama (offline/unlimited)
set LLM_PROVIDER=ollama

# Switch to HuggingFace (backup)
set LLM_PROVIDER=huggingface
set HUGGINGFACE_API_KEY=your_key
```

No code changes needed!

### Issue: "Invalid file format" when uploading dataset

**Solution:**

1. Ensure file is CSV or Excel (.xlsx, .xls)
2. Check required columns: Activity, Avg_CO2_Emission(kg/day), Category
3. Verify column names match exactly (case-sensitive)
4. Check that emission values are numeric and >= 0
5. Ensure Category values are: Transport, Household, Food, or Lifestyle

### Issue: Poor quality recommendations

**Solution:**

1. Try a different LLM model: Edit `LLM_MODEL` in config.py
2. Adjust temperature: Lower values (0.3-0.5) for more factual responses
3. Update sustainability tips: Edit `data/sustainability_tips.txt`
4. Reinitialize vector store: `python scripts/init_vector_store.py`

### Issue: Application won't start

**Solution:**

1. Check port 8501 is not in use: `netstat -an | findstr 8501` (Windows) or `lsof -i :8501` (Linux/Mac)
2. Try a different port: `streamlit run app.py --server.port 8502`
3. Check Streamlit installation: `streamlit --version`
4. Review error logs in terminal output

### Issue: Memory errors with large datasets

**Solution:**

1. Reduce dataset size (process in batches)
2. Increase system RAM or close other applications
3. Use CSV instead of Excel (more memory efficient)
4. Process fewer activities at once

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]

## Performance Tips

1. **For fastest responses**: Use Groq with `llama-3.1-8b-instant`
2. **For unlimited requests**: Use Ollama during development
3. **For offline work**: Use Ollama
4. **For privacy**: Use Ollama (data stays local)
5. **Hit rate limits?**: Switch to Ollama temporarily

## Additional Documentation

- [GROQ_SETUP.md](GROQ_SETUP.md) - Detailed Groq setup guide
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Upgrade from older versions
- [PERFORMANCE_FIXES.md](PERFORMANCE_FIXES.md) - Technical details on improvements

## Support

For issues and questions:

- Check the troubleshooting section above
- Review setup guides for your chosen provider
- Test with `test_groq.py` or `test_huggingface.py`
- Open an issue on GitHub

## Acknowledgments

Built with open-source technologies:

- **Groq** for blazing-fast LLM inference
- **Ollama** for local LLM inference
- **ChromaDB** for vector storage
- **Streamlit** for the web interface
- **SentenceTransformers** for embeddings
- **LangChain** components for RAG orchestration
