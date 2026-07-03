# Environmental Impact AI Agent 🌍

An intelligent AI system that helps you understand and reduce your **complete environmental footprint** through personalized, data-driven recommendations covering CO₂ emissions, water usage, energy consumption, and waste generation.

## ✨ What It Does

Ask questions like _"I drive 20km daily, how can I reduce my environmental impact?"_ or upload your activity data, and get:

- **Comprehensive environmental analysis** across CO₂, water, energy, and waste metrics
- **Personalized recommendations** ranked by overall environmental impact
- **Quantified savings** in kg CO₂/day, liters water, kWh energy, and waste reduction
- **Sustainability grades** (A+ to F) based on your environmental footprint
- **Health & cost benefits** alongside environmental improvements
- **Source-backed advice** from a curated sustainability knowledge base

## 🎯 Key Features

| Feature                         | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| 🌡️ **CO₂ Tracking**             | Measure and reduce carbon emissions from all your activities                  |
| 💧 **Water Footprint**          | Track water consumption and get water-saving recommendations                  |
| ⚡ **Energy Analysis**          | Monitor energy usage and optimize for efficiency                              |
| ♻️ **Waste Management**         | Reduce waste generation with practical alternatives                           |
| 💬 **Natural Language Queries** | Ask questions in plain English, get instant AI-powered answers                |
| 📊 **Dataset Analysis**         | Upload CSV/Excel files for comprehensive multi-metric analysis                |
| 🤖 **RAG-Powered Intelligence** | Combines vector search + LLM reasoning for accurate, grounded recommendations |
| 📈 **Impact Quantification**    | See precise reductions across all environmental metrics                       |
| 🏆 **Sustainability Grading**   | Get an overall grade (A+ to F) based on your environmental performance        |
| 🎨 **Interactive Dashboard**    | Clean Streamlit interface with multi-metric charts and visualizations         |
| 💰 **Financial Calculator**     | Calculate cost savings, ROI on green investments, and carbon credits          |
| 🧾 **Receipt Scanner**          | Analyze purchase receipts for environmental impact of products                |

### 💰 Financial Impact Calculator (NEW!)

Calculate the financial benefits of your eco-friendly choices:

- **Cost Savings Calculator**: Calculate savings from switching transport modes, reducing energy/water usage
- **Green Investment ROI**: Analyze payback periods and returns for solar panels, EVs, heat pumps, and more
- **Utility Cost Comparison**: Compare current vs. optimized utility costs with detailed breakdowns
- **Carbon Credit Calculator**: Estimate your carbon credit earnings or tax liability

### 🧾 Receipt & Product Scanner (NEW!)

Analyze your shopping to understand environmental impact:

- **Receipt Text Analysis**: Paste receipt text to auto-detect products and calculate impact
- **Manual Product Entry**: Add products individually for detailed environmental analysis
- **Category-Based Impact**: See CO₂, water, and waste footprint by product category
- **Eco Recommendations**: Get personalized suggestions for greener alternatives
- **Sustainability Scoring**: Each product gets a 0-100 sustainability score

## 📊 Environmental Metrics Tracked

| Metric             | Unit       | Description                                     |
| ------------------ | ---------- | ----------------------------------------------- |
| 🌡️ CO₂ Emissions   | kg/day     | Carbon dioxide equivalent emissions             |
| 💧 Water Usage     | liters/day | Total water consumption including virtual water |
| ⚡ Energy          | kWh/day    | Electricity and fuel energy consumption         |
| 🗑️ Waste           | kg/day     | Solid waste generation                          |
| 🏭 Pollution Index | 0-100      | Combined air/water pollution score              |
| 🌲 Land Use        | m²         | Land area required for activities               |

## 🚀 Quick Start (3 Steps)

### Prerequisites

- Python 3.9+ installed
- Internet connection (for Groq API) or local compute (for Ollama)

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd March-7

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Initialize the System

```bash
# Load knowledge base into vector database
python scripts/init_vector_store.py

# Verify setup (optional but recommended)
python scripts/verify_setup.py
```

### Step 3: Configure & Run

**Option A: Using Groq (Recommended - Fastest)**

1. Get a free API key from [console.groq.com](https://console.groq.com)
2. Set your key and run:

```powershell
# PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

**Option B: Using Ollama (Offline/Unlimited)**

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model and run:

```bash
ollama pull llama3
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 💡 How to Use

### Ask Questions (Natural Language)

Simply type your question and get instant recommendations:

**Example Questions:**

```
"I drive 20 km daily using a petrol car. How can I reduce emissions?"
"What's more eco-friendly: beef or chicken?"
"Top 3 ways to reduce household carbon footprint?"
```

### Upload Your Data (CSV/Excel)

**Required Format:**

```csv
Activity,Avg_CO2_Emission(kg/day),Category
Driving petrol car 20km,4.6,Transport
Eating beef daily,3.3,Food
Electric heating 8hrs,2.5,Household
```

**Categories:** Transport, Household, Food, Lifestyle

**What You Get:**

- 📊 Total daily & annual emissions
- 🔝 Top emitting activities
- 💡 Ranked recommendations by impact
- 📈 Potential savings projections

### Understanding Results

Each recommendation includes:

| Field              | Meaning                                         |
| ------------------ | ----------------------------------------------- |
| **Action**         | What to do (e.g., "Switch to public transport") |
| **Reduction**      | CO₂ saved per day (kg) and percentage           |
| **Difficulty**     | Easy / Medium / Hard                            |
| **Timeframe**      | Immediate / Short-term / Long-term              |
| **Annual Savings** | Total kg CO₂/year if adopted                    |

---

## ⚙️ Tech Stack

| Component         | Technology                  | Purpose                                 |
| ----------------- | --------------------------- | --------------------------------------- |
| **LLM**           | Groq / Ollama / HuggingFace | Text generation & reasoning             |
| **RAG Framework** | LangChain                   | Retrieval-augmented generation pipeline |
| **Vector DB**     | ChromaDB                    | Semantic search over knowledge base     |
| **Embeddings**    | SentenceTransformers        | Text → vector conversion                |
| **Frontend**      | Streamlit                   | Interactive web interface               |
| **Data**          | Pandas, Pydantic            | Processing & validation                 |

**Why This Stack?**

- ⚡ Fast: Sub-2s responses with Groq
- 🔒 Private: Can run 100% offline with Ollama
- 💰 Free: All tools have generous free tiers
- 🧩 Modular: Easy to swap LLM providers

## Project Structure

```
March-7/
├── app.py                      # Streamlit application entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── streamlit.conf              # Nginx reverse-proxy config (deployment)
├── deploy.tar.gz               # Prebuilt deployment bundle
├── components/                 # Core application components
│   ├── agent.py               # Main agent orchestration
│   ├── llm_client.py          # LLM integration
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── embeddings.py          # Embedding generation
│   ├── query_processor.py     # Query parsing
│   ├── dataset_analyzer.py    # Dataset analysis
│   ├── recommendation_generator.py  # Recommendation generation
│   ├── recommendation_ranker.py     # Recommendation ranking
│   ├── emission_calculator.py       # CO₂ calculations
│   ├── data_validator.py            # Input validation
│   ├── response_validator.py        # LLM response validation
│   ├── response_parser.py           # Response parsing
│   ├── reference_data.py            # Reference emission factors
│   ├── knowledge_loader.py          # Knowledge base loader
│   ├── prompt_templates.py          # LLM prompt templates
│   ├── context_manager.py           # Conversation context
│   ├── feedback_collector.py        # User feedback capture
│   ├── environmental_scorer.py      # Sustainability scoring
│   ├── financial_calculator.py      # ROI & cost savings
│   ├── receipt_scanner.py           # Receipt/product analysis
│   └── example_queries.txt          # Sample queries
├── models/                     # Data models
│   └── data_models.py         # Pydantic models
├── data/                       # Data files
│   ├── reference_activities.csv
│   ├── sustainability_tips.txt
│   ├── environmental_sustainability_tips.txt
│   └── sample_emissions.csv    # Example upload dataset
├── scripts/                    # Utility scripts
│   ├── init_vector_store.py
│   ├── update_reference_data.py
│   └── verify_setup.py
├── utils/                      # Utility modules
│   ├── logger.py
│   └── error_handler.py
├── tests/                      # Test suite
│   └── test_recommendations.py
├── docs/                       # Documentation
│   └── USER_GUIDE.md
├── .github/workflows/          # CI / keep-alive workflows
├── .streamlit/                 # Streamlit theme & server config
└── chroma_db/                  # Vector database storage (gitignored)
```

## 🔧 Common Issues & Fixes

| Problem                       | Quick Fix                                                             |
| ----------------------------- | --------------------------------------------------------------------- |
| 🐌 **Slow responses**         | Use Groq API (fastest): `$env:GROQ_API_KEY="your_key"`                |
| 🔑 **"API key not provided"** | Set in same terminal: `$env:GROQ_API_KEY="gsk_..."` then run app      |
| 🚫 **Rate limit exceeded**    | Switch to Ollama (unlimited): just run `ollama pull llama3`           |
| ❌ **"Module not found"**     | Activate venv: `.\.venv\Scripts\Activate.ps1` then reinstall          |
| 📁 **"ChromaDB not found"**   | Initialize: `python scripts/init_vector_store.py`                     |
| 📊 **"Invalid file format"**  | Check CSV columns: `Activity`, `Avg_CO2_Emission(kg/day)`, `Category` |
| 🔌 **Port already in use**    | Try different port: `streamlit run app.py --server.port 8502`         |

**Still stuck?** Check the detailed troubleshooting in the [wiki](#) or open an issue.

## 📚 Additional Resources

- **[User Guide](docs/USER_GUIDE.md)** - Detailed usage instructions

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Make changes and test thoroughly
4. Submit a pull request with clear description

---

## 📄 License

[Specify your license here]

---

## 🙏 Acknowledgments

Built with powerful open-source tools:

- **Groq** - Ultra-fast LLM inference
- **ChromaDB** - Vector database
- **Streamlit** - Web UI framework
- **SentenceTransformers** - Embeddings
- **LangChain** - RAG orchestration

---

## 📞 Support

Need help?

- 📖 Check the [Common Issues](#-common-issues--fixes) section
- 🐛 Found a bug? [Open an issue](../../issues)
- 💬 Questions? Start a [discussion](../../discussions)

---

**Made with 💚 for a sustainable future**
