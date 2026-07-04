# March7 - Environmental Impact AI Agent

An AI system that helps you understand and reduce your **complete environmental
footprint** through personalized, data-driven recommendations covering CO2
emissions, water usage, energy consumption, and waste generation.

## What It Does

Ask questions like _"I drive 20km daily, how can I reduce my environmental
impact?"_ or upload your activity data, and get:

- Comprehensive environmental analysis across CO2, water, energy, and waste metrics
- Personalized recommendations ranked by overall environmental impact
- Quantified savings in kg CO2/day, liters water, kWh energy, and waste reduction
- Sustainability grades (A+ to F) based on your environmental footprint
- Health and cost benefits alongside environmental improvements
- Source-backed advice from a curated sustainability knowledge base

## Key Features

| Feature                     | Description                                                                   |
| --------------------------- | ----------------------------------------------------------------------------- |
| CO2 Tracking                | Measure and reduce carbon emissions from all your activities                  |
| Water Footprint             | Track water consumption and get water-saving recommendations                  |
| Energy Analysis             | Monitor energy usage and optimize for efficiency                              |
| Waste Management            | Reduce waste generation with practical alternatives                           |
| Natural Language Queries    | Ask questions in plain English, get instant AI-powered answers                |
| Dataset Analysis            | Upload CSV/Excel files for comprehensive multi-metric analysis                |
| RAG-Powered Intelligence    | Combines vector search and LLM reasoning for accurate, grounded recommendations |
| Impact Quantification       | See precise reductions across all environmental metrics                       |
| Sustainability Grading      | Get an overall grade (A+ to F) based on your environmental performance        |
| Interactive Dashboard       | Clean Streamlit interface with multi-metric charts and visualizations         |
| Financial Calculator        | Calculate cost savings, ROI on green investments, and carbon credits          |
| Receipt Scanner             | Analyze purchase receipts for environmental impact of products                |

### Financial Impact Calculator

Calculate the financial benefits of your eco-friendly choices:

- **Cost Savings Calculator**: Savings from switching transport modes, reducing energy/water usage
- **Green Investment ROI**: Payback periods and returns for solar panels, EVs, heat pumps, and more
- **Utility Cost Comparison**: Compare current vs. optimized utility costs with detailed breakdowns
- **Carbon Credit Calculator**: Estimate your carbon credit earnings or tax liability

### Receipt and Product Scanner

Analyze your shopping to understand environmental impact:

- **Receipt Text Analysis**: Paste receipt text to auto-detect products and calculate impact
- **Manual Product Entry**: Add products individually for detailed environmental analysis
- **Category-Based Impact**: See CO2, water, and waste footprint by product category
- **Eco Recommendations**: Get personalized suggestions for greener alternatives
- **Sustainability Scoring**: Each product gets a 0-100 sustainability score

## Environmental Metrics Tracked

| Metric          | Unit       | Description                                     |
| --------------- | ---------- | ----------------------------------------------- |
| CO2 Emissions   | kg/day     | Carbon dioxide equivalent emissions             |
| Water Usage     | liters/day | Total water consumption including virtual water |
| Energy          | kWh/day    | Electricity and fuel energy consumption         |
| Waste           | kg/day     | Solid waste generation                          |
| Pollution Index | 0-100      | Combined air/water pollution score              |
| Land Use        | m2         | Land area required for activities               |

## Quick Start

### Prerequisites

- Python 3.10 or newer
- Internet connection (for the Groq API) or local compute (for Ollama)

### 1. Install

```bash
# Clone the repository
git clone https://github.com/udai7/march7.git
cd march7

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the project (editable, with dev extras)
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY (get a free key at https://console.groq.com)
```

### 3. Initialize and run

```bash
# Load the knowledge base into the vector store
python scripts/init_vector_store.py

# Verify setup (optional but recommended)
python scripts/verify_setup.py

# Launch the app
streamlit run src/march7/app.py
```

The app opens at `http://localhost:8501`.

**Using Ollama instead of Groq (offline):** install Ollama from
[ollama.ai](https://ollama.ai), run `ollama pull llama3`, then set
`LLM_PROVIDER=ollama` in your `.env` before launching.

## How to Use

### Ask Questions (Natural Language)

Type a question and get instant recommendations:

```
"I drive 20 km daily using a petrol car. How can I reduce emissions?"
"What's more eco-friendly: beef or chicken?"
"Top 3 ways to reduce household carbon footprint?"
```

### Upload Your Data (CSV/Excel)

Required format:

```csv
Activity,Avg_CO2_Emission(kg/day),Category
Driving petrol car 20km,4.6,Transport
Eating beef daily,3.3,Food
Electric heating 8hrs,2.5,Household
```

Categories: Transport, Household, Food, Lifestyle.

You get total daily and annual emissions, top emitting activities, ranked
recommendations by impact, and potential savings projections.

### Understanding Results

Each recommendation includes:

| Field          | Meaning                                         |
| -------------- | ----------------------------------------------- |
| Action         | What to do (e.g., "Switch to public transport") |
| Reduction      | CO2 saved per day (kg) and percentage           |
| Difficulty     | Easy / Medium / Hard                            |
| Timeframe      | Immediate / Short-term / Long-term              |
| Annual Savings | Total kg CO2/year if adopted                    |

## Tech Stack

| Component      | Technology                  | Purpose                                 |
| -------------- | --------------------------- | --------------------------------------- |
| LLM            | Groq / Ollama / HuggingFace | Text generation and reasoning           |
| RAG Framework  | LangChain                   | Retrieval-augmented generation pipeline |
| Vector Search  | TF-IDF + scikit-learn       | Lightweight semantic search (no GPU)    |
| Frontend       | Streamlit                   | Interactive web interface               |
| Data           | Pandas, Pydantic            | Processing and validation               |

The retrieval layer uses TF-IDF with cosine similarity, so the project runs
without heavy dependencies like Torch or ChromaDB.

## Project Structure

```
march7/
├── src/march7/                 # Application package
│   ├── app.py                  # Streamlit application entry point
│   ├── config.py               # Configuration settings
│   ├── components/             # Core application components
│   │   ├── agent.py            # Main agent orchestration
│   │   ├── llm_client.py       # LLM integration
│   │   ├── vector_store.py     # TF-IDF vector store
│   │   ├── embeddings.py       # Embedding compatibility shim
│   │   ├── query_processor.py  # Query parsing
│   │   ├── dataset_analyzer.py # Dataset analysis
│   │   ├── recommendation_generator.py
│   │   ├── recommendation_ranker.py
│   │   ├── emission_calculator.py
│   │   ├── data_validator.py
│   │   ├── response_validator.py
│   │   ├── response_parser.py
│   │   ├── reference_data.py
│   │   ├── knowledge_loader.py
│   │   ├── prompt_templates.py
│   │   ├── context_manager.py
│   │   ├── feedback_collector.py
│   │   ├── environmental_scorer.py
│   │   ├── financial_calculator.py
│   │   └── receipt_scanner.py
│   ├── models/                 # Pydantic data models
│   └── utils/                  # Logging and error handling
├── data/                       # Reference datasets and knowledge base
├── docs/                       # Documentation (USER_GUIDE.md)
├── scripts/                    # Setup and maintenance utilities
├── tests/                      # Test suite (pytest)
├── deploy/                     # Dockerfile and reverse-proxy config
├── .streamlit/                 # Streamlit theme and server config
├── pyproject.toml              # Packaging and tooling configuration
├── requirements.txt            # Runtime dependencies
├── .env.example                # Example environment configuration
├── CONTRIBUTING.md
└── LICENSE
```

## Deployment

A container image is defined in `deploy/Dockerfile`. Build and run from the
repository root:

```bash
docker build -f deploy/Dockerfile -t march7:latest .
docker run -d --name march7 --env-file .env -p 8501:8501 march7:latest
```

`deploy/streamlit.conf` contains a sample nginx reverse-proxy configuration for
serving the app behind a domain with TLS.

## Common Issues

| Problem                   | Fix                                                                  |
| ------------------------- | -------------------------------------------------------------------- |
| Slow responses            | Use the Groq provider (`LLM_PROVIDER=groq` in `.env`)                |
| "API key not provided"    | Set `GROQ_API_KEY` in `.env`                                         |
| Rate limit exceeded       | Switch to Ollama (`LLM_PROVIDER=ollama`, `ollama pull llama3`)       |
| "Module not found"        | Activate the venv and run `pip install -e ".[dev]"`                  |
| Vector store not found    | Initialize it: `python scripts/init_vector_store.py`                |
| "Invalid file format"     | Check CSV columns: `Activity`, `Avg_CO2_Emission(kg/day)`, `Category` |
| Port already in use       | Use another port: `streamlit run src/march7/app.py --server.port 8502` |

## Testing

```bash
pytest
```

## Additional Resources

- [User Guide](docs/USER_GUIDE.md) - Detailed usage instructions
- [Contributing Guide](CONTRIBUTING.md) - Development setup and conventions

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup
instructions and conventions. In short: fork the repo, create a feature branch,
make and test your changes, and open a pull request with a clear description.

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

Built with open-source tools: Groq (LLM inference), Streamlit (web UI),
scikit-learn (vector search), LangChain (RAG orchestration), and
Pandas/Pydantic (data processing).
