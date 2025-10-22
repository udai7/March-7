# Running the CO₂ Reduction AI Agent

## Quick Start

1. **Ensure all dependencies are installed:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure Ollama is running with the required model:**

   ```bash
   ollama serve
   ollama pull llama3
   ```

3. **Initialize the vector store (if not already done):**

   ```bash
   python scripts/init_vector_store.py
   ```

4. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

5. **Access the application:**
   - The app will automatically open in your browser
   - Default URL: http://localhost:8501

## Features

### 💬 Ask a Question Tab

- Enter natural language queries about CO₂ reduction
- Get personalized recommendations based on your activities
- View emission comparisons and annual savings projections
- See example questions for guidance

### 📊 Upload Dataset Tab

- Upload CSV or Excel files with your activity data
- Required columns: Activity, Avg_CO2_Emission(kg/day), Category
- Get comprehensive analysis of your carbon footprint
- Identify top emitters and receive prioritized recommendations
- View category breakdowns and visualizations

## System Status

The sidebar shows the health status of all system components:

- ✅ LLM Service (Ollama)
- ✅ Vector Store (ChromaDB)
- ✅ Reference Data

## Troubleshooting

### LLM Service Unavailable

- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Pull the model if needed: `ollama pull llama3`

### File Upload Issues

- Verify file format (CSV or Excel)
- Check required columns are present
- Ensure emission values are numeric
- Verify categories are: Transport, Household, Food, or Lifestyle

### Vector Store Issues

- Run initialization script: `python scripts/init_vector_store.py`
- Check that chroma_db directory exists
- Verify sustainability_tips.txt is in the data folder
