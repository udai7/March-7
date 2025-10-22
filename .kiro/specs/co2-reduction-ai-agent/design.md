# Design Document

## Overview

The CO₂ Reduction AI Agent is a Retrieval-Augmented Generation (RAG) system that helps users identify high-emission activities and provides actionable recommendations to reduce their carbon footprint. The system combines a vector database of sustainability knowledge with an open-source LLM to generate personalized, context-aware responses.

The architecture follows a modular design with clear separation between data ingestion, retrieval, generation, and presentation layers. All components use open-source technologies to ensure reproducibility and cost-effectiveness.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Query Input  │  │ File Upload  │  │ Results View │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Agent Orchestration Layer                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         LangChain/LlamaIndex Agent Workflow          │   │
│  │  • Query Processing  • Context Assembly              │   │
│  │  • Dataset Analysis  • Response Generation           │   │
│  └──────────────────────────────────────────────────────┘   │
└───────┬──────────────────────────┬──────────────────────────┘
        │                          │
┌───────▼──────────┐      ┌────────▼─────────────────────────┐
│  Vector Store    │      │    LLM Service                    │
│  (ChromaDB)      │      │    (Ollama/HuggingFace)          │
│                  │      │                                   │
│  • Embeddings    │      │  • LLaMA 3 / Mistral             │
│  • Sustainability│      │  • Text Generation                │
│    Tips          │      │  • Reasoning                      │
└───────▲──────────┘      └───────────────────────────────────┘
        │
┌───────┴──────────────────────────────────────────────────────┐
│                    Data Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Reference    │  │ Sustainability│  │ User Uploaded│       │
│  │ Dataset CSV  │  │ Tips KB      │  │ Data         │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

### Component Architecture

1. **Presentation Layer (Streamlit)**

   - Handles user interactions
   - Displays results and visualizations
   - Manages file uploads

2. **Agent Layer (LangChain/LlamaIndex)**

   - Orchestrates the RAG workflow
   - Processes queries and datasets
   - Assembles context for LLM

3. **Retrieval Layer (ChromaDB + SentenceTransformers)**

   - Stores and retrieves sustainability knowledge
   - Performs semantic search
   - Manages embeddings

4. **Generation Layer (Ollama/HuggingFace)**

   - Generates natural language responses
   - Performs reasoning over retrieved context
   - Formats recommendations

5. **Data Layer**
   - Reference dataset of activities and emissions
   - Sustainability tips knowledge base
   - User-uploaded data processing

## Components and Interfaces

### 1. Streamlit UI Component

**Purpose:** Provide user-friendly web interface for interaction

**Key Modules:**

- `app.py`: Main Streamlit application entry point
- `ui_components.py`: Reusable UI widgets and layouts

**Interfaces:**

```python
class StreamlitUI:
    def render_query_input() -> str
    def render_file_upload() -> pd.DataFrame
    def render_results(response: AgentResponse) -> None
    def render_examples() -> None
    def show_error(message: str) -> None
```

**Design Decisions:**

- Use Streamlit's native components for rapid development
- Implement session state for maintaining conversation context
- Use columns layout for organized information display
- Include example queries to guide users

### 2. Agent Orchestrator Component

**Purpose:** Coordinate the RAG workflow and manage agent logic

**Key Modules:**

- `agent.py`: Main agent class implementing the workflow
- `query_processor.py`: Parse and understand user queries
- `dataset_analyzer.py`: Process uploaded datasets
- `recommendation_generator.py`: Generate structured recommendations

**Interfaces:**

```python
class CO2ReductionAgent:
    def __init__(self, llm, vector_store, reference_data)
    def process_query(self, query: str) -> AgentResponse
    def analyze_dataset(self, df: pd.DataFrame) -> DatasetAnalysis
    def generate_recommendations(self, context: Context) -> List[Recommendation]

class QueryProcessor:
    def extract_activities(self, query: str) -> List[str]
    def identify_intent(self, query: str) -> QueryIntent
    def extract_parameters(self, query: str) -> Dict[str, Any]

class DatasetAnalyzer:
    def validate_dataset(self, df: pd.DataFrame) -> ValidationResult
    def calculate_total_emissions(self, df: pd.DataFrame) -> float
    def identify_top_emitters(self, df: pd.DataFrame, n: int) -> List[Activity]
```

**Design Decisions:**

- Use LangChain's Agent framework for structured workflow
- Implement chain-of-thought reasoning for complex queries
- Separate concerns: parsing, retrieval, generation
- Use Pydantic models for type safety and validation

### 3. Vector Store Component

**Purpose:** Store and retrieve sustainability knowledge using semantic search

**Key Modules:**

- `vector_store.py`: ChromaDB wrapper and management
- `embeddings.py`: Embedding generation using SentenceTransformers
- `knowledge_loader.py`: Load and index sustainability tips

**Interfaces:**

```python
class VectorStore:
    def __init__(self, collection_name: str, embedding_model: str)
    def add_documents(self, documents: List[Document]) -> None
    def search(self, query: str, k: int) -> List[Document]
    def update_document(self, doc_id: str, document: Document) -> None
    def get_collection_stats() -> Dict[str, Any]

class EmbeddingGenerator:
    def __init__(self, model_name: str)
    def generate_embedding(self, text: str) -> np.ndarray
    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray
```

**Design Decisions:**

- Use ChromaDB for lightweight, embedded vector database
- Use all-MiniLM-L6-v2 for fast, efficient embeddings
- Implement document metadata for filtering and ranking
- Cache embeddings to improve performance
- Store sustainability tips with categories and emission reduction potential

### 4. LLM Integration Component

**Purpose:** Interface with open-source LLMs for text generation

**Key Modules:**

- `llm_client.py`: Abstraction layer for LLM services
- `prompt_templates.py`: Structured prompts for different tasks
- `response_parser.py`: Parse and structure LLM outputs

**Interfaces:**

```python
class LLMClient:
    def __init__(self, model_name: str, base_url: str)
    def generate(self, prompt: str, max_tokens: int) -> str
    def generate_with_context(self, query: str, context: List[str]) -> str
    def check_availability() -> bool

class PromptTemplates:
    @staticmethod
    def recommendation_prompt(activity: str, emission: float, alternatives: List[str]) -> str
    @staticmethod
    def analysis_prompt(activities: List[Activity]) -> str
    @staticmethod
    def comparison_prompt(current: Activity, alternatives: List[Activity]) -> str
```

**Design Decisions:**

- Support both Ollama and HuggingFace Inference endpoints
- Use structured prompts with clear instructions
- Implement retry logic for robustness
- Parse LLM outputs into structured data models
- Use temperature=0.7 for balanced creativity and accuracy

### 5. Data Management Component

**Purpose:** Handle reference data and user uploads

**Key Modules:**

- `reference_data.py`: Load and manage reference dataset
- `data_validator.py`: Validate user-uploaded data
- `emission_calculator.py`: Calculate CO₂ emissions

**Interfaces:**

```python
class ReferenceDataManager:
    def load_reference_data(self, filepath: str) -> pd.DataFrame
    def get_activity_emission(self, activity: str) -> Optional[float]
    def get_activities_by_category(self, category: str) -> List[Activity]
    def search_similar_activities(self, query: str) -> List[Activity]

class DataValidator:
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult
    def validate_values(self, df: pd.DataFrame) -> ValidationResult
    def sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame

class EmissionCalculator:
    def calculate_daily_emission(self, activities: List[Activity]) -> float
    def calculate_annual_emission(self, daily: float) -> float
    def calculate_reduction(self, current: float, alternative: float) -> ReductionMetrics
```

**Design Decisions:**

- Store reference data in CSV for easy updates
- Implement fuzzy matching for activity names
- Validate data types and ranges
- Calculate both absolute and percentage reductions
- Support multiple emission units (kg, tons)

## Data Models

### Core Data Structures

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class Category(str, Enum):
    TRANSPORT = "Transport"
    HOUSEHOLD = "Household"
    FOOD = "Food"
    LIFESTYLE = "Lifestyle"

class Activity(BaseModel):
    name: str
    emission_kg_per_day: float
    category: Category
    description: Optional[str] = None

class Recommendation(BaseModel):
    action: str
    emission_reduction_kg: float
    reduction_percentage: float
    implementation_difficulty: str  # "Easy", "Medium", "Hard"
    timeframe: str  # "Immediate", "Short-term", "Long-term"
    additional_benefits: List[str] = []

class AgentResponse(BaseModel):
    current_emission: float
    recommendations: List[Recommendation]
    total_potential_reduction: float
    annual_savings_kg: float
    summary: str

class DatasetAnalysis(BaseModel):
    total_daily_emission: float
    total_annual_emission: float
    top_emitters: List[Activity]
    category_breakdown: Dict[Category, float]
    recommendations: List[Recommendation]

class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
```

## Error Handling

### Error Categories and Strategies

1. **User Input Errors**

   - Invalid file format → Display supported formats
   - Missing columns → Show required schema
   - Invalid values → Highlight problematic rows
   - Strategy: Validate early, provide clear feedback

2. **LLM Service Errors**

   - Connection timeout → Retry with exponential backoff
   - Model not available → Fallback to alternative model
   - Rate limiting → Queue requests
   - Strategy: Graceful degradation, user notification

3. **Vector Store Errors**

   - Collection not found → Initialize automatically
   - Embedding generation failure → Use cached embeddings
   - Search timeout → Return partial results
   - Strategy: Automatic recovery, logging

4. **Data Processing Errors**
   - Parsing errors → Skip invalid rows, log warnings
   - Calculation errors → Use default values, flag uncertainty
   - Memory errors → Process in batches
   - Strategy: Partial success, detailed error reporting

### Error Handling Implementation

```python
class ErrorHandler:
    @staticmethod
    def handle_file_upload_error(error: Exception) -> str:
        """Return user-friendly error message for file upload issues"""

    @staticmethod
    def handle_llm_error(error: Exception) -> str:
        """Handle LLM service errors with retry logic"""

    @staticmethod
    def handle_vector_store_error(error: Exception) -> str:
        """Handle vector database errors"""

    @staticmethod
    def log_error(error: Exception, context: Dict) -> None:
        """Log errors with context for debugging"""
```

## Testing Strategy

### Unit Testing

**Components to Test:**

- Query processor: Test activity extraction, intent classification
- Emission calculator: Test calculation accuracy
- Data validator: Test schema and value validation
- Embedding generator: Test embedding generation and caching

**Tools:** pytest, pytest-mock

### Integration Testing

**Workflows to Test:**

- End-to-end query processing
- Dataset upload and analysis
- Vector store retrieval accuracy
- LLM response generation

**Tools:** pytest, pytest-asyncio

### Performance Testing

**Metrics to Measure:**

- Query response time (target: <5 seconds)
- Dataset processing time (target: <10 seconds for 100 rows)
- Vector search latency (target: <1 second)
- Memory usage under load

**Tools:** pytest-benchmark, memory_profiler

### User Acceptance Testing

**Scenarios:**

- New user with no prior knowledge
- User uploading personal activity data
- User asking various types of questions
- Error scenarios (invalid data, service unavailable)

**Validation:**

- Response accuracy against reference data
- Recommendation relevance and actionability
- UI usability and clarity

## Deployment Considerations

### Local Development Setup

1. Install Python 3.9+
2. Install Ollama and pull LLaMA 3 or Mistral model
3. Install dependencies: `pip install -r requirements.txt`
4. Initialize vector store with sustainability tips
5. Run Streamlit app: `streamlit run app.py`

### Production Deployment Options

**Option 1: Docker Container**

- Package application with all dependencies
- Include Ollama in container or use external service
- Mount data volumes for persistence

**Option 2: Cloud Deployment (Streamlit Cloud)**

- Deploy UI to Streamlit Cloud
- Use external Ollama instance or HuggingFace Inference
- Store vector database in persistent storage

**Option 3: Local Server**

- Deploy on local server with GPU for faster inference
- Use nginx as reverse proxy
- Implement basic authentication

### Configuration Management

```python
# config.py
class Config:
    # LLM Settings
    LLM_MODEL = "llama3"  # or "mistral"
    LLM_BASE_URL = "http://localhost:11434"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 500

    # Vector Store Settings
    VECTOR_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RETRIEVAL_TOP_K = 5

    # Data Settings
    REFERENCE_DATA_PATH = "./data/reference_activities.csv"
    SUSTAINABILITY_TIPS_PATH = "./data/sustainability_tips.txt"

    # UI Settings
    PAGE_TITLE = "CO₂ Reduction AI Agent"
    MAX_UPLOAD_SIZE_MB = 10
```

## Security Considerations

1. **Input Validation**

   - Sanitize all user inputs
   - Validate file uploads (size, type, content)
   - Prevent injection attacks in queries

2. **Data Privacy**

   - Do not store user-uploaded data permanently
   - Clear session data after use
   - No external API calls with user data

3. **Resource Limits**

   - Limit file upload size (10 MB)
   - Limit query length (500 characters)
   - Implement rate limiting for requests

4. **Dependency Security**
   - Use pinned dependency versions
   - Regular security audits
   - Keep dependencies updated

## Performance Optimization

### Caching Strategy

1. **Embedding Cache**

   - Cache generated embeddings for common queries
   - Use LRU cache with 1000 entry limit
   - Persist cache to disk

2. **LLM Response Cache**

   - Cache responses for identical queries
   - TTL: 1 hour
   - Clear cache on reference data updates

3. **Reference Data Cache**
   - Load reference data once at startup
   - Keep in memory for fast access
   - Reload only on explicit update

### Batch Processing

- Process multiple activities in single LLM call
- Generate embeddings in batches of 32
- Vectorize dataset operations with pandas

### Resource Management

- Lazy load models (load on first use)
- Implement connection pooling for LLM service
- Use streaming for large file uploads
- Implement pagination for large result sets

## Extensibility and Future Enhancements

### Planned Extensions

1. **Multi-language Support**

   - Translate UI and responses
   - Support non-English queries

2. **Advanced Analytics**

   - Historical tracking of user emissions
   - Progress visualization
   - Comparison with community averages

3. **Integration Capabilities**

   - API endpoints for external systems
   - Export recommendations to PDF
   - Calendar integration for action reminders

4. **Enhanced Knowledge Base**
   - Regional-specific recommendations
   - Industry-specific guidelines
   - Real-time data integration (energy prices, weather)

### Architecture for Extensibility

- Plugin system for new data sources
- Modular recommendation engines
- Configurable LLM backends
- Extensible category system
