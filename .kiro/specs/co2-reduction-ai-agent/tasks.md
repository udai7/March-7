# Implementation Plan

- [x] 1. Set up project structure and dependencies

  - Create project directory structure with folders for data, models, components, and tests
  - Create requirements.txt with all necessary dependencies (streamlit, langchain, chromadb, sentence-transformers, pandas, pydantic, ollama)
  - Create config.py for centralized configuration management
  - Create .gitignore file to exclude virtual environments and data files
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 2. Prepare reference dataset and sustainability knowledge base

  - [x] 2.1 Create reference activities CSV file

    - Create data/reference_activities.csv with the 10 sample activities provided
    - Include columns: Activity, Avg_CO2_Emission(kg/day), Category
    - Add 10-15 additional common activities for broader coverage
    - _Requirements: 7.1, 7.2_

  - [x] 2.2 Create sustainability tips knowledge base

    - Create data/sustainability_tips.txt with actionable eco-friendly practices
    - Organize tips by category (Transport, Household, Food, Lifestyle)
    - Include emission reduction potential and implementation difficulty for each tip
    - Add at least 30 diverse sustainability tips covering all categories
    - _Requirements: 3.2_

- [x] 3. Implement data management components

  - [x] 3.1 Create data models using Pydantic

    - Implement Activity, Recommendation, AgentResponse, DatasetAnalysis models in models/data_models.py
    - Add validation rules for emission values (must be >= 0)
    - Implement Category enum with Transport, Household, Food, Lifestyle
    - _Requirements: 7.3, 8.2_

  - [x] 3.2 Implement reference data manager

    - Create components/reference_data.py with ReferenceDataManager class
    - Implement load_reference_data() to read CSV file
    - Implement get_activity_emission() for activity lookup with fuzzy matching
    - Implement get_activities_by_category() for category filtering
    - _Requirements: 7.4, 1.2_

  - [x] 3.3 Implement data validator

    - Create components/data_validator.py with DataValidator class
    - Implement validate_schema() to check required columns
    - Implement validate_values() to check data types and ranges
    - Implement sanitize_data() to clean and normalize input data
    - _Requirements: 2.2, 10.2_

  - [x] 3.4 Implement emission calculator

    - Create components/emission_calculator.py with EmissionCalculator class
    - Implement calculate_daily_emission() for total daily CO₂
    - Implement calculate_annual_emission() (daily × 365)
    - Implement calculate_reduction() for absolute and percentage reductions
    - _Requirements: 4.1, 4.2, 4.5_

- [x] 4. Implement vector store and embeddings

  - [x] 4.1 Set up embedding generator

    - Create components/embeddings.py with EmbeddingGenerator class
    - Initialize SentenceTransformer with all-MiniLM-L6-v2 model
    - Implement generate_embedding() for single text
    - Implement generate_batch_embeddings() for multiple texts with batching
    - Add embedding caching mechanism using functools.lru_cache
    - _Requirements: 3.1, 9.3_

  - [x] 4.2 Implement vector store wrapper

    - Create components/vector_store.py with VectorStore class
    - Initialize ChromaDB client with persistent storage
    - Implement add_documents() to store embeddings with metadata
    - Implement search() for semantic similarity search with top-k retrieval
    - Implement get_collection_stats() for monitoring
    - _Requirements: 3.2, 3.3, 9.3_

  - [x] 4.3 Create knowledge base loader

    - Create components/knowledge_loader.py with load_sustainability_tips()
    - Parse sustainability_tips.txt and extract tips with metadata
    - Generate embeddings for all tips using EmbeddingGenerator
    - Store tips in ChromaDB vector store with category metadata
    - _Requirements: 3.4, 3.5_

- [x] 5. Implement LLM integration

  - [x] 5.1 Create LLM client abstraction

    - Create components/llm_client.py with LLMClient class
    - Support Ollama API endpoint (http://localhost:11434)
    - Implement generate() method with error handling and retries
    - Implement generate_with_context() for RAG-based generation
    - Implement check_availability() to verify LLM service is running
    - _Requirements: 6.1, 1.4, 10.3_

  - [x] 5.2 Create prompt templates

    - Create components/prompt_templates.py with PromptTemplates class
    - Implement recommendation_prompt() for generating alternatives
    - Implement analysis_prompt() for dataset analysis
    - Implement comparison_prompt() for emission comparisons
    - Use structured prompts with clear instructions and examples
    - _Requirements: 1.3, 4.3, 8.1_

  - [x] 5.3 Implement response parser

    - Create components/response_parser.py with parse_llm_response()
    - Extract structured recommendations from LLM text output
    - Parse emission values and reduction percentages
    - Handle malformed responses gracefully with fallbacks
    - _Requirements: 8.5, 10.4_

- [x] 6. Implement agent orchestration

  - [x] 6.1 Create query processor

    - Create components/query_processor.py with QueryProcessor class
    - Implement extract_activities() to identify activities in user query
    - Implement identify_intent() to determine query type (single activity, comparison, general advice)
    - Implement extract_parameters() to get emission values, distances, etc.
    - _Requirements: 1.1, 1.2_

  - [x] 6.2 Create dataset analyzer

    - Create components/dataset_analyzer.py with DatasetAnalyzer class
    - Implement validate_dataset() using DataValidator
    - Implement calculate_total_emissions() for uploaded data
    - Implement identify_top_emitters() to find highest emission activities
    - _Requirements: 2.3, 2.4, 2.5_

  - [x] 6.3 Create recommendation generator

    - Create components/recommendation_generator.py with RecommendationGenerator class
    - Implement generate_alternatives() to find lower-emission options
    - Implement rank_recommendations() by emission reduction potential
    - Implement format_recommendations() to structure output
    - _Requirements: 4.3, 8.1, 8.2_

  - [x] 6.4 Implement main agent class

    - Create components/agent.py with CO2ReductionAgent class
    - Initialize with LLM client, vector store, and reference data
    - Implement process_query() to handle text queries end-to-end
    - Implement analyze_dataset() to process uploaded CSV/Excel files
    - Implement generate_recommendations() combining retrieval and generation
    - Wire together query processor, vector store retrieval, and LLM generation
    - _Requirements: 1.1, 1.3, 1.5, 3.4_

-

- [x] 7. Implement Streamlit user interface

  - [x] 7.1 Create main application file

    - Create app.py as Streamlit entry point
    - Set page configuration (title, icon, layout)
    - Initialize session state for conversation history
    - Load and cache agent components (LLM, vector store, reference data)
    - _Requirements: 5.1, 6.5_

  - [x] 7.2 Implement query input interface

    - Create text input widget for user queries
    - Add submit button with keyboard shortcut (Enter)
    - Display example queries in an expander section
    - Show processing spinner during query execution
    - _Requirements: 5.1, 5.4, 5.5_

  - [x] 7.3 Implement file upload interface

    - Create file uploader widget accepting CSV and Excel formats
    - Display upload instructions and file format requirements
    - Show preview of uploaded data (first 5 rows)
    - Add validation feedback for uploaded files
    - _Requirements: 5.2, 2.1, 2.2_

  - [x] 7.4 Implement results display

    - Create structured layout for displaying agent responses
    - Show current emission with visual indicator (metric widget)
    - Display recommendations in expandable cards
    - Show emission comparisons with bar charts using st.bar_chart()
    - Display annual savings projection prominently
    - _Requirements: 5.3, 4.4, 1.3_

  - [x] 7.5 Implement error handling UI

    - Create error message display function with st.error()
    - Show user-friendly error messages for common issues
    - Add retry button for transient errors
    - Display validation errors with specific guidance
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 8. Implement error handling and logging

  - [x] 8.1 Create error handler utility

    - Create utils/error_handler.py with ErrorHandler class
    - Implement handle_file_upload_error() for file validation errors
    - Implement handle_llm_error() with retry logic and exponential backoff
    - Implement handle_vector_store_error() for database errors
    - Implement log_error() to write errors to log file with context
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 8.2 Add logging configuration

    - Create utils/logger.py with logging setup
    - Configure file and console logging handlers
    - Set appropriate log levels (INFO for production, DEBUG for development)
    - Add structured logging with timestamps and context
    - _Requirements: 10.5_

- [x] 9. Create initialization and setup scripts

  - [x] 9.1 Create vector store initialization script

    - Create scripts/init_vector_store.py
    - Load sustainability tips from data/sustainability_tips.txt
    - Generate embeddings and populate ChromaDB
    - Verify vector store is properly initialized
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 9.2 Create setup script

    - Create setup.sh (or setup.bat for Windows) for initial setup
    - Check Python version (3.9+)
    - Create virtual environment
    - Install dependencies from requirements.txt
    - Run vector store initialization
    - Verify Ollama is installed and model is available
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 10. Add example data and documentation



  - [x] 10.1 Create example queries file

    - Create data/example_queries.txt with 10-15 sample queries

    - Cover different query types (single activity, comparison, general advice)
    - Include queries for all categories
    - _Requirements: 5.4_

  - [x] 10.2 Create README documentation

    - Create README.md with project overview and features
    - Add installation instructions (prerequisites, setup steps)
    - Add usage instructions with screenshots
    - Document configuration options
    - Add troubleshooting section for common issues
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 10.3 Create user guide


    - Create docs/USER_GUIDE.md with detailed usage instructions
    - Explain how to formulate effective queries
    - Provide guidance on uploading and formatting datasets
    - Include interpretation guide for results
    - _Requirements: 5.4, 8.4_

- [ ] 11. Integration and end-to-end testing

  - [ ] 11.1 Test query processing workflow

    - Test with sample query: "I drive 20 km daily using a petrol car"
    - Verify activity extraction and emission calculation
    - Verify retrieval of relevant sustainability tips
    - Verify LLM generates appropriate recommendations
    - Verify response includes quantitative comparisons
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ] 11.2 Test dataset upload workflow

    - Test with sample 10-row CSV dataset
    - Verify file validation and error handling
    - Verify total emission calculation
    - Verify top emitters identification
    - Verify recommendations are prioritized correctly
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 11.3 Test error scenarios

    - Test with invalid file format (e.g., .txt file)
    - Test with CSV missing required columns
    - Test with LLM service unavailable
    - Test with malformed query
    - Verify appropriate error messages are displayed
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ] 11.4 Test performance requirements
    - Measure query response time (should be < 5 seconds)
    - Measure dataset processing time for 100 activities (should be < 10 seconds)
    - Measure vector search latency (should be < 1 second)
    - Test with concurrent users (up to 5)
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 12. Final polish and deployment preparation

  - [ ] 12.1 Add configuration validation

    - Verify all required configuration values are set
    - Check file paths exist and are accessible
    - Verify LLM service is reachable
    - Display helpful error messages for configuration issues
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 12.2 Optimize performance

    - Implement embedding caching for common queries
    - Add response caching for identical queries
    - Optimize pandas operations for dataset processing
    - Profile and optimize slow operations
    - _Requirements: 9.1, 9.2, 9.3, 9.5_

  - [ ] 12.3 Create deployment package

    - Create Dockerfile for containerized deployment
    - Create docker-compose.yml including Ollama service
    - Test Docker deployment locally
    - Create deployment documentation
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 12.4 Final validation
    - Run through all example queries and verify results
    - Test with various dataset sizes and formats
    - Verify all error handling paths work correctly
    - Confirm response quality and accuracy
    - Validate against all acceptance criteria
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
