# Requirements Document

## Introduction

This document outlines the requirements for a Generative AI Agent designed to help communities and individuals identify high CO₂ emission activities and generate actionable recommendations to reduce their carbon footprint. The system leverages Retrieval-Augmented Generation (RAG) with open-source LLMs to provide personalized sustainability suggestions based on user activities and queries.

## Glossary

- **AI Agent**: An intelligent system that processes user queries and data to generate CO₂ reduction recommendations
- **RAG (Retrieval-Augmented Generation)**: A technique combining information retrieval with text generation
- **Vector Database**: A database optimized for storing and querying embedding vectors (ChromaDB)
- **Embeddings**: Numerical representations of text that capture semantic meaning
- **LLM (Large Language Model)**: Pre-trained language model used for text generation
- **CO₂ Emission**: Carbon dioxide output measured in kilograms per day
- **Activity Dataset**: Structured data containing activities and their associated CO₂ emissions
- **Sustainability Tips**: Knowledge base of eco-friendly practices and recommendations
- **Streamlit UI**: Web-based user interface for interacting with the AI Agent

## Requirements

### Requirement 1: Query Processing and Response Generation

**User Story:** As a user, I want to ask natural language questions about reducing CO₂ emissions, so that I can receive personalized recommendations based on my activities.

#### Acceptance Criteria

1. WHEN a user enters a text query about CO₂ reduction, THE AI Agent SHALL process the query and generate relevant recommendations within 5 seconds
2. WHEN a user mentions a specific activity (e.g., "driving 20 km daily"), THE AI Agent SHALL identify the activity category and retrieve corresponding emission data
3. THE AI Agent SHALL provide quantitative comparisons showing potential CO₂ reduction percentages for suggested alternatives
4. THE AI Agent SHALL generate responses using the open-source LLM integrated via Ollama or Hugging Face
5. WHERE the query matches activities in the dataset, THE AI Agent SHALL include specific emission values in kg CO₂/day

### Requirement 2: Dataset Upload and Analysis

**User Story:** As a user, I want to upload my activity data in CSV/Excel format, so that the system can analyze my carbon footprint and provide tailored recommendations.

#### Acceptance Criteria

1. THE Streamlit UI SHALL provide a file upload component that accepts CSV and Excel file formats
2. WHEN a user uploads a dataset, THE AI Agent SHALL validate the file structure contains required columns (Activity, Avg_CO2_Emission, Category)
3. WHEN the dataset is valid, THE AI Agent SHALL process all activities and calculate total daily CO₂ emissions
4. THE AI Agent SHALL identify the top 3 highest emission activities from the uploaded dataset
5. THE AI Agent SHALL generate prioritized recommendations targeting the highest emission activities first

### Requirement 3: Vector Store and Knowledge Retrieval

**User Story:** As a system administrator, I want sustainability tips stored in a vector database, so that the agent can retrieve relevant information efficiently for user queries.

#### Acceptance Criteria

1. THE System SHALL convert sustainability tips into embeddings using SentenceTransformers (all-MiniLM-L6-v2 model)
2. THE System SHALL store embeddings in ChromaDB vector database with metadata including category and emission reduction potential
3. WHEN a user query is received, THE AI Agent SHALL generate query embeddings and retrieve the top 5 most relevant sustainability tips
4. THE AI Agent SHALL use retrieved tips as context for the LLM to generate comprehensive recommendations
5. THE Vector Store SHALL support incremental updates to add new sustainability tips without rebuilding the entire database

### Requirement 4: Emission Calculation and Comparison

**User Story:** As a user, I want to see quantitative comparisons of my current emissions versus recommended alternatives, so that I can make informed decisions about behavior changes.

#### Acceptance Criteria

1. THE AI Agent SHALL calculate current CO₂ emissions based on user-provided activities using the reference dataset
2. THE AI Agent SHALL compute emission reductions for each suggested alternative in both absolute values (kg CO₂/day) and percentages
3. WHEN multiple alternatives exist, THE AI Agent SHALL rank them by emission reduction potential from highest to lowest
4. THE AI Agent SHALL display comparisons in a clear, user-friendly format with visual indicators
5. THE AI Agent SHALL provide cumulative annual CO₂ savings projections (daily savings × 365 days)

### Requirement 5: User Interface and Interaction

**User Story:** As a user, I want a simple web interface where I can enter queries or upload data, so that I can easily interact with the AI agent without technical knowledge.

#### Acceptance Criteria

1. THE Streamlit UI SHALL provide a text input box for natural language queries with a submit button
2. THE Streamlit UI SHALL display a file upload widget supporting CSV and Excel formats with clear instructions
3. WHEN the AI Agent generates recommendations, THE UI SHALL display results in a structured format with sections for current emissions, alternatives, and action steps
4. THE UI SHALL provide example queries to guide users on how to interact with the system
5. THE UI SHALL display processing status indicators during query processing and dataset analysis

### Requirement 6: Open-Source Technology Integration

**User Story:** As a developer, I want the system built entirely with open-source tools, so that the solution is replicable, cost-effective, and not dependent on proprietary APIs.

#### Acceptance Criteria

1. THE System SHALL use open-source LLMs accessible via Ollama (e.g., LLaMA 3, Mistral) or Hugging Face Inference
2. THE System SHALL implement the agent workflow using LangChain or LlamaIndex framework
3. THE System SHALL use ChromaDB as the vector database for storing embeddings
4. THE System SHALL use SentenceTransformers library for generating embeddings
5. THE System SHALL use Streamlit for the web-based user interface

### Requirement 7: Reference Dataset Management

**User Story:** As a system administrator, I want a pre-loaded reference dataset of common activities and their CO₂ emissions, so that the agent can provide accurate baseline information.

#### Acceptance Criteria

1. THE System SHALL include a reference CSV dataset with at least 10 common activities covering Transport, Household, Food, and Lifestyle categories
2. THE Reference Dataset SHALL contain columns: Activity, Avg_CO2_Emission(kg/day), and Category
3. THE System SHALL load the reference dataset on startup and make it available for query matching
4. THE System SHALL allow administrators to update the reference dataset without code changes
5. WHEN an activity in a user query matches the reference dataset, THE AI Agent SHALL use the exact emission values from the dataset

### Requirement 8: Response Quality and Accuracy

**User Story:** As a user, I want accurate and actionable recommendations, so that I can trust the system's suggestions and implement them effectively.

#### Acceptance Criteria

1. THE AI Agent SHALL provide at least 3 alternative actions for each high-emission activity identified
2. THE AI Agent SHALL include both short-term and long-term recommendations where applicable
3. THE AI Agent SHALL base emission calculations on the reference dataset values with a tolerance of ±5%
4. THE AI Agent SHALL include practical implementation steps for each recommendation
5. THE AI Agent SHALL avoid generating recommendations that are not supported by the knowledge base or reference dataset

### Requirement 9: Performance and Scalability

**User Story:** As a user, I want fast response times even when analyzing multiple activities, so that I can get immediate feedback on my carbon footprint.

#### Acceptance Criteria

1. THE AI Agent SHALL respond to simple text queries within 5 seconds on standard hardware
2. THE AI Agent SHALL process uploaded datasets with up to 100 activities within 10 seconds
3. THE Vector Store SHALL retrieve relevant sustainability tips within 1 second for any query
4. THE System SHALL handle concurrent requests from up to 5 users without performance degradation
5. THE System SHALL use efficient embedding caching to avoid redundant computations

### Requirement 10: Error Handling and Validation

**User Story:** As a user, I want clear error messages when something goes wrong, so that I can correct my input and successfully use the system.

#### Acceptance Criteria

1. WHEN a user uploads an invalid file format, THE System SHALL display an error message specifying accepted formats (CSV, Excel)
2. WHEN a dataset is missing required columns, THE System SHALL display an error message listing the required columns
3. WHEN the LLM service is unavailable, THE System SHALL display a user-friendly error message and suggest retry
4. WHEN a query cannot be processed, THE AI Agent SHALL provide a helpful message suggesting query reformulation
5. THE System SHALL log all errors with timestamps and context for debugging purposes
