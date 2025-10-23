"""
Configuration settings for CO2 Reduction AI Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "ollama", "huggingface", or "groq" (fastest!)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # For Groq: "llama-3.1-8b-instant", "mixtral-8x7b-32768"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")  # Only for Ollama
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")  # Get free key from huggingface.co
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Get free key from console.groq.com (FAST & FREE!)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # Lower for faster, more focused responses
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))  # Reduced for faster generation

# Relevance Settings
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))  # Minimum similarity score to consider relevant
MIN_RELEVANT_DOCS = int(os.getenv("MIN_RELEVANT_DOCS", "1"))  # Minimum relevant docs needed to answer

# Vector Store Settings
VECTOR_DB_PATH = str(CHROMA_DIR)
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"  # More stable alternative
RETRIEVAL_TOP_K = 5

# Data Settings
REFERENCE_DATA_PATH = str(DATA_DIR / "reference_activities.csv")
SUSTAINABILITY_TIPS_PATH = str(DATA_DIR / "sustainability_tips.txt")

# UI Settings
PAGE_TITLE = "CO₂ Reduction AI Agent"
PAGE_ICON = "🌱"
MAX_UPLOAD_SIZE_MB = 10

# Performance Settings
EMBEDDING_CACHE_SIZE = 1000
RESPONSE_CACHE_TTL = 3600  # 1 hour in seconds

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(BASE_DIR / "app.log")
