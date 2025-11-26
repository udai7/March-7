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
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.3"))  # Minimum similarity score to consider relevant (lowered for better coverage)
MIN_RELEVANT_DOCS = int(os.getenv("MIN_RELEVANT_DOCS", "1"))  # Minimum relevant docs needed to answer

# Vector Store Settings
VECTOR_DB_PATH = str(CHROMA_DIR)
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"  # More stable alternative
RETRIEVAL_TOP_K = 5

# Data Settings
REFERENCE_DATA_PATH = str(DATA_DIR / "reference_activities.csv")
SUSTAINABILITY_TIPS_PATH = str(DATA_DIR / "sustainability_tips.txt")
ENVIRONMENTAL_TIPS_PATH = str(DATA_DIR / "environmental_sustainability_tips.txt")

# Environmental Metrics Settings
ENABLE_MULTI_METRIC_ANALYSIS = True  # Enable comprehensive environmental analysis beyond CO2
SHOW_WATER_METRICS = True
SHOW_ENERGY_METRICS = True
SHOW_WASTE_METRICS = True
SHOW_POLLUTION_INDEX = True
CALCULATE_ENVIRONMENTAL_SCORE = True

# Sustainability Grading Thresholds (daily per-person values)
GRADE_THRESHOLDS = {
    "A+": {"co2": 2.0, "water": 100, "energy": 3.0, "waste": 0.5},
    "A": {"co2": 3.5, "water": 150, "energy": 5.0, "waste": 1.0},
    "B": {"co2": 5.5, "water": 250, "energy": 8.0, "waste": 1.5},
    "C": {"co2": 8.0, "water": 400, "energy": 12.0, "waste": 2.5},
    "D": {"co2": 12.0, "water": 600, "energy": 18.0, "waste": 4.0},
    "F": {"co2": float('inf'), "water": float('inf'), "energy": float('inf'), "waste": float('inf')}
}

# UI Settings
PAGE_TITLE = "Environmental Impact AI Agent"
PAGE_ICON = "🌍"
MAX_UPLOAD_SIZE_MB = 10

# Visualization Settings
CHART_COLORS = {
    "co2": "#E74C3C",  # Red
    "water": "#3498DB",  # Blue
    "energy": "#F39C12",  # Orange
    "waste": "#95A5A6",  # Gray
    "improvement": "#27AE60"  # Green
}

# Performance Settings
EMBEDDING_CACHE_SIZE = 1000
RESPONSE_CACHE_TTL = 3600  # 1 hour in seconds

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(BASE_DIR / "app.log")
