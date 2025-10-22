"""
Configuration settings for CO2 Reduction AI Agent
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# LLM Settings
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")  # or "mistral"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

# Vector Store Settings
VECTOR_DB_PATH = str(CHROMA_DIR)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
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
