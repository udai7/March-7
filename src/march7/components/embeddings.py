"""
Dummy embedding module to maintain compatibility after de-bloating.
Actual search is now handled internally by the lean VectorStore using TF-IDF.
"""

from typing import List
import numpy as np

class EmbeddingGenerator:
    """
    Dummy class to maintain compatibility.
    No longer needed for the lean implementation.
    """
    
    def __init__(self, model_name: str = "dummy"):
        self.embedding_dim = 384 # Standard size
    
    def generate_embedding(self, text: str) -> np.ndarray:
        return np.zeros(self.embedding_dim)
    
    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), self.embedding_dim))
