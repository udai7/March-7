"""
Embedding generation module for converting text to vector representations.
Uses SentenceTransformers for efficient semantic embeddings.
"""

from functools import lru_cache
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings for text using SentenceTransformer models.
    Includes caching mechanism for improved performance.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with specified model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    @lru_cache(maxsize=1000)
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string with caching.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts with batching for efficiency.
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)
        
        if not valid_texts:
            # Return zero vectors for all texts
            return np.zeros((len(texts), self.embedding_dim))
        
        # Generate embeddings for valid texts
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Create result array with zero vectors for empty texts
        result = np.zeros((len(texts), self.embedding_dim))
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = embeddings[idx]
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.generate_embedding.cache_clear()
    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = self.generate_embedding.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize
        }
