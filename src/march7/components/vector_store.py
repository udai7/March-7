"""
Lightweight Vector store replacement using TF-IDF and Scikit-learn.
Removes dependency on ChromaDB, Torch, and other heavy libraries.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class Document:
    """Represents a document with content and metadata."""
    
    def __init__(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        self.id = doc_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


class VectorStore:
    """
    Lightweight replacement for ChromaDB.
    Uses TF-IDF + Cosine Similarity for semantic search.
    Very fast and has zero heavy dependencies.
    """
    
    def __init__(
        self, 
        collection_name: str = "sustainability_tips",
        persist_directory: str = "./lean_db",
        embedding_model: str = "tfidf"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.documents: List[Document] = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        self.load()

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and update the search index."""
        self.documents.extend(documents)
        self._update_index()
        self.save()

    def _update_index(self):
        """Re-calculate the TF-IDF matrix."""
        if not self.documents:
            return
        contents = [doc.content for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)

    def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search using TF-IDF."""
        if not self.documents or not query:
            return []
            
        if self.tfidf_matrix is None:
            self._update_index()
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Combine with indices and sort
        results = []
        for idx, score in enumerate(similarities):
            # Apply metadata filters if any
            if filter_metadata:
                match = True
                for key, val in filter_metadata.items():
                    if self.documents[idx].metadata.get(key) != val:
                        match = False
                        break
                if not match:
                    continue
            
            doc = self.documents[idx]
            doc.score = float(score)
            results.append(doc)
            
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def search_with_filters(self, query: str, k: int = 5, **filters) -> List[Document]:
        """Mimics the original search_with_filters but lightweight."""
        clean_filters = {k: v for k, v in filters.items() if v is not None}
        return self.search(query, k=k, filter_metadata=clean_filters)

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "persist_directory": self.persist_directory
        }

    def clear_collection(self) -> None:
        self.documents = []
        self.tfidf_matrix = None
        self.save()

    def save(self):
        """Save the documents to disk."""
        path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.documents, f)

    def load(self):
        """Load documents from disk."""
        path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.documents = pickle.load(f)
                self._update_index()
