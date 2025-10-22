"""
Vector store wrapper for ChromaDB to manage embeddings and semantic search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid


class Document:
    """Represents a document with content and metadata."""
    
    def __init__(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        Initialize a document.
        
        Args:
            content: The text content of the document
            metadata: Optional metadata dictionary
            doc_id: Optional document ID (generated if not provided)
        """
        self.id = doc_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


class VectorStore:
    """
    Wrapper for ChromaDB vector database with persistent storage.
    Manages document storage, retrieval, and semantic search.
    """
    
    def __init__(
        self, 
        collection_name: str = "sustainability_tips",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store with ChromaDB.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory for persistent storage
            embedding_model: Name of the embedding model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection with embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Sustainability tips and recommendations"}
        )
    
    def add_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store with embeddings.
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
        """
        if not documents:
            return
        
        # Process in batches for efficiency
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc.id for doc in batch]
            contents = [doc.content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # ChromaDB will automatically generate embeddings
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform semantic similarity search for relevant documents.
        
        Args:
            query: Search query text
            k: Number of top results to return
            filter_metadata: Optional metadata filters (e.g., {"category": "Transport"})
            
        Returns:
            List of Document objects ranked by relevance
        """
        if not query or not query.strip():
            return []
        
        # Perform similarity search
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_metadata
        )
        
        # Convert results to Document objects
        documents = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                doc = Document(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    doc_id=results['ids'][0][i]
                )
                documents.append(doc)
        
        return documents
    
    def update_document(
        self, 
        doc_id: str, 
        document: Document
    ) -> None:
        """
        Update an existing document in the vector store.
        
        Args:
            doc_id: ID of the document to update
            document: New document data
        """
        self.collection.update(
            ids=[doc_id],
            documents=[document.content],
            metadatas=[document.metadata]
        )
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: ID of the document to delete
        """
        self.collection.delete(ids=[doc_id])
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        metadata = self.collection.metadata
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "metadata": metadata
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Sustainability tips and recommendations"}
        )
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document object if found, None otherwise
        """
        results = self.collection.get(ids=[doc_id])
        
        if results and results['ids']:
            return Document(
                content=results['documents'][0],
                metadata=results['metadatas'][0] if results['metadatas'] else {},
                doc_id=results['ids'][0]
            )
        
        return None
