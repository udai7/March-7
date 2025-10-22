"""
Knowledge base loader for parsing and loading sustainability tips into vector store.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from components.vector_store import VectorStore, Document

if TYPE_CHECKING:
    from components.embeddings import EmbeddingGenerator


def parse_sustainability_tips(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse sustainability tips from text file.
    
    Args:
        filepath: Path to the sustainability tips text file
        
    Returns:
        List of dictionaries containing parsed tip data
    """
    tips = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by TIP: markers
    tip_sections = re.split(r'\n(?=TIP:)', content)
    
    for section in tip_sections:
        if not section.strip() or section.startswith('==='):
            continue
        
        # Extract fields using regex
        tip_match = re.search(r'TIP:\s*(.+?)(?=\n|$)', section)
        category_match = re.search(r'CATEGORY:\s*(.+?)(?=\n|$)', section)
        emission_match = re.search(r'EMISSION_REDUCTION:\s*(.+?)(?=\n|$)', section)
        difficulty_match = re.search(r'DIFFICULTY:\s*(.+?)(?=\n|$)', section)
        description_match = re.search(r'DESCRIPTION:\s*(.+?)(?=\n\n|$)', section, re.DOTALL)
        
        if tip_match and category_match:
            tip_data = {
                'tip': tip_match.group(1).strip(),
                'category': category_match.group(1).strip(),
                'emission_reduction': emission_match.group(1).strip() if emission_match else 'Variable',
                'difficulty': difficulty_match.group(1).strip() if difficulty_match else 'Medium',
                'description': description_match.group(1).strip() if description_match else ''
            }
            tips.append(tip_data)
    
    return tips


def load_sustainability_tips(
    filepath: str,
    vector_store: VectorStore,
    embedding_generator: Optional[EmbeddingGenerator] = None
) -> int:
    """
    Load sustainability tips from file into vector store.
    
    Args:
        filepath: Path to the sustainability tips text file
        vector_store: VectorStore instance to load tips into
        embedding_generator: Optional EmbeddingGenerator (not used with ChromaDB auto-embedding)
        
    Returns:
        Number of tips loaded
    """
    # Parse tips from file
    tips = parse_sustainability_tips(filepath)
    
    if not tips:
        return 0
    
    # Convert tips to Document objects
    documents = []
    for tip_data in tips:
        # Create comprehensive content for embedding
        content = f"{tip_data['tip']}. {tip_data['description']}"
        
        # Create metadata
        metadata = {
            'category': tip_data['category'],
            'emission_reduction': tip_data['emission_reduction'],
            'difficulty': tip_data['difficulty'],
            'tip': tip_data['tip']
        }
        
        doc = Document(
            content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    return len(documents)


def initialize_vector_store(
    tips_filepath: str = "./data/sustainability_tips.txt",
    persist_directory: str = "./chroma_db",
    collection_name: str = "sustainability_tips",
    force_reload: bool = False
) -> VectorStore:
    """
    Initialize vector store and load sustainability tips.
    
    Args:
        tips_filepath: Path to sustainability tips file
        persist_directory: Directory for ChromaDB persistence
        collection_name: Name of the collection
        force_reload: If True, clear existing collection and reload
        
    Returns:
        Initialized VectorStore instance
    """
    # Create vector store
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Check if collection already has data
    stats = vector_store.get_collection_stats()
    
    if force_reload or stats['document_count'] == 0:
        if force_reload and stats['document_count'] > 0:
            print(f"Clearing existing collection with {stats['document_count']} documents...")
            vector_store.clear_collection()
        
        print(f"Loading sustainability tips from {tips_filepath}...")
        count = load_sustainability_tips(tips_filepath, vector_store)
        print(f"Successfully loaded {count} sustainability tips into vector store.")
    else:
        print(f"Vector store already contains {stats['document_count']} documents. Skipping reload.")
    
    return vector_store



