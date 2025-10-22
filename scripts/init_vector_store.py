"""
Vector Store Initialization Script

This script loads sustainability tips from the knowledge base,
generates embeddings, and populates the ChromaDB vector store.
"""

import sys
from pathlib import Path

# Add parent directory to path to import components
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.vector_store import VectorStore, Document
from components.embeddings import EmbeddingGenerator
import config


def parse_sustainability_tips(filepath: str) -> list[Document]:
    """
    Parse sustainability tips from text file into Document objects.
    
    Args:
        filepath: Path to sustainability_tips.txt
        
    Returns:
        List of Document objects with parsed tips
    """
    documents = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by TIP: markers
    tips = content.split('TIP:')[1:]  # Skip first empty split
    
    for tip_text in tips:
        lines = tip_text.strip().split('\n')
        if not lines:
            continue
        
        # Parse tip components
        tip_title = lines[0].strip()
        category = ""
        emission_reduction = ""
        difficulty = ""
        description = ""
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('CATEGORY:'):
                category = line.replace('CATEGORY:', '').strip()
            elif line.startswith('EMISSION_REDUCTION:'):
                emission_reduction = line.replace('EMISSION_REDUCTION:', '').strip()
            elif line.startswith('DIFFICULTY:'):
                difficulty = line.replace('DIFFICULTY:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
        
        # Create document with full content and metadata
        full_content = f"{tip_title}\n\n{description}"
        
        metadata = {
            "title": tip_title,
            "category": category,
            "emission_reduction": emission_reduction,
            "difficulty": difficulty
        }
        
        doc = Document(
            content=full_content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def initialize_vector_store(
    tips_path: str = None,
    persist_dir: str = None,
    clear_existing: bool = False
) -> dict:
    """
    Initialize the vector store with sustainability tips.
    
    Args:
        tips_path: Path to sustainability tips file (uses config default if None)
        persist_dir: Directory for ChromaDB persistence (uses config default if None)
        clear_existing: Whether to clear existing collection before adding
        
    Returns:
        Dictionary with initialization statistics
    """
    # Use config defaults if not provided
    tips_path = tips_path or config.SUSTAINABILITY_TIPS_PATH
    persist_dir = persist_dir or config.VECTOR_DB_PATH
    
    print(f"Initializing vector store...")
    print(f"Tips file: {tips_path}")
    print(f"Persist directory: {persist_dir}")
    
    # Verify tips file exists
    if not Path(tips_path).exists():
        raise FileNotFoundError(f"Sustainability tips file not found: {tips_path}")
    
    # Parse sustainability tips
    print("\nParsing sustainability tips...")
    documents = parse_sustainability_tips(tips_path)
    print(f"Parsed {len(documents)} tips")
    
    # Initialize vector store
    print("\nInitializing ChromaDB vector store...")
    vector_store = VectorStore(
        collection_name="sustainability_tips",
        persist_directory=persist_dir,
        embedding_model=config.EMBEDDING_MODEL
    )
    
    # Clear existing if requested
    if clear_existing:
        print("Clearing existing collection...")
        vector_store.clear_collection()
    
    # Check if collection already has documents
    stats = vector_store.get_collection_stats()
    existing_count = stats['document_count']
    
    if existing_count > 0:
        print(f"\nWarning: Collection already contains {existing_count} documents")
        response = input("Do you want to clear and reinitialize? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            vector_store.clear_collection()
            existing_count = 0
        else:
            print("Skipping initialization. Existing data preserved.")
            return stats
    
    # Add documents to vector store
    print(f"\nAdding {len(documents)} documents to vector store...")
    print("Generating embeddings (this may take a moment)...")
    vector_store.add_documents(documents)
    
    # Verify initialization
    print("\nVerifying vector store initialization...")
    final_stats = vector_store.get_collection_stats()
    
    print(f"\n✓ Vector store initialized successfully!")
    print(f"  Collection: {final_stats['collection_name']}")
    print(f"  Documents: {final_stats['document_count']}")
    print(f"  Location: {final_stats['persist_directory']}")
    
    # Test search functionality
    print("\nTesting search functionality...")
    test_query = "reduce car emissions"
    results = vector_store.search(test_query, k=3)
    
    if results:
        print(f"✓ Search test successful! Found {len(results)} relevant tips for '{test_query}'")
        print("\nTop result:")
        print(f"  Title: {results[0].metadata.get('title', 'N/A')}")
        print(f"  Category: {results[0].metadata.get('category', 'N/A')}")
    else:
        print("✗ Search test failed - no results returned")
        return {"error": "Search functionality not working"}
    
    return final_stats


def main():
    """Main entry point for the script."""
    try:
        stats = initialize_vector_store(clear_existing=False)
        
        if "error" not in stats:
            print("\n" + "="*60)
            print("Vector store initialization complete!")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("Vector store initialization failed!")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
