import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Any
import json

# BGE embedding function used for both indexing and querying.
# Must be the same model as embedder.py so stored vectors are compatible.
_BGE_EF = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    normalize_embeddings=True,
)

def get_chroma_client(persist_directory: str = "./chroma_db") -> chromadb.ClientAPI:
    """
    Initializes a persistent ChromaDB client for vector storage.
    """
    # Create the directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)

def build_vector_index(
    document_id: str, 
    metadata_blocks: List[Dict[str, Any]], 
    collection_name: str = "project_akshar"
):
    """
    Indexes the text chunks and their Visual Grounding Metadata into ChromaDB.
    
    Args:
        document_id: A unique identifier for the processed document from Module 1 (e.g., "doc_001_book.pdf")
        metadata_blocks: List of dicts, expectation: [{"text": str, "bbox": [float,...], "page_num": int},...]
        collection_name: Name of the chromadb collection.
    
    Returns:
        chromadb.Collection: The populated Chroma collection
    """
    client = get_chroma_client()

    # Explicitly use BGE embedding function — avoids the default ONNX MiniLM
    # that ChromaDB would otherwise apply silently.
    # If the collection was previously created with a different/default embedding
    # function, ChromaDB raises a conflict error. We delete and recreate it cleanly.
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=_BGE_EF,
        )
    except Exception as e:
        err_str = str(e).lower()
        if "embedding function" in err_str and ("conflict" in err_str or "already exists" in err_str):
            print(f"[vector_store] Embedding function conflict on '{collection_name}'. "
                  f"Deleting stale collection and recreating with BGE embedder...")
            client.delete_collection(name=collection_name)
            collection = client.create_collection(
                name=collection_name,
                embedding_function=_BGE_EF,
            )
            print(f"[vector_store] Collection '{collection_name}' recreated successfully.")
        else:
            raise
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, block in enumerate(metadata_blocks):
        text = block.get("text", "")
        if not text:
            continue

        bbox     = block.get("bbox", [])
        # module3 outputs use the key "page"; legacy code uses "page_num".
        # Store both so rag_inference.py can always find the right value.
        page_num = int(block.get("page") or block.get("page_num") or 1)

        # Prepare for bulk insert
        documents.append(text)

        # Chroma metadata requires strictly int, float, str, or bool values.
        # Bounding box list must be serialised to a JSON string.
        metadatas.append({
            "document_source_id": document_id,
            "page":     page_num,   # module3 new key
            "page_num": page_num,   # legacy key — kept for backward compatibility
            "bbox":     json.dumps(bbox),
        })
        
        ids.append(f"{document_id}_block_{idx}")
        
    if documents:
        # Upsert allows us to overwrite existing blocks associated with the same document safely
        print(f"Indexing {len(documents)} blocks into '{collection_name}' collection for document {document_id}...")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Vector Database indexing complete!")
        
    return collection

if __name__ == "__main__":
    pass