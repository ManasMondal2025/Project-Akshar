import json
import requests
from .vector_store import get_chroma_client, _BGE_EF
from typing import Dict, Any

# Local Ollama endpoint and default model
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3"

# Maximum number of chunks to retrieve from ChromaDB
_N_RETRIEVE = 5
# Show only the top N most-relevant chunks as visual references
_N_REFS     = 3


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to the local Ollama /api/generate endpoint and return the full response text."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("response", "")


def answer_query(
    query: str,
    document_id: str,
    collection_name: str = "project_akshar",
    model: str = DEFAULT_MODEL,
    **kwargs,          # absorbs legacy api_key etc. without breaking callers
) -> Dict[str, Any]:
    """
    Retrieves the most relevant chunks from the Vector Database and generates
    a verifiable answer using a local Llama 3 model served by Ollama.

    Args:
        query:           The user's question.
        document_id:     The identifier for the document to query against.
        collection_name: The name of the collection in ChromaDB.
        model:           Ollama model tag to use (default: 'llama3').

    Returns:
        Dict containing the final generated 'answer', along with 'source_text',
        'highlights' (top-3 references only), and 'document_source_id'.
    """
    client = get_chroma_client()
    # Must use the same BGE embedding function as indexing — otherwise
    # ChromaDB falls back to its default MiniLM and query vectors won't
    # match the stored BGE vectors.
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_BGE_EF,
    )

    # 1. Retrieve the most relevant chunks
    print(f"Retrieving context for query: '{query}'")
    
    # BAAI/bge models require a specific instruction prefix for search queries
    bge_query = f"Represent this sentence for searching relevant passages: {query}"
    
    results = collection.query(
        query_texts=[bge_query],
        n_results=_N_RETRIEVE,
        where={"document_source_id": document_id},
    )

    if not results["documents"][0]:
        return {"error": "No relevant context found in the document."}

    # Combine top chunks to give Llama 3 maximum context
    retrieved_texts = results["documents"][0]
    source_text = "\n---\n".join(retrieved_texts)

    # ── Build visual references ───────────────────────────────────────────────
    # Only take the top _N_REFS most-relevant chunks as shown references.
    # ChromaDB returns results sorted by relevance (closest first), so we simply
    # take the first _N_REFS entries.
    #
    # Metadata key mapping:
    #   module3 stores:  { page: int,  bbox: [x1,y1,x2,y2], document_source_id: str }
    #   Legacy fallback: { page_num: int, bbox: ... }
    # We handle both key names so the UI always gets a valid page number.
    supporting_highlights = []
    for idx, meta in enumerate(results["metadatas"][0][:_N_REFS]):
        # Support both 'page' (module3 new) and 'page_num' (legacy) keys
        pg = meta.get("page") or meta.get("page_num") or 1
        bx_str = meta.get("bbox")
        bx = json.loads(bx_str) if bx_str else []
        text = retrieved_texts[idx]

        if bx:
            supporting_highlights.append({
                "page_num": int(pg),
                "bbox":     bx,
                "text":     text,
            })

    # 2. Build prompt and call local Llama 3 via Ollama
    prompt = (
        "You are a precise document assistant for Project Akshar.\n"
        "Answer the user's question using ONLY the information in the Provided Context below.\n"
        "Rules:\n"
        "- Be direct and concise. If the answer is a single word, number, or short phrase, say just that.\n"
        "- Do NOT add explanations, background theory, or information not present in the context.\n"
        "- Do NOT speculate or infer beyond what is explicitly stated.\n"
        "- If the context does not contain the answer, say: 'The document does not mention this.'\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{source_text}\n\n"
        "Answer:"
    )

    print(f"Generating grounded answer via local Ollama (model: '{model}')...")
    answer_text = _call_ollama(prompt, model=model)

    # 3. Compile grounded response
    return {
        "answer":             answer_text,
        "source_text":        source_text,
        "highlights":         supporting_highlights,
        "document_source_id": document_id,
    }


if __name__ == "__main__":
    pass
