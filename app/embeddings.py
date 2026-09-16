"""
Embedding utilities backed by a local Ollama server.

Requires:
    ollama pull nomic-embed-text
    ollama serve   (usually already running as a background service)
"""

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> np.ndarray:
    """Get a single embedding vector from Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return np.array(data["embedding"], dtype=np.float32)


import concurrent.futures

def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Get embeddings for a list of texts concurrently (up to 5 workers).
    Preserves order and handles individual connection errors."""
    if not texts:
        return np.array([])
        
    embeddings = []
    # Use max_workers=5 to prevent overwhelming local Ollama on typical hardware
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # map guarantees results in the same order as texts
        # any exception in get_embedding will be raised here
        try:
            for emb in executor.map(get_embedding, texts):
                embeddings.append(emb)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch embeddings concurrently: {e}") from e
            
    return np.vstack(embeddings)


def cosine_similarity_matrix(a_vecs: np.ndarray, b_vecs: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between every row of a_vecs and every row of b_vecs."""
    a_norm = a_vecs / (np.linalg.norm(a_vecs, axis=1, keepdims=True) + 1e-8)
    b_norm = b_vecs / (np.linalg.norm(b_vecs, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T
