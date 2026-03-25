import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rag.ingest import load_index

load_dotenv()

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-large")


def retrieve(query: str, top_k: int = 2, score_threshold: float = 0.3) -> list[dict]:
    """Embed query, search FAISS index, return matching chunks above threshold."""
    client = OpenAI()
    index, chunks = load_index()

    # Embed the query
    response = client.embeddings.create(model=EMBEDDINGS_MODEL, input=[query])
    query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    # Normalize
    norm = np.linalg.norm(query_embedding)
    query_embedding = query_embedding / norm

    # Search
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < score_threshold:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(score),
        })

    return results
