import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")


def build_index(resume_dir: str) -> None:
    """Read .txt files from resume_dir, embed them, and store a FAISS index."""
    client = OpenAI()

    # Read all .txt files sorted by filename
    files = sorted(
        [f for f in os.listdir(resume_dir) if f.endswith(".txt")]
    )

    chunks = []
    texts = []
    for f in files:
        with open(os.path.join(resume_dir, f), "r", encoding="utf-8") as fh:
            text = fh.read()
        chunk_id = os.path.splitext(f)[0]  # e.g. "chunk1"
        chunks.append({"text": text, "metadata": {"source": "resume", "chunk_id": chunk_id}})
        texts.append(text)

    # Embed all chunks
    response = client.embeddings.create(model="text-embedding-3-large", input=texts)
    embeddings = np.array([item.embedding for item in response.data], dtype="float32")

    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Build FAISS index (inner product)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)


def load_index():
    """Load and return (faiss_index, chunks_list) from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as fh:
        chunks = pickle.load(fh)
    return index, chunks
