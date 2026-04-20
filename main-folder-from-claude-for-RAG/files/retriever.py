# ============================================================
# retriever.py — Vector Store Interaction Layer
# ============================================================
# This module manages ChromaDB: creating it, connecting to it,
# and performing similarity searches against it.
#
# Used by:
#   - ingest.py  → to store new embeddings
#   - chain.py   → to retrieve relevant chunks at query time
# ============================================================

import sys
from pathlib import Path

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    TOP_K,
)


# ─── Embedding Model (singleton) ──────────────────────────────────────────
# We create this once and reuse it across the app.
# Why singleton? The model loads ~90MB from disk. Creating it on every
# request would be extremely slow.

_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model, loading it only on first call.

    HuggingFaceEmbeddings wraps sentence-transformers.
    By default it uses CPU — no GPU needed. Perfect for free tier.

    MODEL DETAILS — all-MiniLM-L6-v2:
    - Architecture: MiniLM (distilled from BERT)
    - Output: 384-dimensional vectors
    - Size: ~22M parameters / ~90MB download
    - Speed: ~14,000 sentences/second on CPU
    - Quality: Excellent for semantic similarity tasks
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # Change to "cuda" if you have a GPU
            encode_kwargs={
                "normalize_embeddings": True,  # L2 normalize → cosine sim = dot product
                "batch_size": 32,              # Process 32 chunks at once
            },
        )
        logger.success("Embedding model loaded successfully")
    return _embedding_model


# ─── ChromaDB Vector Store ─────────────────────────────────────────────────
def get_vectorstore(reset: bool = False) -> Chroma:
    """
    Returns a LangChain Chroma vectorstore backed by persistent storage.

    HOW CHROMADB STORES DATA:
    ChromaDB creates a folder (chroma_store/) with:
      - chroma.sqlite3     : metadata index (source, page, chunk text)
      - data_level0.bin    : HNSW graph index for fast approximate nearest neighbor

    HNSW (Hierarchical Navigable Small World):
    The algorithm ChromaDB uses for fast similarity search. Instead of
    comparing your query vector against every single stored vector (O(n)),
    HNSW builds a graph that lets you "navigate" to nearby vectors in
    O(log n) time. At 10,000 documents, this is the difference between
    ~50ms and ~0.1ms per query.

    Args:
        reset: If True, deletes all existing vectors before returning.

    Returns:
        LangChain Chroma vectorstore object
    """
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    if reset:
        logger.warning("Resetting vector store — all existing vectors will be deleted")
        # Connect to existing collection and delete it
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted collection: {COLLECTION_NAME}")
        except Exception:
            pass  # Collection didn't exist yet, that's fine

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    return vectorstore


# ─── Similarity Search ─────────────────────────────────────────────────────
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K) -> list:
    """
    Given a user query, find the most semantically similar document chunks.

    WHAT HAPPENS INTERNALLY:
    1. The query text is converted to a 384-dim vector by the embedding model
       (the SAME model used during ingestion — this is critical).
    2. ChromaDB computes cosine similarity between the query vector and
       all stored chunk vectors.
    3. The top-k chunks with highest similarity scores are returned.

    COSINE SIMILARITY EXPLAINED:
    Imagine each text is a point in 384-dimensional space. Cosine similarity
    measures the angle between two points, not their distance. Two texts
    are "similar" if they point in the same direction — regardless of length.

    Formula: similarity = (A · B) / (|A| × |B|)
    Score = 1.0: identical meaning
    Score = 0.7: highly related
    Score < 0.3: probably unrelated

    Args:
        query:  The user's question in natural language
        top_k:  Number of chunks to retrieve

    Returns:
        List of LangChain Document objects (the most relevant chunks)
    """
    vectorstore = get_vectorstore()

    # similarity_search_with_score returns [(Document, score), ...]
    results_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    # Filter out low-quality matches (score < 0.3 is probably noise)
    # Note: ChromaDB uses L2 distance by default when not normalizing,
    # but since we normalize embeddings, the score is cosine distance (lower = better)
    filtered = [
        (doc, score)
        for doc, score in results_with_scores
        if score < 1.5  # L2 distance threshold (adjust based on your data)
    ]

    if not filtered:
        logger.warning(f"No relevant chunks found for query: {query[:50]}...")
        return []

    # Log what was retrieved (useful for debugging)
    for i, (doc, score) in enumerate(filtered):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:80].replace("\n", " ")
        logger.debug(f"Chunk {i+1} | Score: {score:.3f} | {source} p.{page} | {preview}...")

    return [doc for doc, _ in filtered]


# ─── Health Check ──────────────────────────────────────────────────────────
def get_collection_stats() -> dict:
    """
    Returns statistics about the current vector store.
    Useful for monitoring and the API health endpoint.
    """
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        return {
            "status": "healthy",
            "collection": COLLECTION_NAME,
            "total_chunks": count,
            "embedding_model": EMBEDDING_MODEL,
            "persist_dir": str(CHROMA_DB_DIR),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
