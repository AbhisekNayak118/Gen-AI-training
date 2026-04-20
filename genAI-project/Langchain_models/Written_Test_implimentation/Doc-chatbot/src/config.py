# ============================================================
# config.py — Central configuration for the RAG pipeline
# ============================================================
# All settings live here. Change once, affects everywhere.
# No hardcoded values scattered across files.

from pathlib import Path

# --- Project paths ---
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"      # Put your PDFs/FAQs here
CHROMA_DB_DIR = BASE_DIR / "chroma_store"  # Vector store persists here

# --- Embedding model (open-source, runs on CPU, completely free) ---
# all-MiniLM-L6-v2: Fast, small (22M params), great for semantic search
# Alternatives if you have more RAM:
#   "BAAI/bge-base-en-v1.5"   — better quality, slightly slower
#   "thenlper/gte-large"      — best quality, needs more RAM
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (via Ollama — runs locally, completely free) ---
# Requires: `ollama pull mistral` after installing Ollama
# Alternatives:
#   "llama3.2"   — Meta's Llama 3.2 (excellent quality)
#   "phi3"       — Microsoft Phi-3 (great on low-RAM machines)
#   "gemma2:2b"  — Google Gemma 2 (very lightweight)
LLM_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama port

# --- Text Chunking strategy ---
# CHUNK_SIZE: How many tokens per chunk. 
#   Too small → misses context. Too large → retrieval is noisy.
#   500 is a good starting point for support docs.
CHUNK_SIZE = 500
# CHUNK_OVERLAP: Tokens shared between adjacent chunks.
#   Prevents answers from being cut off at chunk boundaries.
CHUNK_OVERLAP = 50

# --- Retrieval settings ---
# TOP_K: How many chunks to retrieve per query.
#   More = more context for LLM, but also more noise.
#   3-5 is standard for focused Q&A.
TOP_K = 4

# --- ChromaDB collection name ---
COLLECTION_NAME = "support_docs"

# --- API settings ---
API_HOST = "0.0.0.0"
API_PORT = 8000
