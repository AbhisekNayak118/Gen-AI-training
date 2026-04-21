# ============================================================
# ingest.py — The Offline Indexing Pipeline
# ============================================================
# This script runs ONCE (or when your documents change).
# What it does:
#   1. Load documents (PDF, Word, TXT, Markdown) from /documents
#   2. Split them into overlapping chunks
#   3. Convert each chunk into a vector embedding
#   4. Store vectors + metadata in ChromaDB (persistent on disk)
#
# Run with: python src/ingest.py
# ============================================================

import os
import sys
from pathlib import Path
from loguru import logger

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,          # For PDF files
    Docx2txtLoader,       # For Word documents (.docx)
    TextLoader,           # For .txt files
    # UnstructuredMarkdownLoader,  # For .md files
    DirectoryLoader,      # Loads all files from a folder
)

# Splits text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Our custom modules
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    DOCUMENTS_DIR,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)
from src.retriever import get_vectorstore


# ─── Step 1: Load Documents ────────────────────────────────────────────────
def load_documents(directory: Path) -> list:
    """
    Load all supported document types from the given directory.

    Supports: PDF, DOCX, TXT, MD

    Each loaded document becomes a LangChain Document object with:
      - .page_content : the raw text
      - .metadata     : dict with 'source', 'page', etc.

    Returns:
        List of LangChain Document objects
    """
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    documents = []

    # Walk through every file in the documents folder
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        loader = None

        if suffix == ".pdf":
            # PyPDFLoader extracts text page by page.
            # Each page becomes a separate Document object with metadata:
            #   {"source": "path/to/file.pdf", "page": 0}
            loader = PyPDFLoader(str(file_path))

        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))

        elif suffix == ".txt":
            # encoding="utf-8" handles special characters in support docs
            loader = TextLoader(str(file_path), encoding="utf-8")



        else:
            logger.warning(f"Skipping unsupported file type: {file_path.name}")
            continue

        try:
            loaded = loader.load()
            documents.extend(loaded)
            logger.info(f"Loaded {len(loaded)} page(s) from: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.success(f"Total documents loaded: {len(documents)}")
    return documents


# ─── Step 2: Chunk Documents ───────────────────────────────────────────────
def chunk_documents(documents: list) -> list:
    """
    Split large documents into smaller overlapping chunks.

    WHY CHUNKING?
    LLMs have a context window limit. You can't dump an entire 50-page
    manual into the prompt. Instead, you split it into focused pieces,
    retrieve only the relevant pieces, and feed those to the LLM.

    WHY RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in order:
      1. Paragraph breaks (\n\n)
      2. Line breaks (\n)
      3. Sentence ends (". ")
      4. Words (" ")
      5. Characters (last resort)
    This keeps semantic units intact better than naive character splitting.

    WHY OVERLAP?
    If an answer spans a chunk boundary (e.g., question is at end of
    chunk A, answer is at start of chunk B), overlap ensures both chunks
    contain enough context to be retrieved.

    Returns:
        List of smaller LangChain Document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Max tokens per chunk
        chunk_overlap=CHUNK_OVERLAP,  # Shared tokens between neighbors
        length_function=len,          # Simple character-based length
        # Separators tried in order (most to least preferred)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    logger.info(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


# ─── Step 3 & 4: Embed + Store in ChromaDB ────────────────────────────────
def embed_and_store(chunks: list, reset: bool = False):
    """
    Convert text chunks to vectors and store them in ChromaDB.

    HOW EMBEDDINGS WORK:
    The embedding model (all-MiniLM-L6-v2) maps each chunk of text to a
    384-dimensional vector. Chunks with similar meaning end up with similar
    vectors (high cosine similarity). This is what enables semantic search
    — "laptop battery life" can match "notebook power duration" even though
    the words differ.

    ChromaDB stores:
      - The vector (384 floats)
      - The original text (for LLM context)
      - Metadata (source filename, page number)

    The data is PERSISTED to disk, so you only run this once.

    Args:
        chunks: List of Document objects to embed and store
        reset:  If True, clears existing data before inserting.
                Use when you've updated your documents.
    """
    vectorstore = get_vectorstore(reset=reset)

    logger.info(f"Embedding {len(chunks)} chunks with model: {EMBEDDING_MODEL}")
    logger.info("This may take a few minutes on first run (model downloads ~90MB)...")

    # Add documents in batches to avoid memory issues with large collections
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info(f"Stored batch {i // BATCH_SIZE + 1} / {-(-len(chunks) // BATCH_SIZE)}")

    logger.success(f"Successfully embedded and stored {len(chunks)} chunks in ChromaDB")
    logger.info(f"Vector store location: {CHROMA_DB_DIR}")


# ─── Main Ingestion Runner ─────────────────────────────────────────────────
def run_ingestion(reset: bool = False):
    """
    Full pipeline: load → chunk → embed → store.

    Args:
        reset: Pass True to wipe existing vectors and re-index from scratch.
    """
    logger.info("=" * 60)
    logger.info("Starting RAG ingestion pipeline")
    logger.info("=" * 60)

    # Step 1
    logger.info(f"Loading documents from: {DOCUMENTS_DIR}")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        logger.error("No documents found! Add PDFs/docs to the /documents folder.")
        return

    # Step 2
    logger.info("Chunking documents...")
    chunks = chunk_documents(documents)

    # Steps 3 & 4
    logger.info("Embedding and storing in ChromaDB...")
    embed_and_store(chunks, reset=reset)

    logger.success("Ingestion complete! Your chatbot is ready to answer questions.")


if __name__ == "__main__":
    # Pass --reset flag to re-index: python src/ingest.py --reset
    reset_flag = "--reset" in sys.argv
    run_ingestion(reset=reset_flag)
