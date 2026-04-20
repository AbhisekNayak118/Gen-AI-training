# ============================================================
# Dockerfile — Production Container for the RAG Chatbot
# ============================================================
# Build:  docker build -t rag-chatbot .
# Run:    docker-compose up
# ============================================================

# Use Python 3.11 slim — smaller than the full image (~130MB vs ~900MB)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────────────
# These are needed by some Python packages (like unstructured for PDFs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    # For PDF processing
    poppler-utils \
    # For file type detection
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer.
# If you only change app code, Docker won't re-install packages.
# This makes rebuilds MUCH faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model ───────────────────────────────────────
# Download the embedding model during build so it's baked into the image.
# Otherwise, the first request would wait ~2 minutes for the download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ── Copy application code ──────────────────────────────────────────────────
COPY src/ ./src/
COPY api.py app.py ./

# ── Create directories ─────────────────────────────────────────────────────
RUN mkdir -p /app/documents /app/chroma_store

# ── Expose ports ───────────────────────────────────────────────────────────
EXPOSE 8000    # FastAPI
EXPOSE 8501    # Streamlit

# ── Default command: start the FastAPI server ──────────────────────────────
# Override in docker-compose for different services
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
