FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src/ ./src/
COPY api.py app.py ./

RUN mkdir -p /app/src/documents /app/src/chroma_store

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]