# RAG Customer Support Chatbot
## Full Open-Source Stack · Zero API Costs · Production-Ready

---

## Tech Stack (All Free)

| Component | Tool | Why |
|---|---|---|
| LLM | Mistral 7B via Ollama | Runs locally, no API key |
| Embeddings | all-MiniLM-L6-v2 | Fast, CPU-friendly, 384-dim |
| Vector Store | ChromaDB | Persistent, open-source |
| Framework | LangChain | Orchestration |
| API | FastAPI | Production REST API |
| Frontend | Streamlit | Chat UI |
| Deployment | Docker Compose | One-command deploy |

---

## Project Structure

```
rag-chatbot/
├── documents/          ← PUT YOUR PDFs AND FAQs HERE
├── src/
│   ├── config.py       ← All settings (model names, chunk size, etc.)
│   ├── ingest.py       ← Offline indexing pipeline (run once)
│   ├── retriever.py    ← ChromaDB operations
│   └── chain.py        ← RAG chain (retrieval + LLM)
├── api.py              ← FastAPI REST endpoints
├── app.py              ← Streamlit chat frontend
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup Guide

### Step 1 — Install Ollama (one-time setup)

Ollama lets you run LLMs locally for free.

**Mac/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:** Download installer from https://ollama.ai

After installing, pull the Mistral model (~4GB):
```bash
ollama pull mistral
```

Want a lighter model? Try:
```bash
ollama pull phi3      # Microsoft Phi-3 (~2.3GB, good on 8GB RAM)
ollama pull gemma2:2b # Google Gemma 2 (~1.6GB, fastest)
```

If you switch models, update `LLM_MODEL` in `src/config.py`.

---

### Step 2 — Add your documents

Put your support documents in the `/documents` folder:
- Customer FAQs (.pdf, .docx, .txt, .md)
- Product manuals
- Return policy documents
- Shipping information

Example structure:
```
documents/
├── return_policy.pdf
├── shipping_faq.txt
├── product_manual_v2.pdf
└── billing_questions.md
```

---

### Step 3 — Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 4 — Run the Ingestion Pipeline (indexes your documents)

```bash
python src/ingest.py
```

What this does:
1. Loads all documents from `/documents`
2. Splits them into 500-token chunks with 50-token overlap
3. Converts each chunk to a 384-dim vector (using all-MiniLM-L6-v2)
4. Stores everything in ChromaDB (saved to `/chroma_store`)

**First run:** Downloads the embedding model (~90MB). Future runs are instant.

Re-index after adding new documents:
```bash
python src/ingest.py --reset    # Wipes old vectors and re-indexes everything
```

---

### Step 5 — Start the services

**Option A: Run locally (development)**

Terminal 1 — Start Ollama:
```bash
ollama serve
```

Terminal 2 — Start FastAPI:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 3 — Start Streamlit UI:
```bash
streamlit run app.py
```

Open your browser: http://localhost:8501

**Option B: Docker (production)**
```bash
docker-compose up -d
```

Wait ~2 minutes for Mistral to download, then:
- Chat UI: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## API Usage

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I return a product?"}'
```

Response:
```json
{
  "question": "How do I return a product?",
  "answer": "To return a product, follow these steps: 1. Log into your account... 2. Go to Orders...",
  "sources": [
    {
      "file": "return_policy.pdf",
      "page": 2,
      "preview": "Items can be returned within 30 days of purchase..."
    }
  ],
  "model_used": "mistral (via Ollama)"
}
```

### Streaming response (JavaScript)
```javascript
const response = await fetch('http://localhost:8000/ask/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'How do I return a product?' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  // chunk = "data: <token>\n\n"
  console.log(chunk.replace('data: ', '').replace('\n\n', ''));
}
```

---

## Customization

### Change chunk size (in config.py)
```python
CHUNK_SIZE = 500      # Increase for longer context, decrease for precision
CHUNK_OVERLAP = 50    # Always ~10% of chunk size
```

### Change number of retrieved chunks
```python
TOP_K = 4    # More = richer context but slower. 3-5 is ideal.
```

### Switch embedding model
```python
# Higher quality, needs more RAM:
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # 768-dim, better quality
EMBEDDING_MODEL = "thenlper/gte-large"       # 1024-dim, best quality
```

### Switch LLM model
```python
LLM_MODEL = "llama3.2"  # Meta's Llama 3.2
LLM_MODEL = "phi3"      # Microsoft Phi-3 (faster on low-RAM machines)
```

---

## Free Deployment Options

### Option 1: Hugging Face Spaces (easiest, completely free)
- Upload your code to a HF Space
- Use HF's free GPU for inference
- Limited to models available on HF Hub

### Option 2: Railway.app (free tier)
- Connect your GitHub repo
- Railway auto-builds the Docker image
- 512MB RAM free tier — use phi3 or gemma2:2b models

### Option 3: Google Cloud Run (free tier)
- 2 million requests/month free
- Auto-scales to zero when not in use

### Option 4: Your own VPS (cheapest long-term)
- DigitalOcean Droplet: $6/month (2GB RAM)
- Hetzner: €3.79/month (2GB RAM)
- SCP your code, run docker-compose up -d

---

## Common Issues

**"Cannot connect to Ollama"**
→ Make sure Ollama is running: `ollama serve`

**"No documents found"**
→ Add files to the `/documents` folder, then run `python src/ingest.py`

**"Out of memory"**
→ Switch to a lighter model: `ollama pull phi3`, then update `LLM_MODEL = "phi3"` in config.py

**Slow responses**
→ Normal! Mistral 7B on CPU takes 10-30 seconds. For faster responses use phi3 or upgrade to a machine with GPU.
