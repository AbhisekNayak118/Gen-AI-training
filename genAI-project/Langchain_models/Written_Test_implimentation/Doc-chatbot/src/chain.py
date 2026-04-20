# ============================================================
# chain.py — The RAG Chain (Heart of the System)
# ============================================================
# This connects everything together:
#   User Question
#       ↓
#   Retrieve relevant chunks from ChromaDB
#       ↓
#   Build a prompt: [System instruction] + [Retrieved context] + [Question]
#       ↓
#   Send to Mistral LLM via Ollama
#       ↓
#   Return grounded answer + source documents
#
# The LangChain "chain" concept means each step is linked.
# Output of one step feeds directly into the next.
# ============================================================

import sys
from pathlib import Path
from typing import Generator

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from src.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from src.retriever import retrieve_relevant_chunks


# ─── LLM Setup (Ollama — free, local) ─────────────────────────────────────
def get_llm(streaming: bool = False) -> Ollama:
    """
    Returns a Mistral LLM instance running via Ollama.

    WHAT IS OLLAMA?
    Ollama is a free tool that runs open-source LLMs locally on your machine.
    Think of it as "Docker for LLMs". It handles:
      - Model downloading (one-time, ~4GB for Mistral 7B)
      - Quantization (compresses model to fit in RAM)
      - API server (REST API at localhost:11434)

    SETUP (one-time, before running this code):
    1. Install Ollama: https://ollama.ai
    2. Pull Mistral:   ollama pull mistral
    3. (Optional)      ollama pull llama3.2  or  ollama pull phi3

    QUANTIZATION EXPLAINED:
    Full-precision Mistral 7B needs ~28GB RAM. That's a lot.
    Ollama downloads the 4-bit quantized version (Q4_K_M) which needs
    only ~4.1GB RAM, with minimal quality loss. Magic!

    Args:
        streaming: If True, enables token-by-token streaming to the frontend.
    """
    return Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,       # Low temp = factual, deterministic answers
                               # Higher temp = more creative but less accurate
                               # For customer support: always keep this low
        num_ctx=4096,          # Context window size in tokens
        streaming=streaming,
    )


# ─── Prompt Template ───────────────────────────────────────────────────────
# This is the most important part of your RAG system.
# A well-crafted prompt controls:
#   - How the LLM uses the retrieved context
#   - Whether it hallucinates or stays grounded
#   - The tone and format of responses

RAG_PROMPT_TEMPLATE = """You are a helpful customer support assistant. Your job is to answer customer questions accurately using ONLY the information provided in the context below.

CONTEXT (retrieved from our support documents):
{context}

STRICT RULES:
1. Answer ONLY based on the context above. Do not use outside knowledge.
2. If the context does not contain enough information, say: "I don't have that information in our support documents. Please contact our support team directly."
3. Be concise and direct. Customers want quick answers.
4. If the answer involves steps, use a numbered list.
5. Never make up product names, prices, policies, or contact information.

CUSTOMER QUESTION:
{question}

ANSWER:"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ─── Helper: Format retrieved docs into a context string ──────────────────
def format_context(docs: list[Document]) -> str:
    """
    Joins retrieved document chunks into a single context string.

    Why we add source labels:
    The LLM can see which document each piece came from. This helps it
    generate more precise answers and allows us to show citations to users.

    Example output:
        [Source: refund_policy.pdf, Page 2]
        Items must be returned within 30 days of purchase...

        [Source: shipping_faq.txt]
        Standard shipping takes 3-5 business days...
    """
    formatted_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "")
        page_label = f", Page {page + 1}" if page != "" else ""

        # Get just the filename, not the full path
        source_name = Path(source).name if source != "Unknown source" else source

        formatted_parts.append(
            f"[Source: {source_name}{page_label}]\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted_parts)


# ─── Main RAG Function ─────────────────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Full RAG pipeline for a single question.

    FLOW:
    1. Retrieve top-k relevant chunks from ChromaDB
    2. Format them into a context string
    3. Fill the prompt template with context + question
    4. Send to Mistral LLM
    5. Return answer + source documents for citation

    Args:
        question: The customer's question in natural language

    Returns:
        {
          "answer":   str  — LLM's grounded response
          "sources":  list — Source documents used (for citations)
          "question": str  — Original question (for logging)
        }
    """
    logger.info(f"Processing question: {question[:80]}...")

    # Step 1: Retrieve relevant chunks
    relevant_docs = retrieve_relevant_chunks(question, top_k=TOP_K)

    if not relevant_docs:
        return {
            "answer": (
                "I couldn't find relevant information in our support documents for your question. "
                "Please contact our support team directly for assistance."
            ),
            "sources": [],
            "question": question,
        }

    # Step 2: Format context
    context = format_context(relevant_docs)
    logger.debug(f"Context length: {len(context)} chars from {len(relevant_docs)} chunks")

    # Step 3: Build prompt
    filled_prompt = PROMPT.format(context=context, question=question)

    # Step 4: Call LLM
    llm = get_llm(streaming=False)
    answer = llm.invoke(filled_prompt)

    # Step 5: Return result with sources
    sources = [
        {
            "file": Path(doc.metadata.get("source", "")).name,
            "page": doc.metadata.get("page", "N/A"),
            "preview": doc.page_content[:200] + "...",
        }
        for doc in relevant_docs
    ]

    logger.success(f"Answer generated ({len(answer)} chars)")

    return {
        "answer": answer.strip(),
        "sources": sources,
        "question": question,
    }


# ─── Streaming Version ─────────────────────────────────────────────────────
def ask_stream(question: str) -> Generator[str, None, None]:
    """
    Streaming version of ask() — yields tokens as they're generated.

    This is what powers the real-time typing effect in the Streamlit UI.
    Instead of waiting for the full answer (could take 5-10 seconds),
    tokens appear on screen as the LLM produces them.

    Yields:
        Individual tokens (strings) as they stream from Ollama
    """
    relevant_docs = retrieve_relevant_chunks(question, top_k=TOP_K)

    if not relevant_docs:
        yield "I couldn't find relevant information. Please contact support directly."
        return

    context = format_context(relevant_docs)
    filled_prompt = PROMPT.format(context=context, question=question)

    llm = get_llm(streaming=True)

    # LLM.stream() yields tokens one by one
    for token in llm.stream(filled_prompt):
        yield token


# ─── LangChain LCEL Alternative (modern approach) ─────────────────────────
def build_rag_chain():
    """
    Builds a RAG chain using LangChain Expression Language (LCEL).

    LCEL is LangChain's modern way to compose chains using the pipe (|) operator.
    Each | passes the output of one step as input to the next.

    This is equivalent to the ask() function above, but written in
    the declarative LCEL style.

    Usage:
        chain = build_rag_chain()
        result = chain.invoke({"question": "How do I return a product?"})
    """
    from src.retriever import get_vectorstore

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    def format_docs(docs):
        return format_context(docs)

    chain = (
        {
            # RunnablePassthrough passes the question through unchanged
            # while also running the retriever in parallel
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT         # Fill the template
        | get_llm()      # Send to Mistral
        | StrOutputParser()  # Parse the LLM output to a string
    )

    return chain
