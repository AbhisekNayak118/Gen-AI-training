# ============================================================
# app.py — Streamlit Frontend
# ============================================================
# A polished chat interface for the RAG chatbot.
#
# Run with: streamlit run app.py
# Opens at: http://localhost:8501
#
# Features:
#   - Real-time streaming responses (typing effect)
#   - Conversation history (multi-turn chat)
#   - Source citations for every answer
#   - System stats in sidebar
# ============================================================

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))
from src.chain import ask, ask_stream
from src.retriever import get_collection_stats


# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 6px 0;
        font-size: 13px;
    }
    .source-label {
        font-weight: 600;
        color: #4a5568;
    }
    .answer-box {
        padding: 1rem;
        border-radius: 8px;
        background: #f7fafc;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Support Bot")
    st.caption("Powered by Mistral + ChromaDB")

    st.divider()

    # System stats
    st.subheader("System Status")
    stats = get_collection_stats()

    if stats["status"] == "healthy":
        st.success("Vector Store: Online")
        st.metric("Documents Indexed", stats["total_chunks"])
        st.caption(f"Model: `{stats['embedding_model'].split('/')[-1]}`")
    else:
        st.error(f"Vector Store Error: {stats.get('error', 'Unknown')}")

    st.divider()

    # Settings
    st.subheader("Settings")
    streaming_enabled = st.toggle("Streaming responses", value=True)
    show_sources = st.toggle("Show source documents", value=True)

    st.divider()
    st.caption("Built with LangChain + Ollama + ChromaDB")
    st.caption("100% open-source, runs locally")


# ─── Main Chat Interface ────────────────────────────────────────────────────
st.title("Customer Support Assistant")
st.caption("Ask me anything about our products, shipping, returns, and more.")

# Initialize conversation history in session state
# Session state persists across re-runs within the same browser session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages if available
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} documents)"):
                for src in msg["sources"]:
                    page_info = f" · Page {src['page'] + 1}" if src.get("page") != "N/A" else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-label">📄 {src['file']}{page_info}</div>
                        <div style="color: #718096; margin-top: 4px;">{src['preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─── Chat Input ─────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about our products or services..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display assistant response
    with st.chat_message("assistant"):

        if streaming_enabled:
            # ── Streaming mode: tokens appear as they're generated ──────────
            # st.write_stream() handles the streaming display for us
            full_response = ""
            sources = []

            with st.spinner("Searching documents..."):
                # Get sources separately (before streaming the answer)
                result = ask(question)
                sources = result.get("sources", [])

            # Now stream the answer
            # We use a custom generator that replays the tokens
            def response_generator():
                for token in ask_stream(question):
                    yield token

            full_response = st.write_stream(response_generator())

        else:
            # ── Non-streaming mode: wait for full answer ────────────────────
            with st.spinner("Searching documents and generating answer..."):
                result = ask(question)
                full_response = result["answer"]
                sources = result.get("sources", [])

            st.markdown(full_response)

        # Show sources
        if show_sources and sources:
            with st.expander(f"📚 Sources ({len(sources)} documents used)"):
                for src in sources:
                    page_info = f" · Page {src['page'] + 1}" if src.get("page") != "N/A" else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-label">📄 {src['file']}{page_info}</div>
                        <div style="color: #718096; margin-top: 4px;">{src['preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })


# ─── Example Questions (shown when chat is empty) ──────────────────────────
if not st.session_state.messages:
    st.divider()
    st.subheader("💡 Try asking:")
    example_questions = [
        "How do I return a product?",
        "What is your refund policy?",
        "How long does shipping take?",
        "How do I track my order?",
        "What payment methods do you accept?",
    ]

    cols = st.columns(3)
    for i, q in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                # Clicking an example question submits it
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
