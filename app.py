"""
app.py
------
Main Streamlit chatbot interface.

Features (as required by assignment):
  - User input field for natural language queries
  - Real-time streaming model response (token-by-token)
  - Display of source chunks used to generate the answer
  - Sidebar: current model, total indexed chunks
  - Clear chat / reset functionality
"""

import sys
import os

# Add project root to path so we can import our modules from src
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.retriever import Retriever
from src.generator import generate_response, MODEL_NAME


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="eBay Legal Doc Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Initialize Retriever (cached — load only once per session) ────────────────

@st.cache_resource(show_spinner="Loading vector index...")
def load_retriever() -> Retriever:
    """
    Cache the Retriever across Streamlit reruns.

    WHY @st.cache_resource?
      Streamlit reruns the entire script on every user interaction.
      Without caching, we'd reload the FAISS index + embedding model
      on every keypress — making the app unusably slow.

      cache_resource is for heavy objects (models, DB connections)
      that should be shared across all sessions.
    """
    return Retriever()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(retriever: Retriever):
    """Render the sidebar with model info and stats."""
    with st.sidebar:
        st.title("⚖️ RAG Chatbot")
        st.markdown("**eBay User Agreement Assistant**")
        st.divider()

        # Model info section (required by assignment)
        st.subheader("🤖 Model Info")
        st.info(f"**LLM:** {MODEL_NAME}")
        st.info("**Embeddings:** all-MiniLM-L6-v2")
        st.info("**Vector DB:** FAISS (cosine similarity)")

        st.divider()

        # Chunk stats (required by assignment)
        st.subheader("📚 Index Stats")
        chunk_count = retriever.get_chunk_count()
        st.metric(label="Total Indexed Chunks", value=chunk_count)

        st.divider()

        # Clear chat button (required by assignment)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            # Reset chat history in session state
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # How-to-use guide
        st.subheader("💡 Try asking:")
        sample_queries = [
            "What is eBay's arbitration policy?",
            "Can sellers charge outside of eBay?",
            "How does eBay Money Back Guarantee work?",
            "What fees does eBay charge?",
            "How can I opt out of arbitration?",
        ]
        for q in sample_queries:
            st.markdown(f"- *{q}*")


# ── Chat History Init ─────────────────────────────────────────────────────────

def init_session_state():
    """
    Initialize session state variables.

    WHY session state?
      Streamlit reruns the whole script on every interaction.
      Session state persists values across reruns within the same browser session.
      Without it, chat history would vanish on every message.
    """
    if "messages" not in st.session_state:
        # Each message: {"role": "user"/"assistant", "content": str, "sources": list}
        st.session_state.messages = []


# ── Render Chat History ───────────────────────────────────────────────────────

def render_chat_history():
    """Display all previous messages in the chat."""
    # WHY enumerate? — we need idx to build a unique key_prefix per message
    # Without idx, same chunk IDs across different messages cause duplicate key errors
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show source chunks for assistant messages (collapsed in history)
            if msg["role"] == "assistant" and msg.get("sources"):
                render_source_chunks(
                    msg["sources"],
                    collapsed=True,
                    key_prefix=f"hist_{idx}",  # Unique prefix per message
                )


def render_source_chunks(docs, collapsed: bool = False, key_prefix: str = "default"):
    """
    Display retrieved source chunks below the assistant's answer.

    WHY show source chunks?
      - Assignment explicitly requires it
      - Allows user to verify the answer is grounded in the document
      - Builds trust — user can see WHERE the answer came from
      - Expander keeps the UI clean (hidden by default in history)

    Args:
        docs       : List of retrieved Document objects.
        collapsed  : Whether to collapse the expander by default.
        key_prefix : Unique prefix — kept for future extensibility.
    """
    if not docs:
        return

    with st.expander(
        f"📄 Source Passages Used ({len(docs)} chunks)",
        expanded=not collapsed
    ):
        for i, doc in enumerate(docs):
            st.markdown(f"**{doc.metadata.get('source', f'Chunk {i+1}')}**")

            # WHY st.markdown instead of st.text_area?
            # st.text_area needs a unique key — causes duplicate key errors when
            # same chunks appear in multiple messages in chat history.
            # st.markdown is stateless — no key needed, no conflicts ever.
            st.markdown(
                f"<div style='background-color:#1e1e1e; padding:12px; "
                f"border-radius:6px; font-size:0.85em; color:#ccc; "
                f"white-space:pre-wrap;'>{doc.page_content}</div>",
                unsafe_allow_html=True,
            )

            if i < len(docs) - 1:
                st.divider()


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""

    # Initialize session state
    init_session_state()

    # Load retriever (cached)
    retriever = load_retriever()

    # Render sidebar
    render_sidebar(retriever)

    # Main chat area header
    st.title("⚖️ eBay User Agreement Chatbot")
    st.caption(
        "Ask questions about eBay's Terms & Conditions. "
        "Answers are grounded strictly in the official User Agreement document."
    )
    st.divider()

    # Show welcome message if no chat history yet
    if not st.session_state.messages:
        st.info(
            "👋 Hello! I can answer questions about eBay's User Agreement. "
            "Try asking about arbitration, fees, returns, or buyer/seller policies."
        )

    # Render existing chat history
    render_chat_history()

    # ── Chat Input ────────────────────────────────────────────────────────────
    # st.chat_input stays fixed at the bottom of the page
    user_query = st.chat_input("Ask a question about the eBay User Agreement...")

    if user_query:
        # 1. Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Save user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "sources": [],
        })

        # 3. Retrieve relevant chunks
        with st.spinner("Searching document..."):
            retrieved_docs = retriever.retrieve(user_query)

        # 4. Stream assistant response
        with st.chat_message("assistant"):

            # st.write_stream() accepts a generator and displays tokens as they arrive
            # This is what creates the "typing" streaming effect
            # WHY use st.write_stream?
            #   It's Streamlit's native streaming API — handles token buffering,
            #   rendering, and scroll behavior automatically.
            response_stream = generate_response(
                query=user_query,
                docs=retrieved_docs,
                streaming=True,
            )

            # Stream tokens to UI — collects full response as return value
            full_response = st.write_stream(response_stream)

            # 5. Show source chunks below the streamed answer
            # key_prefix="current" — this render is always for the latest response only
            render_source_chunks(retrieved_docs, collapsed=False, key_prefix="current")

        # 6. Save assistant message + sources to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": retrieved_docs,
        })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()