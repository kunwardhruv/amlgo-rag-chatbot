"""
embedder.py
-----------
Responsible for:
  1. Loading text chunks from /chunks/
  2. Generating vector embeddings using all-MiniLM-L6-v2
  3. Building a FAISS index from those embeddings
  4. Saving the FAISS index to /vectordb/ for reuse

WHY this module is separate from retriever.py:
  Embedding generation is also a one-time step (like ingestion).
  We build the index ONCE and save it. The retriever just LOADS and QUERIES it.
  This separation keeps responsibilities clean and startup time fast.
"""

import os
import json

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from ingest import load_chunks, run_ingestion, CHUNKS_DIR


# ── Constants ─────────────────────────────────────────────────────────────────

# Where the FAISS index will be saved
VECTORDB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectordb")

# The embedding model we use — runs locally, no API key needed
# WHY all-MiniLM-L6-v2?
#   - Small (80MB), fast inference on CPU
#   - 384-dimensional vectors — good balance of quality vs. speed
#   - Trained on 1B+ sentence pairs — excellent semantic similarity
#   - Assignment explicitly mentions this model as a valid choice
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Name of the saved FAISS index folder inside /vectordb/
FAISS_INDEX_NAME = "faiss_index"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the HuggingFace embedding model.

    WHY HuggingFaceEmbeddings wrapper instead of raw sentence-transformers?
      LangChain's HuggingFaceEmbeddings integrates directly with FAISS vector store.
      It handles batching, encoding, and normalization automatically.

    WHY model_kwargs device='cpu'?
      We're targeting local deployment without GPU.
      CPU inference is perfectly fine for a 384-dim model on a small document.

    Returns:
        Initialized HuggingFaceEmbeddings instance.
    """
    print(f"[Embedder] Loading embedding model: {EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # Run on CPU — no GPU required
        encode_kwargs={
            "normalize_embeddings": True,  # L2-normalize so cosine similarity = dot product
            "batch_size": 32,              # Process 32 chunks at a time — balances speed/memory
        },
    )

    print("[Embedder] Embedding model loaded successfully")
    return embeddings


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """
    Convert raw chunk dicts into LangChain Document objects.

    WHY LangChain Document?
      FAISS vector store expects Document objects — they hold both:
        - page_content : the actual text (used for embedding + display)
        - metadata     : extra info we attach (chunk_id, word_count)

      The metadata is returned during retrieval so we can show the user
      WHICH chunk was used to answer their question (as required by assignment).

    Args:
        chunks: List of chunk dicts from ingest.py.

    Returns:
        List of LangChain Document objects.
    """
    documents = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "word_count": chunk["word_count"],
                # Source label shown in the Streamlit UI under each retrieved chunk
                "source": f"eBay User Agreement – Chunk #{chunk['chunk_id']}",
            },
        )
        documents.append(doc)

    return documents


def build_faiss_index(documents: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build a FAISS vector store from document embeddings.

    HOW FAISS works (interview answer):
      1. Each Document's page_content is passed through the embedding model
         → produces a 384-dimensional float vector
      2. FAISS stores all vectors in an index (using IndexFlatL2 by default)
      3. At query time: query text → embed → find k nearest vectors by L2/cosine distance
      4. Return the Documents corresponding to those nearest vectors

    WHY FAISS over ChromaDB?
      - No server / no database process needed — pure in-memory + file
      - Faster for small-medium corpora (< 100k docs)
      - We had ChromaDB tenant state bugs in a previous project — FAISS is simpler
      - LangChain's FAISS wrapper handles everything with 2 lines of code

    Args:
        documents  : LangChain Document objects with text + metadata.
        embeddings : Initialized embedding model.

    Returns:
        FAISS vector store instance (in memory, ready for similarity search).
    """
    print(f"[Embedder] Building FAISS index from {len(documents)} documents...")
    print("[Embedder] This may take 30-60 seconds on first run (embedding generation)...")

    # FAISS.from_documents() does 3 things internally:
    #   1. Calls embeddings.embed_documents([doc.page_content for doc in documents])
    #   2. Creates a FAISS index and adds all vectors
    #   3. Maintains a mapping from vector index → Document for retrieval
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    print(f"[Embedder] FAISS index built with {vectorstore.index.ntotal} vectors")
    return vectorstore


def save_faiss_index(vectorstore: FAISS, embeddings: HuggingFaceEmbeddings) -> str:
    """
    Save the FAISS index to disk so it can be reloaded without re-embedding.

    WHY save to disk?
      Embedding 100+ chunks takes 30-60 seconds. On every app restart, we'd
      wait for this. Instead, save once → load instantly every time after.

    LangChain's save_local() saves TWO files:
      - index.faiss : the actual FAISS binary index (vectors + index structure)
      - index.pkl   : Python pickle of the Document objects + metadata mapping

    Args:
        vectorstore : Built FAISS instance.
        embeddings  : Embedding model (needed for future similarity searches).

    Returns:
        Path to the saved index directory.
    """
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    index_path = os.path.join(VECTORDB_DIR, FAISS_INDEX_NAME)

    vectorstore.save_local(index_path)
    print(f"[Embedder] FAISS index saved to: {index_path}")

    return index_path


def load_faiss_index(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    WHY allow_dangerous_deserialization=True?
      LangChain loads the .pkl file using Python's pickle module.
      Pickle can execute arbitrary code if the file is tampered with.
      We set this flag to True because WE created this file ourselves —
      it's not coming from an untrusted source.

    Args:
        embeddings: The SAME embedding model used to build the index.
                    Must match — different models produce incompatible vector spaces.

    Returns:
        Loaded FAISS vector store ready for similarity_search().
    """
    index_path = os.path.join(VECTORDB_DIR, FAISS_INDEX_NAME)

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run the embedding pipeline first: python src/embedder.py"
        )

    print(f"[Embedder] Loading FAISS index from: {index_path}")

    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # Safe — we built this file ourselves
    )

    print(f"[Embedder] Loaded {vectorstore.index.ntotal} vectors from index")
    return vectorstore


def index_exists() -> bool:
    """
    Check if a saved FAISS index already exists on disk.

    Used by app.py and retriever.py to decide whether to build or load.

    Returns:
        True if the index exists, False otherwise.
    """
    index_path = os.path.join(VECTORDB_DIR, FAISS_INDEX_NAME)
    # FAISS saves two files: index.faiss and index.pkl
    # Check for index.faiss as the indicator
    return os.path.exists(os.path.join(index_path, "index.faiss"))


def run_embedding_pipeline() -> FAISS:
    """
    Full embedding pipeline:
      load chunks → convert to Documents → embed → build FAISS → save

    Entry point for one-time setup. After this runs once, the app
    uses load_faiss_index() on every subsequent startup.

    Returns:
        Built and saved FAISS vector store.
    """
    # Step 1: Load chunks (run ingestion if chunks don't exist yet)
    try:
        chunks = load_chunks()
        print(f"[Embedder] Loaded {len(chunks)} existing chunks")
    except FileNotFoundError:
        print("[Embedder] Chunks not found — running ingestion first...")
        chunks = run_ingestion()

    # Step 2: Convert to LangChain Document objects
    documents = chunks_to_documents(chunks)

    # Step 3: Load the embedding model
    embeddings = get_embedding_model()

    # Step 4: Build the FAISS index
    vectorstore = build_faiss_index(documents, embeddings)

    # Step 5: Save to disk
    save_faiss_index(vectorstore, embeddings)

    return vectorstore


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run this file directly to build and save the vector index:
    # $ python src/embedder.py
    vectorstore = run_embedding_pipeline()
    print(f"\n[Done] FAISS index ready with {vectorstore.index.ntotal} vectors.")
