"""
retriever.py
------------
Responsible for:
  1. Loading the FAISS index (or building it if not present)
  2. Taking a user query and finding the most semantically relevant chunks
  3. Returning those chunks to the generator with their metadata

WHY a separate retriever module?
  Clean separation of concerns:
    - embedder.py  → builds and saves the index (one-time)
    - retriever.py → loads and queries the index (every request)
  
  This means we can swap out the retrieval strategy (e.g., MMR instead of
  similarity search) without touching the embedding or generation logic.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Import from our own modules
from embedder import get_embedding_model, load_faiss_index, run_embedding_pipeline, index_exists


# ── Constants ─────────────────────────────────────────────────────────────────

# Number of chunks to retrieve per query
# WHY 4?
#   - Too few (1-2) → model might miss important context
#   - Too many (8+) → prompt gets too long, irrelevant chunks confuse the model
#   - 4 is the sweet spot for a 10,500-word legal document
TOP_K = 4


class Retriever:
    """
    Semantic retriever that finds the most relevant document chunks
    for a given user query using FAISS similarity search.

    Usage:
        retriever = Retriever()
        results = retriever.retrieve("What is eBay's arbitration policy?")
    """

    def __init__(self, top_k: int = TOP_K):
        """
        Initialize the retriever by loading (or building) the FAISS index.

        WHY load in __init__ instead of on every query?
          Loading the index from disk takes ~1-2 seconds. If we did it on
          every query, the chatbot would feel slow. Loading once at startup
          keeps all subsequent queries fast (<100ms retrieval time).

        Args:
            top_k: Number of chunks to return per query.
        """
        self.top_k = top_k

        # Load the embedding model — needed for query embedding at retrieval time
        # The SAME model must be used for both indexing and querying.
        # Different models produce different vector spaces — mixing them breaks retrieval.
        self.embeddings = get_embedding_model()

        # Load or build the FAISS index
        if index_exists():
            self.vectorstore = load_faiss_index(self.embeddings)
        else:
            print("[Retriever] No existing index found — building from scratch...")
            self.vectorstore = run_embedding_pipeline()

        print(f"[Retriever] Ready — will return top-{self.top_k} chunks per query")

    def retrieve(self, query: str) -> list[Document]:
        """
        Find the most semantically relevant chunks for the user's query.

        HOW it works (explain in interview):
          1. The query string is passed through all-MiniLM-L6-v2
             → produces a 384-dim query vector
          2. FAISS computes L2 distance (or cosine, since we normalized)
             between the query vector and ALL stored chunk vectors
          3. Returns the top_k Documents with smallest distance
             (= most semantically similar to the query)

        WHY semantic search over keyword search?
          Keyword search (BM25) would match "arbitration" only if the exact
          word appears. Semantic search understands that "dispute resolution"
          and "arbitration" are related concepts — better for legal documents
          where terminology varies.

        Args:
            query: The user's natural language question.

        Returns:
            List of up to top_k LangChain Document objects, each containing:
              - page_content : the chunk text
              - metadata     : chunk_id, word_count, source label
        """
        if not query.strip():
            return []

        print(f"[Retriever] Searching for: '{query[:80]}...' " if len(query) > 80 else f"[Retriever] Searching for: '{query}'")

        # similarity_search embeds the query and finds nearest neighbors in FAISS
        results = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k,
        )

        print(f"[Retriever] Found {len(results)} relevant chunks")

        # Log chunk IDs for debugging
        for doc in results:
            print(f"  → Chunk #{doc.metadata['chunk_id']} "
                  f"({doc.metadata['word_count']} words)")

        return results

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """
        Same as retrieve() but also returns similarity scores.

        WHY have this variant?
          Scores let us filter out low-confidence results.
          If the best matching chunk has a very low score, the document
          might not contain the answer — we can tell the user that.

          Also useful for the PDF report: we can show relevance scores
          as evidence of retrieval quality.

        Args:
            query: The user's question.

        Returns:
            List of (Document, score) tuples, sorted by score descending.
            Higher score = more similar (since we use normalized cosine similarity).
        """
        if not query.strip():
            return []

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.top_k,
        )

        # Log scores for debugging
        for doc, score in results:
            print(f"  → Chunk #{doc.metadata['chunk_id']} | Score: {score:.4f}")

        return results

    def get_chunk_count(self) -> int:
        """
        Return total number of indexed chunks.

        Used by the Streamlit sidebar to display:
        "📚 Total indexed chunks: 87"

        Returns:
            Integer count of vectors in the FAISS index.
        """
        return self.vectorstore.index.ntotal


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test the retriever directly:
    # $ python src/retriever.py
    retriever = Retriever()

    test_queries = [
        "What is the arbitration policy?",
        "Can I return items purchased on eBay?",
        "What fees does eBay charge sellers?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        docs = retriever.retrieve(query)
        for i, doc in enumerate(docs):
            print(f"\n[Result {i+1}] {doc.metadata['source']}")
            print(doc.page_content[:200] + "...")
