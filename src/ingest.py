"""
ingest.py
---------
Responsible for:
  1. Loading the PDF document using PyMuPDF
  2. Cleaning the raw extracted text
  3. Splitting into sentence-aware chunks (100-300 words)
  4. Saving chunks to /chunks/ directory as JSON

WHY this module exists separately:
  Ingestion is a one-time preprocessing step. Keeping it separate means
  we don't re-process the PDF every time the app starts — we just load
  the saved chunks from disk.
"""

import os
import re
import json

import fitz  # PyMuPDF — imported as 'fitz' (historical name from the library)
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── Constants ─────────────────────────────────────────────────────────────────

# Path to the input PDF document
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Path where processed chunks will be saved as JSON
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "chunks")

# Chunk size in characters (~200 words × 5 chars/word = ~1000 chars)
# We target 100-300 words as per assignment — 1000 chars ≈ 200 words (safe middle ground)
CHUNK_SIZE = 1000

# Overlap between consecutive chunks — ensures context isn't lost at boundaries
# Example: if chunk 1 ends mid-sentence about "arbitration", chunk 2 starts
# with that same context so retrieval doesn't miss it
CHUNK_OVERLAP = 150


def load_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.

    WHY PyMuPDF over pypdf?
      PyMuPDF handles complex PDFs (multi-column, legal formatting) more reliably.
      It preserves paragraph structure better than pypdf.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        A single string containing all text from all pages.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)

    full_text = []

    for page_num, page in enumerate(doc):
        # Extract text from the current page
        # "text" mode returns plain text — better than "html" for our use case
        page_text = page.get_text("text")
        full_text.append(page_text)

    # Close the document to free memory
    doc.close()

    # Join all pages with a newline separator
    return "\n".join(full_text)


def clean_text(raw_text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    PDF extraction often produces noisy text:
      - Multiple spaces between words
      - Excessive newlines (from page layout)
      - Bullet point symbols that become garbled characters
      - Page numbers appearing as standalone lines

    WHY clean before chunking?
      Dirty text → bad chunks → irrelevant embeddings → wrong retrieval results.
      Garbage in, garbage out.

    Args:
        raw_text: Raw string from PDF extraction.

    Returns:
        Cleaned, normalized text string.
    """
    # Replace multiple consecutive whitespace/newlines with a single space
    # This collapses the page-layout artifacts into readable paragraphs
    text = re.sub(r'\s+', ' ', raw_text)

    # Remove standalone page numbers (e.g., " 1 ", " 12 ")
    text = re.sub(r'\s\d{1,3}\s', ' ', text)

    # Remove common bullet point unicode characters that appear after PDF extraction
    text = re.sub(r'[•·▪▸►●]', '', text)

    # Remove any non-ASCII characters that are just noise (garbled encoding)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Strip leading/trailing whitespace from the final string
    text = text.strip()

    return text


def chunk_text(clean_text: str) -> list[dict]:
    """
    Split the cleaned text into overlapping chunks using sentence-aware splitting.

    WHY RecursiveCharacterTextSplitter?
      It tries to split at natural language boundaries in this priority order:
        1. Paragraph breaks (\n\n)
        2. Sentence ends (.\n or . )
        3. Clause breaks (, )
        4. Word boundaries (space)
        5. Character level (last resort)

      This ensures chunks don't cut mid-sentence, preserving semantic meaning.
      A legal document like eBay's ToS has long sentences — we need this.

    WHY 1000 char chunk size?
      100-300 words × ~5 chars/word = 500-1500 chars.
      1000 chars ≈ 200 words — sits in the sweet spot.
      Too small → loses context. Too large → irrelevant info gets retrieved.

    WHY 150 char overlap?
      Legal clauses often reference the previous clause.
      Overlap ensures the model sees connecting context across chunk boundaries.

    Args:
        clean_text: The cleaned full document text.

    Returns:
        List of dicts, each with:
          - 'chunk_id'  : unique integer identifier
          - 'text'      : the chunk content
          - 'word_count': approximate word count for verification
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separator priority list — tries each in order before falling back
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        # Keep separators attached to the preceding chunk for readability
        keep_separator=True,
    )

    # Split the text — returns a list of string chunks
    raw_chunks = splitter.split_text(clean_text)

    # Wrap each chunk in a dict with metadata
    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": idx,
            "text": chunk.strip(),
            # Approximate word count — useful for debugging chunk size distribution
            "word_count": len(chunk.split()),
        })

    return chunks


def save_chunks(chunks: list[dict], filename: str = "chunks.json") -> str:
    """
    Save the processed chunks to a JSON file in the /chunks/ directory.

    WHY save as JSON?
      Human-readable, easy to inspect during debugging.
      Can be loaded later without re-processing the PDF.

    Args:
        chunks   : List of chunk dicts from chunk_text().
        filename : Output filename (default: chunks.json).

    Returns:
        Full path to the saved file.
    """
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    output_path = os.path.join(CHUNKS_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path


def load_chunks(filename: str = "chunks.json") -> list[dict]:
    """
    Load previously saved chunks from the /chunks/ directory.

    WHY load from disk instead of re-processing?
      PDF processing + chunking takes time. On every app restart, we skip
      this step and load the pre-computed chunks directly.

    Args:
        filename: Name of the JSON file to load.

    Returns:
        List of chunk dicts.
    """
    chunks_path = os.path.join(CHUNKS_DIR, filename)

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            "Run the ingestion pipeline first: python src/ingest.py"
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ingestion(pdf_filename: str = "AI_Training_Document.pdf") -> list[dict]:
    """
    Full ingestion pipeline: PDF → clean text → chunks → saved JSON.

    This is the main entry point called by embedder.py and also
    runnable directly as a script for one-time preprocessing.

    Args:
        pdf_filename: Name of the PDF file inside the /data/ directory.

    Returns:
        List of chunk dicts ready for embedding.
    """
    pdf_path = os.path.join(DATA_DIR, pdf_filename)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. "
            "Place the document inside the /data/ directory."
        )

    print(f"[Ingest] Loading PDF from: {pdf_path}")
    raw_text = load_pdf(pdf_path)
    print(f"[Ingest] Extracted {len(raw_text):,} characters of raw text")

    print("[Ingest] Cleaning text...")
    cleaned = clean_text(raw_text)
    print(f"[Ingest] After cleaning: {len(cleaned):,} characters")

    print("[Ingest] Chunking text...")
    chunks = chunk_text(cleaned)
    print(f"[Ingest] Created {len(chunks)} chunks")

    # Print word count distribution for verification
    word_counts = [c["word_count"] for c in chunks]
    print(f"[Ingest] Word count per chunk — "
          f"min: {min(word_counts)}, "
          f"max: {max(word_counts)}, "
          f"avg: {sum(word_counts) // len(word_counts)}")

    output_path = save_chunks(chunks)
    print(f"[Ingest] Chunks saved to: {output_path}")

    return chunks


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run this file directly to preprocess the document:
    # $ python src/ingest.py
    chunks = run_ingestion()
    print(f"\n[Done] {len(chunks)} chunks ready for embedding.")