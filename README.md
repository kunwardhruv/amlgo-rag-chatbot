# ⚖️ RAG Chatbot — eBay User Agreement Assistant
### Junior AI Engineer Assignment | Amlgo Labs

An AI-powered chatbot that answers questions about eBay's User Agreement using a **Retrieval-Augmented Generation (RAG)** pipeline with **real-time token-by-token streaming responses**.

---

## 📌 Project Architecture & Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   STREAMLIT UI        │  ← app.py
          │   (Chat Interface)    │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │   RETRIEVER           │  ← src/retriever.py
          │   Query → Embed →     │
          │   FAISS Search →      │
          │   Top-4 Chunks        │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │   GENERATOR           │  ← src/generator.py
          │   Chunks + Query →    │
          │   Prompt Template →   │
          │   Groq LLM →          │
          │   Streaming Response  │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │   STREAMED ANSWER     │
          │   + SOURCE CHUNKS     │
          └───────────────────────┘
```

### One-Time Preprocessing Flow (run before first use)
```
PDF Document
    │
    ▼
src/ingest.py  →  PyMuPDF extracts text  →  clean_text()  →  RecursiveCharacterTextSplitter
    │
    ▼
src/embedder.py  →  all-MiniLM-L6-v2 generates 384-dim vectors  →  FAISS index saved to /vectordb/
```

---

## 📁 Folder Structure

```
amlgo-rag-chatbot/
├── data/               ← Input PDF document (eBay User Agreement)
├── chunks/             ← Preprocessed text chunks saved as JSON
├── vectordb/           ← Saved FAISS index (index.faiss + index.pkl)
├── notebooks/          ← Preprocessing & evaluation notebooks
├── src/
│   ├── __init__.py
│   ├── ingest.py       ← PDF loading, cleaning, chunking
│   ├── embedder.py     ← Embedding generation + FAISS index builder
│   ├── retriever.py    ← Semantic search over FAISS index
│   └── generator.py    ← Prompt template + Groq LLM + streaming
├── app.py              ← Streamlit chatbot with streaming UI
├── requirements.txt    ← All dependencies
├── .env.example        ← API key template
└── README.md
```

---

## 🧠 Model & Embedding Choices

### Embedding Model — `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Model size | ~80 MB |
| Vector dimensions | 384 |
| Runs on | CPU (no GPU needed) |
| Training data | 1 billion+ sentence pairs |

**Why this model?**
- Explicitly listed in the assignment as a valid choice
- Lightweight and fast — suitable for local deployment without a GPU
- 384-dimensional vectors strike the right balance between quality and speed
- Excellent semantic similarity performance on domain-specific text like legal documents
- No API key required — runs fully locally via `sentence-transformers`

---

### LLM — `llama-3.3-70b-versatile` via Groq API

| Property | Value |
|---|---|
| Type | Instruction-optimized open-source LLM |
| Parameters | 70 Billion |
| Provider | Groq (free tier) |
| Inference speed | ~500 tokens/second |

**Why this model?**
- LLaMA 3.3 is an instruction-optimized model — fulfills the assignment requirement
- 70B parameters provide strong reasoning over complex legal text
- Groq's API is free-tier accessible and supports native streaming
- No local GPU required — inference is handled by Groq's cloud infrastructure
- Alternatives (running Mistral/Zephyr locally) require 16GB+ RAM — impractical for local dev

---

### Vector Database — `FAISS` (Facebook AI Similarity Search)

**Why FAISS over ChromaDB or Qdrant?**
- No external server or database process required — pure file-based storage
- Extremely fast for our corpus size (88 chunks)
- `save_local()` and `load_local()` make persistence simple
- LangChain's FAISS wrapper integrates cleanly with our pipeline
- Production-ready and battle-tested at scale

---

### Chunking Strategy — `RecursiveCharacterTextSplitter`

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 1000 characters | ≈ 200 words — middle of the 100–300 word range |
| `chunk_overlap` | 150 characters | Preserves context across chunk boundaries |
| Separators | `\n\n`, `\n`, `. `, `, `, ` ` | Tries paragraph → sentence → word breaks in order |

**Why sentence-aware splitting?**
Legal documents have long, interconnected clauses. Cutting mid-sentence would break the semantic meaning of chunks, leading to poor retrieval quality. `RecursiveCharacterTextSplitter` prioritizes splitting at natural language boundaries.

**Result:** 88 chunks from the eBay User Agreement (67,262 characters of cleaned text)

---

## 🚀 Setup & Run Instructions

### Prerequisites
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository
```bash
git clone https://github.com/kunwardhruv/amlgo-rag-chatbot
cd amlgo-rag-chatbot
```

### Step 2 — Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure API key
```bash
# Copy the template
cp .env.example .env

# Open .env and add your Groq API key
# GROQ_API_KEY=your_key_here
```

### Step 5 — Add the document
Place the eBay User Agreement PDF inside the `/data/` folder:
```
data/
└── AI_Training_Document.pdf
```

### Step 6 — Run preprocessing (one-time only)
```bash
# Ingest PDF → clean → chunk → save to /chunks/
python src/ingest.py

# Generate embeddings → build FAISS index → save to /vectordb/
python src/embedder.py
```

Expected output:
```
[Ingest] Created 88 chunks
[Ingest] Word count per chunk — min: 55, max: 177, avg: 130
[Embedder] FAISS index built with 88 vectors
```

### Step 7 — Launch the chatbot with streaming
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

Streaming is enabled by default — responses appear token-by-token as the model generates them.

---

## 💬 Sample Queries & Expected Outputs

| # | Query | Result | Notes |
|---|---|---|---|
| 1 | *"What is eBay's arbitration policy?"* | ✅ Success | Accurately cites Section 19.B with process details |
| 2 | *"How does eBay Money Back Guarantee work?"* | ✅ Success | Covers claim process and seller reimbursement correctly |
| 3 | *"How can I opt out of arbitration?"* | ✅ Success | Returns exact mailing address and 30-day deadline |
| 4 | *"What fees does eBay charge sellers?"* | ✅ Success | References selling fees and payment method requirements |
| 5 | *"What is eBay's climate change policy?"* | ❌ Expected failure | Correctly responds: "not covered in the provided document" |

---

## 🎥 Demo
> ## Video link 
> 📹 **https://drive.google.com/file/d/1VIrY_DiyR75Erln6p_9jkGKSeOCQpWRG/view?usp=sharing**

---

## Screenshots 

> ### Welcome Screen

<img width="1918" height="962" alt="welcome" src="https://github.com/user-attachments/assets/af48a7b3-72fd-482f-bf04-2eebace93046" />
<br><br>

> ### Sidebar — Model Info & 88 Indexed Chunks

<img width="1919" height="960" alt="sidebar" src="https://github.com/user-attachments/assets/6976d26b-f4b2-401a-8341-d3771ddf2409" />
<br><br>

> ### Query: What is eBay's arbitration policy? — Success Case

<img width="1920" height="973" alt="arbitration" src="https://github.com/user-attachments/assets/d73c8af4-8418-472a-9b3e-c788585e2761" />
<br><br>

> ### Query: Can sellers charge outside of eBay? — Success Case with Source Chunks

<img width="1920" height="972" alt="sellers" src="https://github.com/user-attachments/assets/88596073-61e7-4536-9f78-a6d8d32917c4" />
<img width="1919" height="972" alt="sellers" src="https://github.com/user-attachments/assets/5f7c7854-f649-49f4-bd17-4def7fcf57a6" />
<br><br>

> ### Query: How can I opt out of arbitration? — Source Chunks Expanded

<img width="1919" height="830" alt="opt-out" src="https://github.com/user-attachments/assets/604ce2cf-8ab7-4d96-82f2-767bc5957185" />
<br><br>

> ### Query: Money Back Guarantee ✅ & Climate Change ❌ — Success + Failure Case

<img width="1918" height="962" alt="success-fail" src="https://github.com/user-attachments/assets/ca328ffa-fd0e-40da-9f33-ed3490ad3453" />

---

## ⚠️ Known Limitations & Hallucination Notes

- **Chunk boundary issue:** Very specific numerical data (e.g., exact fee percentages) may sometimes be split across chunks. If the answer spans two chunks and only one is retrieved, the response may be incomplete.
- **Groq rate limits:** The free tier allows ~30 requests/minute. Heavy usage may trigger temporary throttling.
- **No document update detection:** If the source PDF changes, the FAISS index must be manually rebuilt by re-running `src/embedder.py`.
- **Prompt grounding:** The system prompt strictly instructs the model to use only retrieved context. However, on very vague queries, the model may occasionally blend context with general knowledge — this is logged as a known limitation.

---

## 📊 Evaluation Criteria Coverage

| Criteria | Weight | Implementation |
|---|---|---|
| Functionality & Integration | 30% | Full RAG pipeline: ingest → embed → retrieve → generate |
| Streaming Output | 20% | `chain.stream()` → `st.write_stream()` — token-by-token |
| Code Quality & Modularity | 20% | 4 separate modules, detailed comments explaining every decision |
| Grounded & Accurate Answers | 20% | Strict system prompt + source chunk display for verification |
| App Usability & UX | 10% | Clean sidebar, chat history, clear button, source expander |
