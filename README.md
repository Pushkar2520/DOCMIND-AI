# DocuMind: High-Performance Hybrid RAG Chatbot

DocuMind is a Retrieval-Augmented Generation (RAG) chatbot designed to run locally on CPU hardware and answer complex technical queries from a large private corpus of PDFs (over 4,000 pages). It achieves an end-to-end response latency of **2–5 seconds** through a combined single-pass generation/verification pipeline and a lightweight Cross-Encoder reranking model.

---

## 🚀 Architectural Approach & Latency Rationale

To build a RAG chatbot capable of answering questions from thousands of PDF pages in under 5 seconds on **CPU hardware**, we optimized the three primary latency bottlenecks:

```
[User Query]
     │
     ├──► Vector Search (ChromaDB - Cosine Similarity) ──┐
     │                                                   ├──► Reciprocal Rank Fusion (RRF) ──► Top 10 Candidates
     └──► Lexical Search (BM25 - Tokenized Index) ───────┘
                                                                │
                                                                ▼
                                                     Cross-Encoder Reranker 
                                                (ms-marco-MiniLM-L-2-v2 on CPU)
                                                                │
                                                                ▼
                                                          Top 3 Chunks
                                                                │
                                                                ▼
                                                     Gemini 2.5 Flash Stream
                                                (Combined Generation & Fact-Check)
                                                                │
                                                                ▼
                                                Streamed Answer + Citations & Badges
```

1. **Embedding & DB Search (Latency: ~100ms)**
   - Fusing Dense Vector Search (ChromaDB) and Sparse Lexical Search (BM25) ensures high recall of both semantic topics and precise keyword matches.
2. **CPU Reranking (Latency: ~200–500ms)**
   - Instead of using heavy 6-layer or 12-layer Cross-Encoders, we used `cross-encoder/ms-marco-MiniLM-L-2-v2` (a 2-layer model) and reduced the candidate reranking pool from 20 to 10. This achieves a **3x speedup** on CPU while retaining high reranking quality.
3. **Combined LLM Generation & Verification (Latency: ~3–5s, Zero Perceived Latency)**
   - Running a separate fact-checking call on Gemini added 7 seconds of latency and depleted free-tier API quotas. We **combined generation and verification into a single LLM request**. The model generates the answer and immediately appends a JSON fact-verification block.
   - We built a custom stream parser that feeds the answer text to the UI as it streams in real-time (first token in **~1.0 second**), then extracts the verification block behind the scenes once the stream completes.

---

## 🛠️ Tech Stack & Rationale

* **User Interface**: **Streamlit**
  - Allows rapid dashboard construction, native markdown rendering, and real-time text streaming via WebSockets.
* **Vector Database**: **ChromaDB**
  - Lightweight, persistent, file-based vector store. It integrates cleanly in Python and supports HNSW index structures with cosine similarity.
* **Lexical Index**: **Rank-BM25**
  - A lightweight, pure-python BM25 index serialized to a pickle file for sub-millisecond keyword matches.
* **Embedding Model**: **SentenceTransformers (`all-MiniLM-L6-v2`)**
  - Compact (384-dimensional) embedding model. It runs fast on CPU and represents semantic concepts with high density.
* **Reranker Model**: **SentenceTransformers (`ms-marco-MiniLM-L-2-v2`)**
  - Extremely fast 2-layer reranker that runs locally on CPU and matches query-passage relevance with high precision.
* **LLM**: **Google Gemini 2.5 Flash**
  - Extremely fast hosted LLM with a generous free tier (5 RPM / 15 RPM).
* **PDF Extraction**: **PyMuPDF (`fitz`) & Tesseract OCR**
  - PyMuPDF is the fastest native text extractor available. For scanned pages/images, we render the page at 150 DPI and run Tesseract OCR (`pytesseract`).

---

## 📂 Project Structure & File Purpose

```
├── Docs/                       # Folder containing raw corpus PDFs
├── data/
│   ├── chroma_db/              # ChromaDB vector index files
│   ├── bm25_index.pkl          # Serialized BM25 index & chunk mapping
│   ├── corpus.db               # SQLite database caching document texts, pages, and chunks
│   └── monitoring.db           # SQLite database logging queries & latencies
├── src/
│   ├── ingest.py               # Native/OCR PDF text extraction & cleaning (SQLite DB)
│   ├── index.py                # Chunker, vector embedding, and indexing (Batch & Incremental)
│   ├── retrieve.py             # Hybrid search (RRF) and Cross-Encoder reranker
│   └── rag.py                  # Gemini API wrapper with combined stream verification
├── scratch/
│   ├── check_pdfs.py           # Pre-ingestion diagnostic script
│   ├── test_pipeline.py        # CLI test suite for the RAG engine
│   └── debug_rag.py            # Stream tracing and API debugging script
├── app.py                      # Main Streamlit web application & dashboard
├── .env                        # Environment configurations (API Keys)
├── requirements.txt            # System dependencies
└── README.md                   # System documentation (this file)
```

---

## 🧬 Core Pipelines & Implementations

### 1. Ingestion & Chunking
- **Parsing**: `fitz` extracts native text. If a page contains zero native text blocks or words and has images, Tesseract OCR is triggered. Page extraction and OCR are parallelized using a `ProcessPoolExecutor` across multiple CPU cores.
- **Incremental Cache**: Stored in a local SQLite database (`data/corpus.db`) using MD5 file hashing. Unaltered PDFs are skipped instantly. Deleted/modified documents are purged and updated incrementally.
- **Chunking**: LangChain's `RecursiveCharacterTextSplitter` chunks the text into `600` character passages with a `120` character overlap to preserve semantic context.
- **Streaming Vectors**: Generated embeddings are streamed to ChromaDB in batches of `128` to maintain a low memory profile, avoiding keeping all float lists in RAM.
- **BM25 Rebuild**: Recreated on index updates by loading metadata directly from the SQLite `chunks` table.

### 2. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
To query our corpus, we retrieve candidates from both Vector Search and BM25 Search. We combine the lists using Reciprocal Rank Fusion (RRF):

$$RRF(d) = \frac{1}{60 + r_{vector}(d)} + \frac{1}{60 + r_{bm25}(d)}$$

Where $r_{vector}(d)$ is the 1-based rank of document $d$ in the vector results, and $r_{bm25}(d)$ is its rank in the BM25 results. Candidates are sorted by their fused $RRF(d)$ score.

### 3. Cross-Encoder Reranking
Dense vector embeddings sometimes fail on keyword structures, and BM25 fails on semantic overlap. RRF aligns them, and our local **Cross-Encoder** (`ms-marco-MiniLM-L-2-v2`) scores the candidate chunks against the user query directly:

$$\text{Rerank Score} = \text{CrossEncoder}([\text{Query}, \text{Chunk}])$$

We sort by this score and feed the **top 3** chunks to the LLM.

### 4. Combined RAG Prompt Strategy
We query Gemini 2.5 Flash using a specialized prompt that forces both generation and fact validation in a single call:

```
FORMAT EXAMPLE:
[ANSWER]
Python is...
Sources:
📄 python_programming_guide.pdf (Page 21)
[VALIDATION]
{
  "status": "FULLY_SUPPORTED",
  "confidence": 100,
  "explanation": "..."
}
```

Our custom stream parser isolates the `[ANSWER]` block and streams it to the user in real-time, while accumulating the `[VALIDATION]` block in the background. Once the stream completes, the validation JSON is parsed.

### 5. Hallucination Detection & Fallback
If the validation status parsed from the JSON is `"UNSUPPORTED"` (e.g., when the LLM flags that its answer is not backed by the chunks, or when the query is out-of-scope), the UI overrides the response with:
`"Information not found in documents."`

If a system or rate limit error occurs, the status becomes `"API_ERROR"`, displaying the rate limit warning rather than claiming the information is missing.

### 6. Confidence Scoring System
We compute an overall confidence level (High/Medium/Low) by combining:
1. **Retrieval Confidence**: Sigmoided maximum reranker score ($C_{retrieval} = \frac{100}{1 + e^{-s_{rerank}}}$).
2. **Validation Confidence**: The self-reported LLM confidence score ($C_{validation}$).

$$\text{Overall Confidence} = 0.3 \times C_{retrieval} + 0.7 \times C_{validation}$$

- **High**: $\ge 75\%$ (requires `FULLY_SUPPORTED` status).
- **Medium**: $40\% - 74\%$.
- **Low**: $< 40\%$.

---

## ⚙️ Installation & Setup

1. **Install Tesseract OCR**:
   Ensure Tesseract OCR is installed on Windows at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
2. **Install Dependencies**:
   ```bash
   uv pip install --system chromadb rank-bm25 python-dotenv pytesseract sentencepiece sentence-transformers langchain langchain-text-splitters streamlit
   ```
3. **Configure API Key**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 🏃 Running the Application

1. **Ingest PDFs**:
   ```bash
   python src/ingest.py
   ```
2. **Build Search Indices**:
   ```bash
   python src/index.py
   ```
3. **Run RAG Tests**:
   ```bash
   python scratch/test_pipeline.py
   ```
4. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 💡 Troubleshooting Rate Limits (429)

If you are using a Free Tier Gemini API key and hit rate limits, look for the **🔑 API Key Configuration** input field in the Streamlit sidebar. You can paste a new key there directly from the browser to continue querying.
