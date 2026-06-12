import os
import json
import re
import pickle
import torch
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Optimize CPU threads for local inference
torch.set_num_threads(4)

CLEANED_DOCS_PATH = "data/cleaned_docs.json"
CHROMA_DB_PATH = "data/chroma_db"
BM25_INDEX_PATH = "data/bm25_index.pkl"

def is_low_quality_ocr(text):
    """Heuristic to detect and remove garbled OCR text or noise chunks."""
    if not text:
        return True
        
    stripped = text.strip()
    total_len = len(stripped)
    if total_len < 50: # Prune tiny chunks
        return True
        
    # Heuristic 1: Letter density (standard English text should be >40% letters)
    letters = len(re.findall(r'[a-zA-Z]', stripped))
    letter_ratio = letters / total_len
    if letter_ratio < 0.40:
        return True
        
    # Heuristic 2: Vowel ratio (English words typically have at least 15% vowels in their letters)
    vowels = len(re.findall(r'[aeiouAEIOU]', stripped))
    if letters > 0:
        vowel_ratio = vowels / letters
        if vowel_ratio < 0.15:
            return True
            
    # Heuristic 3: Excessive whitespace/noise sequences
    if len(re.findall(r'[\\_#\*\-\|]{4,}', stripped)) > 2:
        return True
        
    return False

def tokenize_bm25(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def build_indices():
    if not os.path.exists(CLEANED_DOCS_PATH):
        print(f"Error: {CLEANED_DOCS_PATH} not found. Run ingest.py first.")
        return
        
    with open(CLEANED_DOCS_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)
        
    print(f"Loaded {len(pages)} pages from {CLEANED_DOCS_PATH}.")
    
    # Chunking using RecursiveCharacterTextSplitter with optimized defaults
    print("Chunking documents using LangChain RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )
    
    chunks = []
    skipped_tiny_count = 0
    skipped_ocr_count = 0
    
    for page in pages:
        text = page["text"]
        if not text.strip():
            continue
            
        sub_chunks = splitter.split_text(text)
        
        for c_idx, chunk_text in enumerate(sub_chunks):
            # 1. Quality Filters: Skip tiny chunks and low-quality OCR
            if len(chunk_text.strip()) < 50:
                skipped_tiny_count += 1
                continue
                
            if page["text_source"] == "ocr" and is_low_quality_ocr(chunk_text):
                skipped_ocr_count += 1
                continue
                
            chunk_id = f"{page['filename']}_p{page['page_number']}_c{c_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "pdf_id": page["pdf_id"],
                "filename": page["filename"],
                "page_number": page["page_number"],
                "text": chunk_text,
                "text_source": page["text_source"]
            })
            
    print(f"Created {len(chunks)} chunks.")
    print(f"  Skipped {skipped_tiny_count} tiny chunks (<50 chars).")
    print(f"  Skipped {skipped_ocr_count} noisy OCR chunks.")
    
    # 2. Embedding using SentenceTransformers directly
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    chunk_texts = [c["text"] for c in chunks]
    print("Computing embeddings...")
    embeddings = model.encode(chunk_texts, batch_size=128, show_progress_bar=True)
    embeddings = [emb.tolist() for emb in embeddings]
    
    # 3. Save to ChromaDB
    print("Initializing ChromaDB Persistent Client...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        chroma_client.delete_collection("pdf_chunks")
        print("Deleted existing ChromaDB collection.")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "pdf_id": c["pdf_id"],
        "filename": c["filename"],
        "page_number": c["page_number"],
        "text_source": c["text_source"]
    } for c in chunks]
    
    print("Writing to ChromaDB...")
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=chunk_texts[i:end_idx]
        )
        print(f"  Indexed chunks {i} to {end_idx}...")
        
    print("ChromaDB vector indexing completed.")
    
    # 4. Build BM25 Index
    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    
    tokenized_corpus = [tokenize_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": chunks
        }, f)
        
    print(f"BM25 index saved to {BM25_INDEX_PATH}.")
    print("Indexing process completed successfully!")

if __name__ == "__main__":
    build_indices()
