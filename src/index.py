import os
import re
import pickle
import sqlite3
import torch
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# Optimize CPU threads for local inference
torch.set_num_threads(4)

# Load configuration paths
DOCS_DIR = os.getenv("DOCS_DIR", "Docs")
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "corpus.db")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

def is_low_quality_ocr(text):
    """Heuristic to detect and remove garbled OCR text or noise chunks."""
    if not text:
        return True
        
    stripped = text.strip()
    total_len = len(stripped)
    if total_len < 50:
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
    if not os.path.exists(DB_PATH):
        print(f"Error: Database '{DB_PATH}' not found. Run ingest.py first.")
        return
        
    # Initialize ChromaDB client and collection
    print("Initializing ChromaDB Persistent Client...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    collection = chroma_client.get_or_create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # 1. Reconcile deleted documents in ChromaDB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM documents")
    active_files = set(row[0] for row in cursor.fetchall())
    
    # Get all metadatas from Chroma to check what files it holds
    chroma_results = collection.get(include=["metadatas"])
    if chroma_results and chroma_results["metadatas"]:
        chroma_files = set(m["filename"] for m in chroma_results["metadatas"] if m and "filename" in m)
        deleted_files = chroma_files - active_files
        for del_file in deleted_files:
            print(f"Purging deleted document from ChromaDB: {del_file}")
            collection.delete(where={"filename": del_file})
            
    # 2. Find new/modified documents that need chunking and indexing
    # We identify documents that have pages in SQLite but no entries in the chunks table.
    cursor.execute("""
        SELECT pdf_id, filename FROM documents 
        WHERE pdf_id NOT IN (SELECT DISTINCT pdf_id FROM chunks)
    """)
    docs_to_index = cursor.fetchall()
    
    if not docs_to_index:
        print("No new or modified documents require indexing.")
    else:
        print(f"Found {len(docs_to_index)} documents that need chunking and indexing.")
        
        # Load SentenceTransformer model
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        
        for pdf_id, filename in docs_to_index:
            print(f"Processing '{filename}'...")
            
            # Delete any existing vectors in Chroma for safety (incase of partial past runs)
            collection.delete(where={"filename": filename})
            
            # Fetch all pages for this PDF into memory to release the read lock immediately
            cursor.execute("""
                SELECT page_number, text, text_source FROM pages 
                WHERE pdf_id = ? ORDER BY page_number
            """, (pdf_id,))
            pages = cursor.fetchall()
            
            batch_chunks = []
            batch_limit = 128
            skipped_tiny = 0
            skipped_ocr = 0
            
            for page_num, text, text_source in pages:
                if not text.strip():
                    continue
                    
                sub_chunks = splitter.split_text(text)
                for c_idx, chunk_text in enumerate(sub_chunks):
                    if len(chunk_text.strip()) < 50:
                        skipped_tiny += 1
                        continue
                        
                    if text_source == "ocr" and is_low_quality_ocr(chunk_text):
                        skipped_ocr += 1
                        continue
                        
                    chunk_id = f"{filename}_p{page_num}_c{c_idx}"
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "page_number": page_num,
                        "text": chunk_text,
                        "text_source": text_source
                    }
                    batch_chunks.append(chunk_data)
                    
                    # Once the buffer reaches batch limit, generate embeddings and stream to ChromaDB
                    if len(batch_chunks) >= batch_limit:
                        stream_to_chroma_and_sqlite(collection, model, batch_chunks, DB_PATH)
                        batch_chunks = []
                            
            # Process remaining chunks in the buffer
            if batch_chunks:
                stream_to_chroma_and_sqlite(collection, model, batch_chunks, DB_PATH)
                
            print(f"  Completed indexing '{filename}'. Skipped {skipped_tiny} tiny chunks, {skipped_ocr} noisy OCR chunks.")
            
    # 3. Rebuild the BM25 Index from the SQLite chunks table (acts as the source of truth)
    print("Rebuilding BM25 index from all current chunks in database...")
    cursor.execute("SELECT chunk_id, pdf_id, filename, page_number, text, text_source FROM chunks")
    all_db_chunks = cursor.fetchall()
    
    if not all_db_chunks:
        print("Warning: No chunks found in the database. BM25 index will be empty.")
        # Save empty index structure
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": None, "chunks": []}, f)
    else:
        # Reconstruct list of chunk dicts for retriever compatibility
        chunks_list = []
        tokenized_corpus = []
        for cid, pid, fname, pnum, text, tsource in all_db_chunks:
            chunk_dict = {
                "chunk_id": cid,
                "pdf_id": pid,
                "filename": fname,
                "page_number": pnum,
                "text": text,
                "text_source": tsource
            }
            chunks_list.append(chunk_dict)
            tokenized_corpus.append(tokenize_bm25(text))
            
        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "chunks": chunks_list
            }, f)
        print(f"BM25 index saved to {BM25_INDEX_PATH} ({len(chunks_list)} chunks).")
        
    conn.close()
    print("Indexing process completed successfully!")

def stream_to_chroma_and_sqlite(collection, model, chunks, db_path):
    """Embeds a batch of chunks, uploads to ChromaDB, and inserts metadata to SQLite."""
    texts = [c["text"] for c in chunks]
    
    # Compute embeddings in a single batch
    embeddings = model.encode(texts, batch_size=len(chunks), show_progress_bar=False)
    emb_list = [emb.tolist() for emb in embeddings]
    
    # Upload directly to ChromaDB
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=emb_list,
        metadatas=[{
            "pdf_id": c["pdf_id"],
            "filename": c["filename"],
            "page_number": c["page_number"],
            "text_source": c["text_source"]
        } for c in chunks],
        documents=texts
    )
    
    # Insert chunks metadata into SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.conn.cursor() if hasattr(conn, "conn") else conn.cursor()
    for c in chunks:
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, pdf_id, filename, page_number, text, text_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (c["chunk_id"], c["pdf_id"], c["filename"], c["page_number"], c["text"], c["text_source"]))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    build_indices()
