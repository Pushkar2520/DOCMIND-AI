import os
import pickle
import time
import re
import numpy as np
import torch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# CPU Thread Optimization for PyTorch local inference
torch.set_num_threads(4)

CHROMA_DB_PATH = "data/chroma_db"
BM25_INDEX_PATH = "data/bm25_index.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache models and clients
_embedding_model = None
_reranker_model = None
_chroma_client = None
_chroma_collection = None
_bm25_data = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading local embedding model: sentence-transformers/all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _embedding_model

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        print("Loading local reranker: cross-encoder/ms-marco-MiniLM-L-2-v2...")
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2", device=device)
    return _reranker_model

def get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_client is None or _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            _chroma_collection = _chroma_client.get_collection("pdf_chunks")
        except Exception as e:
            print(f"Error loading Chroma collection: {e}. Check if index.py has run successfully.")
            _chroma_collection = None
    return _chroma_collection

def get_bm25_data():
    global _bm25_data
    if _bm25_data is None:
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                _bm25_data = pickle.load(f)
        else:
            print(f"Error: {BM25_INDEX_PATH} not found. Check if index.py has run successfully.")
            _bm25_data = None
    return _bm25_data

def tokenize_bm25(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

# LRU Cache to store query embeddings (returns tuple to be hashable)
@lru_cache(maxsize=128)
def embed_query(query_text):
    """Embed query using SentenceTransformers, cached in memory."""
    model = get_embedding_model()
    emb = model.encode(query_text, convert_to_numpy=True)
    return tuple(emb.tolist())

# ThreadPool Workers
def run_vector_search(collection, q_emb_list, top_k):
    t0 = time.time()
    try:
        res = collection.query(
            query_embeddings=[q_emb_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        dt_ms = int((time.time() - t0) * 1000)
        return res, dt_ms
    except Exception as e:
        print(f"Vector search failed: {e}")
        return None, int((time.time() - t0) * 1000)

def run_bm25_search(bm25, q_tokens):
    t0 = time.time()
    try:
        scores = bm25.get_scores(q_tokens)
        dt_ms = int((time.time() - t0) * 1000)
        return scores, dt_ms
    except Exception as e:
        print(f"BM25 search failed: {e}")
        return None, int((time.time() - t0) * 1000)

def hybrid_retrieve(query, top_k=10):
    """Retrieve top_k documents by fusing vector and lexical searches in parallel using RRF."""
    t_start = time.time()
    
    collection = get_chroma_collection()
    bm25_data = get_bm25_data()
    
    if collection is None or bm25_data is None:
        return [], {
            "emb_time_ms": 0,
            "vector_time_ms": 0,
            "bm25_time_ms": 0,
            "parallel_search_time_ms": 0,
            "rrf_time_ms": 0
        }
        
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]
    
    # 1. Query Embedding (LRU Cached)
    t_emb_start = time.time()
    q_emb = embed_query(query)
    q_emb_list = list(q_emb)
    emb_time_ms = int((time.time() - t_emb_start) * 1000)
    
    # 2. Parallel Vector & Lexical Search
    q_tokens = tokenize_bm25(query)
    
    t_search_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vector = executor.submit(run_vector_search, collection, q_emb_list, top_k)
        future_bm25 = executor.submit(run_bm25_search, bm25, q_tokens)
        
        v_results, vector_time_ms = future_vector.result()
        bm25_scores, bm25_time_ms = future_bm25.result()
        
    parallel_search_time_ms = int((time.time() - t_search_start) * 1000)
    
    # 3. Reciprocal Rank Fusion (RRF)
    t_rrf_start = time.time()
    vector_hits = {}
    if v_results and v_results["ids"] and len(v_results["ids"][0]) > 0:
        ids = v_results["ids"][0]
        docs = v_results["documents"][0]
        metas = v_results["metadatas"][0]
        dists = v_results["distances"][0]
        
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            sim = 1.0 - dist
            vector_hits[cid] = {
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "vector_score": float(sim),
                "vector_rank": rank + 1
            }
            
    bm25_hits = {}
    if bm25_scores is not None:
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_bm25_indices):
            score = bm25_scores[idx]
            if score <= 0:  # Skip irrelevant hits
                continue
            chunk = chunks[idx]
            cid = chunk["chunk_id"]
            bm25_hits[cid] = {
                "chunk_id": cid,
                "text": chunk["text"],
                "metadata": {
                    "pdf_id": chunk["pdf_id"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "text_source": chunk["text_source"]
                },
                "bm25_score": float(score),
                "bm25_rank": rank + 1
            }
            
    rrf_constant = 60
    rrf_scores = {}
    all_cids = set(vector_hits.keys()) | set(bm25_hits.keys())
    
    for cid in all_cids:
        v_rank = vector_hits[cid]["vector_rank"] if cid in vector_hits else None
        b_rank = bm25_hits[cid]["bm25_rank"] if cid in bm25_hits else None
        
        v_contrib = 1.0 / (rrf_constant + v_rank) if v_rank is not None else 0.0
        b_contrib = 1.0 / (rrf_constant + b_rank) if b_rank is not None else 0.0
        
        rrf_scores[cid] = v_contrib + b_contrib
        
    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    fused_results = []
    for rank, cid in enumerate(sorted_cids):
        text = vector_hits[cid]["text"] if cid in vector_hits else bm25_hits[cid]["text"]
        metadata = vector_hits[cid]["metadata"] if cid in vector_hits else bm25_hits[cid]["metadata"]
        
        fused_results.append({
            "chunk_id": cid,
            "text": text,
            "metadata": metadata,
            "vector_score": vector_hits[cid]["vector_score"] if cid in vector_hits else None,
            "vector_rank": vector_hits[cid]["vector_rank"] if cid in vector_hits else None,
            "bm25_score": bm25_hits[cid]["bm25_score"] if cid in bm25_hits else None,
            "bm25_rank": bm25_hits[cid]["bm25_rank"] if cid in bm25_hits else None,
            "rrf_score": rrf_scores[cid],
            "rrf_rank": rank + 1
        })
        
    rrf_time_ms = int((time.time() - t_rrf_start) * 1000)
    
    profile_times = {
        "emb_time_ms": emb_time_ms,
        "vector_time_ms": vector_time_ms,
        "bm25_time_ms": bm25_time_ms,
        "parallel_search_time_ms": parallel_search_time_ms,
        "rrf_time_ms": rrf_time_ms
    }
    
    return fused_results, profile_times

def retrieve_and_rerank(query, retrieve_top_k=10, final_top_k=3, use_reranker=True):
    """Retrieve top chunks, run Cross-Encoder reranking, and filter by score thresholds & adaptive limits."""
    start_time = time.time()
    
    # 1. Parallel Hybrid Retrieve
    candidates, profile_times = hybrid_retrieve(query, top_k=retrieve_top_k)
    retrieval_time_ms = int((time.time() - start_time) * 1000)
    
    if not candidates:
        profile_times.update({
            "retrieval_time_ms": retrieval_time_ms,
            "rerank_time_ms": 0,
            "total_retrieve_time_ms": retrieval_time_ms
        })
        return [], profile_times
        
    rerank_start = time.time()
    
    # 2. Rerank using local CrossEncoder
    if use_reranker and len(candidates) > 0:
        model = get_reranker_model()
        doc_texts = [c["text"] for c in candidates]
        pairs = [[query, doc] for doc in doc_texts]
        
        reranker_scores = model.predict(pairs).tolist()
        
        if isinstance(reranker_scores, float):
            reranker_scores = [reranker_scores]
            
        for c, score in zip(candidates, reranker_scores):
            c["reranker_score"] = float(score)
            
        # 3. Optimization: Filter candidates with reranker score < -2.0 (relevance thresholding)
        candidates = [c for c in candidates if c["reranker_score"] >= -2.0]
        
        # Sort by reranker score descending
        candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
    else:
        for c in candidates:
            c["reranker_score"] = None
            
    rerank_time_ms = int((time.time() - rerank_start) * 1000)
    
    # 4. Adaptive Context: Dynamically reduce final chunks count if top match is exceptionally high
    final_chunks = candidates[:final_top_k]
    
    if len(final_chunks) > 1 and final_chunks[0]["reranker_score"] is not None:
        top_score = final_chunks[0]["reranker_score"]
        # If the top document matches very strongly (> 4.0), we prune less relevant subsequent chunks
        if top_score > 4.0:
            filtered = [final_chunks[0]]
            for chunk in final_chunks[1:]:
                # Only keep other chunks if they have a positive score and are within 3.5 points of the top
                if chunk["reranker_score"] > 0.0 and (top_score - chunk["reranker_score"] < 3.5):
                    filtered.append(chunk)
            final_chunks = filtered
            
    profile_times.update({
        "retrieval_time_ms": retrieval_time_ms,
        "rerank_time_ms": rerank_time_ms,
        "total_retrieve_time_ms": int((time.time() - start_time) * 1000)
    })
    
    return final_chunks, profile_times

if __name__ == "__main__":
    print("Testing retrieval engine...")
    test_query = "What is docker?"
    results, times = retrieve_and_rerank(test_query)
    print(f"Retrieved {len(results)} chunks in {times['total_retrieve_time_ms']}ms.")
    print("Profile times:", times)
    for idx, r in enumerate(results):
        print(f"\n[{idx+1}] Source: {r['metadata']['filename']} Page {r['metadata']['page_number']}")
        print(f"    RRF Rank: {r['rrf_rank']} | Reranker Score: {r['reranker_score']:.4f}")
