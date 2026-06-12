import os
import sys
import json

# Reconfigure stdout to support UTF-8 (emojis) on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieve import retrieve_and_rerank
from src.rag import generate_answer_stream, compute_overall_confidence

def debug():
    query = "what is python?"
    print(f"QUERY: '{query}'")
    
    # 1. Retrieval
    context_chunks, retrieve_times = retrieve_and_rerank(
        query, 
        retrieve_top_k=10, 
        final_top_k=3, 
        use_reranker=True
    )
    print(f"Retrieval Chunks: {len(context_chunks)}")
    for idx, c in enumerate(context_chunks):
        print(f"  Chunk {idx+1}: {c['metadata']['filename']} Page {c['metadata']['page_number']} (Score: {c['reranker_score']})")
        
    if not context_chunks:
        print("No chunks found!")
        return
        
    # 2. Generation Stream
    print("\nStarting generation stream...")
    stream = generate_answer_stream(query, context_chunks)
    
    full_text = ""
    for event_type, content in stream:
        print(f"[{event_type.upper()}]: {content}")
        if event_type == "text":
            full_text += content
            
    print("\nStream finished.")
    print(f"Accumulated text length: {len(full_text)}")

if __name__ == "__main__":
    debug()
