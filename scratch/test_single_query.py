import os
import sys
import time
import json
from dotenv import load_dotenv

# Reconfigure stdout to support UTF-8 (emojis) on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from src.retrieve import retrieve_and_rerank
from src.rag import generate_answer_stream, compute_overall_confidence

def run_test():
    print("Sleeping 65 seconds to guarantee the 5 RPM rate limit window has completely reset...")
    time.sleep(65)
    
    query = "Explain FastAPI dependency injection."
    print(f"\nQUERY: '{query}'")
    print("-" * 80)
    
    start_time = time.time()
    
    # 1. Retrieval
    print("[1/3] Retrieving and reranking...")
    context_chunks, retrieve_times = retrieve_and_rerank(
        query, 
        retrieve_top_k=10, 
        final_top_k=3, 
        use_reranker=True
    )
    print(f"      Search Time: {retrieve_times['total_retrieve_time_ms']} ms")
    
    # 2. Generation & Verification
    print("[2/3] Generating answer (streaming)...")
    gen_start = time.time()
    answer = ""
    validation_status = "FULLY_SUPPORTED"
    validation_confidence = 100
    validation_explanation = ""
    
    stream = generate_answer_stream(query, context_chunks)
    for event_type, content in stream:
        if event_type == "text":
            answer += content
            print(content, end="", flush=True)
        elif event_type == "data":
            clean_answer, val_data = content
            answer = clean_answer
            validation_status = val_data.get("status", "FULLY_SUPPORTED")
            validation_confidence = val_data.get("confidence", 100)
            validation_explanation = val_data.get("explanation", "")
            
    print()
    gen_verify_time_ms = int((time.time() - gen_start) * 1000)
    print(f"      Gen & Verify Time: {gen_verify_time_ms} ms")
    
    # 3. Overall confidence
    reranker_scores = [c["reranker_score"] for c in context_chunks if c["reranker_score"] is not None]
    overall_score, confidence_level = compute_overall_confidence(
        reranker_scores, 
        validation_status, 
        validation_confidence
    )
    
    total_time_ms = int((time.time() - start_time) * 1000)
    print("-" * 80)
    print(f"Total Latency: {total_time_ms} ms ({total_time_ms/1000:.2f} seconds)")
    print(f"Confidence:    {confidence_level} ({overall_score}%)")
    print(f"Fact Check:    {validation_status}")
    print(f"Explanation:   {validation_explanation}")
    print("-" * 80)

if __name__ == "__main__":
    run_test()
