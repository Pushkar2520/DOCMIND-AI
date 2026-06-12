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

# Import pipeline components
from src.retrieve import retrieve_and_rerank
from src.rag import generate_answer_stream, compute_overall_confidence

def test_pipeline(query):
    print("=" * 80)
    print(f"TEST QUERY: '{query}'")
    print("=" * 80)
    
    # 1. Retrieval & Reranking
    print("\n[Step 1] Retrieving and reranking candidate chunks...")
    start_time = time.time()
    
    try:
        # Using the faster, optimized default of top 10 candidates
        context_chunks, retrieve_times = retrieve_and_rerank(
            query, 
            retrieve_top_k=10, 
            final_top_k=3, 
            use_reranker=True
        )
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return
        
    print(f"  Retrieval time: {retrieve_times['retrieval_time_ms']} ms")
    print(f"  Reranking time: {retrieve_times['rerank_time_ms']} ms")
    print(f"  Total search time: {retrieve_times['total_retrieve_time_ms']} ms")
    print(f"  Chunks retrieved: {len(context_chunks)}")
    
    if not context_chunks:
        print("No chunks retrieved. Aborting RAG pipeline test.")
        return
        
    # Print source metadata
    for idx, chunk in enumerate(context_chunks):
        print(f"  Chunk {idx+1}: {chunk['metadata']['filename']} Page {chunk['metadata']['page_number']} | "
              f"RRF Rank: {chunk['rrf_rank']} | Reranker Score: {chunk['reranker_score']:.4f}")
              
    # 2. Answer Generation & Fact Validation (Combined Pass)
    print("\n[Step 2] Generating answer and verifying facts (streaming)...")
    gen_start = time.time()
    answer = ""
    validation_status = "FULLY_SUPPORTED"
    validation_confidence = 100
    validation_explanation = "Skipped"
    
    try:
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
    except Exception as e:
        print(f"\nError during generation/verification: {e}")
        return
        
    generation_time_ms = int((time.time() - gen_start) * 1000)
    print(f"\n  Generation & Verification time: {generation_time_ms} ms")
    print(f"  Verification status: {validation_status}")
    print(f"  Verification confidence: {validation_confidence}/100")
    print(f"  Verification explanation: {validation_explanation}")
    
    # 3. Overall Confidence & Latency Summary
    reranker_scores = [c["reranker_score"] for c in context_chunks if c["reranker_score"] is not None]
    overall_score, confidence_level = compute_overall_confidence(
        reranker_scores, 
        validation_status, 
        validation_confidence
    )
    
    # Check if unsupported
    if validation_status == "UNSUPPORTED" or validation_confidence < 30:
        answer = "Information not found in documents."
        overall_score = 0
        confidence_level = "Low"
        validation_status = "UNSUPPORTED"
        
    total_latency_ms = int((time.time() - start_time) * 1000)
    
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Final Answer:     {answer[:200]}...")
    print(f"Total Latency:    {total_latency_ms} ms ({total_latency_ms/1000:.2f} seconds)")
    print(f"Confidence Level: {confidence_level} ({overall_score}%)")
    print(f"Fact Check:       {validation_status}")
    print(f"Target Latency (2-5s): {'PASSED' if 2000 <= total_latency_ms <= 5000 else 'OUTSIDE TARGET'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    queries = [
        "What is Docker and how does OS-level virtualization work?",
        "What is a bash shell script and how do you write a simple hello world script?",
        "Explain FastAPI dependency injection.",
        "What is the capital of France?" # Test out-of-scope query
    ]
    
    for q in queries:
        test_pipeline(q)
        print("Sleeping 15 seconds to respect Gemini API Free Tier rate limits...")
        time.sleep(15)
