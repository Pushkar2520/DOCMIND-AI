import os
import json
import time
import math
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment. Please set it in .env file.")

# Choose model (gemini-2.5-flash is fast, reliable, and has free tier)
DEFAULT_MODEL = "gemini-2.5-flash"

def get_llm_model(model_name=DEFAULT_MODEL):
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

def generate_answer_stream(query, context_chunks, model_name=DEFAULT_MODEL):
    """
    Generate the answer and run fact validation in a SINGLE LLM call.
    Yields:
        ("text", chunk_text) during streaming of the answer.
        ("data", (clean_answer, validation_json)) at the very end.
    """
    model = get_llm_model(model_name)
    if model is None:
        yield "text", "Error: LLM model could not be initialized."
        yield "data", ("Error: LLM model could not be initialized.", {"status": "ERROR", "confidence": 0, "explanation": "Model init failed"})
        return

    # Build context string
    context_str = ""
    for idx, chunk in enumerate(context_chunks):
        context_str += f"Document Excerpt [{chunk['metadata']['filename']} Page {chunk['metadata']['page_number']}]:\n{chunk['text']}\n\n"

    # Build prompt that forces answer generation followed by JSON validation in a single call
    prompt = f"""You are an expert Retrieval-Augmented Generation (RAG) assistant. 
Your task is to answer the user query based ONLY on the provided document excerpts.

CONTEXT DOCUMENTS:
{context_str}

USER QUERY:
{query}

INSTRUCTIONS:
1. Answer the query accurately and concisely using the provided context.
2. Do NOT include any inline citations (e.g. no [file.pdf page X] inside sentences). 
3. At the end of your answer, list all unique source documents used under a "Sources:" heading. Format it exactly like this:
   Sources:
   📄 filename (Page X)
4. Do NOT use external knowledge. If the provided context does not contain sufficient details to answer, state: "Information not found in documents."
5. You MUST append a fact-validation check at the very end of your response. First write `[VALIDATION]` on a new line, followed by a JSON object containing:
   - "status": "FULLY_SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED"
   - "confidence": 0-100 (rating the factual alignment of your answer with the source documents)
   - "explanation": "one-sentence explanation of why it is supported/unsupported"
   
   If you output "Information not found in documents.", classify it as "UNSUPPORTED" with 0 confidence.

FORMAT EXAMPLE:
[ANSWER]
Python is...
Sources:
📄 python_programming_guide.pdf (Page 21)
[VALIDATION]
{{
  "status": "FULLY_SUPPORTED",
  "confidence": 100,
  "explanation": "The explanation of python is fully supported by page 21."
}}

Begin your response now:"""

    max_retries = 3
    delay = 2.0
    response_stream = None
    
    for attempt in range(max_retries):
        try:
            response_stream = model.generate_content(prompt, stream=True)
            break
        except ResourceExhausted as e:
            if attempt == max_retries - 1:
                yield "text", "[API Rate Limit Exceeded: Please wait a moment or enter a new Gemini API Key in the sidebar.]"
                yield "data", ("[API Rate Limit Exceeded: Please wait a moment or enter a new Gemini API Key in the sidebar.]", {"status": "API_ERROR", "confidence": 0, "explanation": "Rate limit exceeded"})
                return
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            yield "text", f"Error calling LLM: {e}"
            yield "data", (f"Error calling LLM: {e}", {"status": "ERROR", "confidence": 0, "explanation": str(e)})
            return

    if response_stream is None:
        return

    # Streaming parser to separate answer from validation block
    answer_buffer = ""
    validation_buffer = ""
    in_validation = False
    
    try:
        for chunk in response_stream:
            if not chunk.text:
                continue
                
            text = chunk.text
            
            if in_validation:
                validation_buffer += text
                continue
                
            combined = answer_buffer + text
            if "[VALIDATION]" in combined:
                parts = combined.split("[VALIDATION]")
                answer_part = parts[0]
                val_part = "[VALIDATION]".join(parts[1:])
                
                # Yield only the new tokens belonging to the answer part
                new_tokens = answer_part[len(answer_buffer):]
                if new_tokens:
                    # Clean out the [ANSWER] tag if it gets streamed
                    yield "text", new_tokens.replace("[ANSWER]", "")
                    
                answer_buffer = answer_part
                validation_buffer = val_part
                in_validation = True
            else:
                yield "text", text.replace("[ANSWER]", "")
                answer_buffer += text
                
    except Exception as e:
        yield "text", f"\n[Stream interrupted: {e}]"
        
    # Clean up answer text
    clean_answer = answer_buffer.replace("[ANSWER]", "").strip()
    
    # Parse validation JSON
    validation_data = {"status": "FULLY_SUPPORTED", "confidence": 100, "explanation": "Success"}
    if validation_buffer:
        try:
            clean_json = validation_buffer.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()
            validation_data = json.loads(clean_json)
        except Exception as e:
            print(f"Error parsing validation JSON: {e}, text was: {validation_buffer}")
            validation_data = {
                "status": "ERROR",
                "confidence": 50,
                "explanation": f"Failed to parse validation JSON: {e}"
            }
            
    # Yield the final clean answer and validation metadata
    yield "data", (clean_answer, validation_data)

def compute_overall_confidence(reranker_scores, validation_status, validation_confidence):
    """Compute overall confidence score (High/Medium/Low) based on retrieval and validation."""
    if validation_status == "UNSUPPORTED":
        return 0, "Low"
        
    retrieval_conf = 50.0 # Default fallback
    
    if reranker_scores and len(reranker_scores) > 0:
        max_score = max(reranker_scores)
        # Map logit from cross-encoder to 0-100 range
        # If score is from ms-marco-MiniLM-L-2-v2, it ranges roughly between -10 and +10
        retrieval_conf = 100.0 / (1.0 + math.exp(-max_score))
        
    overall = (0.3 * retrieval_conf) + (0.7 * validation_confidence)
    overall = round(max(0, min(100, overall)))
    
    if validation_status == "FULLY_SUPPORTED" and overall >= 75:
        level = "High"
    elif overall >= 40:
        level = "Medium"
    else:
        level = "Low"
        
    return overall, level

if __name__ == "__main__":
    # Test RAG module
    print("Testing RAG pipeline...")
    test_chunks = [{
        "chunk_id": "test_c1",
        "text": "Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. Containers are isolated from one another and bundle their own software, libraries and configuration files.",
        "metadata": {"filename": "docker_tutorial.pdf", "page_number": 1}
    }]
    print("Testing stream generation:")
    answer = ""
    val = None
    for event_type, content in generate_answer_stream("What is Docker?", test_chunks):
        if event_type == "text":
            answer += content
            print(content, end="", flush=True)
        elif event_type == "data":
            clean, val = content
            print(f"\n\nClean Answer:\n{clean}")
            print(f"\nValidation Metadata:\n{json.dumps(val, indent=2)}")
