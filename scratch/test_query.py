import os
import sys

# Reconfigure stdout to support UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieve import retrieve_and_rerank
from src.rag import generate_answer_stream

def run():
    query = "what is python?"
    print(f"Query: {query}")
    chunks, times = retrieve_and_rerank(query)
    print(f"Retrieved {len(chunks)} chunks.")
    
    stream = generate_answer_stream(query, chunks)
    for event_type, content in stream:
        print(f"[{event_type.upper()}]: {repr(content)}")

if __name__ == '__main__':
    run()
