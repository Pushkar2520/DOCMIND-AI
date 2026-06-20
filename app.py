import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import RAG pipeline modules
from src.retrieve import retrieve_and_rerank
from src.rag import generate_answer_stream, compute_overall_confidence

# Streamlit data cache for RAG search (expires in 10 minutes)
@st.cache_data(ttl=600, show_spinner=False)
def cached_retrieve_and_rerank(query, retrieve_top_k, use_reranker):
    return retrieve_and_rerank(query, retrieve_top_k=retrieve_top_k, final_top_k=3, use_reranker=use_reranker)

# Page setup
st.set_page_config(
    page_title="DocuMind - Premium PDF RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    
    /* Premium glassmorphic metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    .badge-high {
        background-color: #2f855a;
        color: #c6f6d5;
    }
    .badge-medium {
        background-color: #c05621;
        color: #feebc8;
    }
    .badge-low {
        background-color: #9b2c2c;
        color: #fed7d7;
    }
    .badge-source-native {
        background-color: #2b6cb0;
        color: #ebf8ff;
    }
    .badge-source-ocr {
        background-color: #553c9a;
        color: #faf5ff;
    }
    
    /* Latency Breakdown */
    .latency-container {
        font-family: monospace;
        font-size: 0.85rem;
        background-color: #1a202c;
        color: #48bb78;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #2d3748;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Database file for logging
DB_PATH = "data/monitoring.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            response TEXT,
            confidence_score INTEGER,
            confidence_level TEXT,
            hallucination_status TEXT,
            total_latency_ms INTEGER,
            retrieval_latency_ms INTEGER,
            rerank_latency_ms INTEGER,
            generation_latency_ms INTEGER,
            validation_latency_ms INTEGER,
            sources TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(query, response, confidence_score, confidence_level, hallucination_status, 
              total_latency_ms, retrieval_latency_ms, rerank_latency_ms, 
              generation_latency_ms, validation_latency_ms, sources):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_logs (
                timestamp, query, response, confidence_score, confidence_level, 
                hallucination_status, total_latency_ms, retrieval_latency_ms, 
                rerank_latency_ms, generation_latency_ms, validation_latency_ms, sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            response,
            confidence_score,
            confidence_level,
            hallucination_status,
            total_latency_ms,
            retrieval_latency_ms,
            rerank_latency_ms,
            generation_latency_ms,
            validation_latency_ms,
            json.dumps(sources)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error logging query to database: {e}")

def get_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM query_logs ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error reading DB logs: {e}")
        return pd.DataFrame()

# Initialize Database
init_db()

# Main Application Layout
st.markdown('<div class="main-header">DocuMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Hybrid RAG System with Fact Validation & Real-time Streaming</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_dashboard = st.tabs(["💬 Chatbot Interface", "📊 Monitoring Dashboard"])

# Sidebar Configurations
with st.sidebar:
    st.image("https://img.icons8.com/gradient/100/processor.png", width=60)
    st.subheader("Configuration Panel")
    
    st.markdown("---")
    st.markdown("### 🔑 API Key Configuration")
    user_api_key = st.text_input(
        "Gemini API Key", 
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="If the default key is rate-limited or exhausted, paste a new Gemini API Key here to continue."
    )
    if user_api_key:
        import google.generativeai as genai
        genai.configure(api_key=user_api_key)
        
    st.markdown("---")
    use_reranker = st.toggle("Enable Cross-Encoder Reranker", value=True, help="Use cross-encoder/ms-marco-MiniLM-L-2-v2 to rerank candidate passages.")
    
    retrieve_top_k = st.slider("Retrieve Top K Candidates", min_value=5, max_value=40, value=10, step=5, 
                               help="Number of chunks retrieved initially from vector and lexical search to pass to the reranker.")
                               
    st.markdown("---")
    st.markdown("### System Check & Verification")
    
    # Check if indices are loaded
    indices_status = "🔴 Missing Indices"
    DATA_DIR = os.getenv("DATA_DIR", "data")
    corpus_db = os.path.join(DATA_DIR, "corpus.db")
    chromadb_dir = os.path.join(DATA_DIR, "chroma_db")
    bm25_file = os.path.join(DATA_DIR, "bm25_index.pkl")
    
    if os.path.exists(corpus_db) and os.path.exists(chromadb_dir) and os.path.exists(bm25_file):
        indices_status = "🟢 Ready"
        
    st.markdown(f"**Index Status**: {indices_status}")
    
    # Simple document list
    if os.path.exists(corpus_db):
        try:
            conn = sqlite3.connect(corpus_db)
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM documents")
            filenames = [row[0] for row in cursor.fetchall()]
            conn.close()
            st.markdown(f"**Ingested PDFs ({len(filenames)})**:")
            for f_name in sorted(filenames):
                st.markdown(f"- `{f_name}`")
        except Exception:
            st.markdown("*Error loading document names.*")
    else:
        st.markdown("*No documents ingested yet.*")
        
    st.markdown("---")
    st.markdown("### Dev Stats & Hardware")
    st.markdown("- **Device**: `CPU` (Local CPU optimized)")
    st.markdown("- **Embedding Model**: `all-MiniLM-L6-v2`")
    st.markdown("- **Reranker Model**: `ms-marco-MiniLM-L-2-v2`")
    st.markdown("- **LLM Provider**: `Google Gemini`")

# --- Tab 1: Chatbot Interface ---
with tab_chat:
    # Check if index exists before allowing chat
    if indices_status != "🟢 Ready":
        st.warning("⚠️ Warning: System indices are not ready. Please verify that the Ingestion and Indexing scripts have finished successfully.")
        
    # Session state for message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources if assistant message and they exist
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                # Show badges
                conf_level = msg.get("confidence_level", "Medium")
                conf_score = msg.get("confidence_score", 50)
                status = msg.get("hallucination_status", "FULLY_SUPPORTED")
                
                badge_class = "badge-high" if conf_level == "High" else ("badge-medium" if conf_level == "Medium" else "badge-low")
                status_badge = "badge-high" if status == "FULLY_SUPPORTED" else ("badge-medium" if status == "PARTIALLY_SUPPORTED" else "badge-low")
                
                st.markdown(f"""
                <div style="margin-top: 5px; margin-bottom: 10px;">
                    <span class="badge {badge_class}">Confidence: {conf_level} ({conf_score}%)</span>
                    <span class="badge {status_badge}">Fact Check: {status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📚 Show Cited Sources"):
                    for s_idx, src in enumerate(msg["sources"]):
                        source_type = "badge-source-native" if src["metadata"].get("text_source") == "native" else "badge-source-ocr"
                        st.markdown(f"""
                        **Source [{s_idx+1}]**: `{src['metadata']['filename']}` - Page {src['metadata']['page_number']} 
                        <span class="badge {source_type}">{src['metadata'].get('text_source', 'unknown').upper()}</span>
                        <div style="font-size:0.9rem; background-color:rgba(255,255,255,0.03); padding:8px; border-left:3px solid #667eea; margin-top:5px; margin-bottom:10px;">
                            {src['text']}
                        </div>
                        <div style="font-size:0.75rem; color:#718096; margin-bottom:15px;">
                            Vector Score: {f"{src['vector_score']:.4f}" if src['vector_score'] is not None else 'N/A'} (Rank {src['vector_rank'] or 'N/A'}) | 
                            BM25 Score: {f"{src['bm25_score']:.2f}" if src['bm25_score'] is not None else 'N/A'} (Rank {src['bm25_rank'] or 'N/A'}) | 
                            RRF Rank: {src['rrf_rank']} | 
                            Reranker Score: {f"{src['reranker_score']:.4f}" if src['reranker_score'] is not None else 'N/A'}
                        </div>
                        """, unsafe_allow_html=True)
                        
                if "latency_breakdown" in msg and msg["latency_breakdown"]:
                    lat = msg["latency_breakdown"]
                    st.markdown(f"""
                    <div class="latency-container">
                        Latency: Retrieval: {lat.get('retrieval_time_ms', 0)}ms | Rerank: {lat.get('rerank_time_ms', 0)}ms | Gen: {lat.get('generation_time_ms', 0)}ms | Verify: {lat.get('validation_time_ms', 0)}ms | Total: {lat.get('total_time_ms', 0)}ms
                    </div>
                    """, unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask a question about the documents...", disabled=(indices_status != "🟢 Ready")):
        # 1. Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
            
        start_total = time.time()
        
        # 2. Retrieve & Rerank (top 20 candidates fused, reranked to top 3)
        retrieval_error = False
        with st.spinner("Searching and ranking documents..."):
            try:
                context_chunks, retrieve_times = cached_retrieve_and_rerank(
                    question, 
                    retrieve_top_k=retrieve_top_k, 
                    use_reranker=use_reranker
                )
            except Exception as retrieve_err:
                st.error(f"Error during retrieval: {retrieve_err}")
                retrieval_error = True
                context_chunks = []
                retrieve_times = {
                    "emb_time_ms": 0, "vector_time_ms": 0, "bm25_time_ms": 0,
                    "parallel_search_time_ms": 0, "rrf_time_ms": 0, "retrieval_time_ms": 0,
                    "rerank_time_ms": 0, "total_retrieve_time_ms": 0
                }
                
        # 3. Stream LLM Answer Generation and Verification
        full_response = ""
        generation_start = time.time()
        
        validation_status = "FULLY_SUPPORTED"
        validation_confidence = 100
        validation_explanation = "Skipped"
        generation_time_ms = 0
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            if retrieval_error or not context_chunks:
                full_response = "Information not found in documents."
                response_placeholder.markdown(full_response)
                validation_status = "UNSUPPORTED"
                validation_confidence = 0
            else:
                try:
                    # Stream answer to placeholder (which yields text events, then data)
                    stream = generate_answer_stream(question, context_chunks)
                    for event_type, content in stream:
                        if event_type == "text":
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                        elif event_type == "data":
                            clean_answer, val_data = content
                            response_placeholder.markdown(clean_answer)
                            full_response = clean_answer
                            
                            validation_status = val_data.get("status", "FULLY_SUPPORTED")
                            validation_confidence = val_data.get("confidence", 100)
                            validation_explanation = val_data.get("explanation", "")
                except Exception as gen_err:
                    full_response = f"Error generating answer: {gen_err}"
                    response_placeholder.markdown(full_response)
                    validation_status = "ERROR"
                    validation_confidence = 50
                
                generation_time_ms = int((time.time() - generation_start) * 1000)
                
                # If answer is unsupported (and it's not a rate limit / system API error), override it
                if validation_status == "UNSUPPORTED" or (validation_status == "PARTIALLY_SUPPORTED" and validation_confidence < 30):
                    full_response = "Information not found in documents."
                    response_placeholder.markdown(full_response)
                    
        total_latency_ms = int((time.time() - start_total) * 1000)
        validation_time_ms = 0 # Included in single-pass generation
        
        # 4. Compute overall confidence score
        reranker_scores = [c["reranker_score"] for c in context_chunks if c["reranker_score"] is not None]
        overall_score, confidence_level = compute_overall_confidence(
            reranker_scores, 
            validation_status, 
            validation_confidence
        )
        
        if "Information not found" in full_response:
            confidence_level = "Low"
            overall_score = 0
            validation_status = "UNSUPPORTED"
            
        # 5. Render Badges, Sources, and Latency Info
        badge_class = "badge-high" if confidence_level == "High" else ("badge-medium" if confidence_level == "Medium" else "badge-low")
        status_badge = "badge-high" if validation_status == "FULLY_SUPPORTED" else ("badge-medium" if validation_status == "PARTIALLY_SUPPORTED" else "badge-low")
        
        st.markdown(f"""
        <div style="margin-top: 5px; margin-bottom: 10px;">
            <span class="badge {badge_class}">Confidence: {confidence_level} ({overall_score}%)</span>
            <span class="badge {status_badge}">Fact Check: {validation_status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Citations expander
        if context_chunks and "Information not found" not in full_response:
            with st.expander("📚 Show Cited Sources"):
                for s_idx, src in enumerate(context_chunks):
                    source_type = "badge-source-native" if src["metadata"].get("text_source") == "native" else "badge-source-ocr"
                    st.markdown(f"""
                    **Source [{s_idx+1}]**: `{src['metadata']['filename']}` - Page {src['metadata']['page_number']} 
                    <span class="badge {source_type}">{src['metadata'].get('text_source', 'unknown').upper()}</span>
                    <div style="font-size:0.9rem; background-color:rgba(255,255,255,0.03); padding:8px; border-left:3px solid #667eea; margin-top:5px; margin-bottom:10px;">
                        {src['text']}
                    </div>
                    <div style="font-size:0.75rem; color:#718096; margin-bottom:15px;">
                        Vector Score: {f"{src['vector_score']:.4f}" if src['vector_score'] is not None else 'N/A'} (Rank {src['vector_rank'] or 'N/A'}) | 
                        BM25 Score: {f"{src['bm25_score']:.2f}" if src['bm25_score'] is not None else 'N/A'} (Rank {src['bm25_rank'] or 'N/A'}) | 
                        RRF Rank: {src['rrf_rank']} | 
                        Reranker Score: {f"{src['reranker_score']:.4f}" if src['reranker_score'] is not None else 'N/A'}
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Latency breakdown
        latency_breakdown = {
            "emb_time_ms": retrieve_times.get("emb_time_ms", 0),
            "vector_time_ms": retrieve_times.get("vector_time_ms", 0),
            "bm25_time_ms": retrieve_times.get("bm25_time_ms", 0),
            "parallel_search_time_ms": retrieve_times.get("parallel_search_time_ms", 0),
            "rrf_time_ms": retrieve_times.get("rrf_time_ms", 0),
            "retrieval_time_ms": retrieve_times.get("retrieval_time_ms", 0),
            "rerank_time_ms": retrieve_times.get("rerank_time_ms", 0),
            "generation_time_ms": generation_time_ms,
            "validation_time_ms": validation_time_ms,
            "total_time_ms": total_latency_ms
        }
        
        st.markdown(f"""
        <div class="latency-container">
            <b>⏱️ Latency Breakdown:</b><br>
            • Query Embedding: {latency_breakdown['emb_time_ms']}ms<br>
            • ChromaDB Vector Search: {latency_breakdown['vector_time_ms']}ms<br>
            • BM25 Lexical Search: {latency_breakdown['bm25_time_ms']}ms<br>
            • Parallel Search overhead: {latency_breakdown['parallel_search_time_ms']}ms<br>
            • RRF Fusion: {latency_breakdown['rrf_time_ms']}ms<br>
            • Cross-Encoder Reranking: {latency_breakdown['rerank_time_ms']}ms<br>
            • Gemini Gen & Verify: {latency_breakdown['generation_time_ms']}ms<br>
            • <b>Total End-to-End Latency: {latency_breakdown['total_time_ms']}ms</b>
        </div>
        """, unsafe_allow_html=True)
        
        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": context_chunks if "Information not found" not in full_response else [],
            "confidence_score": overall_score,
            "confidence_level": confidence_level,
            "hallucination_status": validation_status,
            "latency_breakdown": latency_breakdown
        })
        
        # Log to Database
        log_query(
            query=question,
            response=full_response,
            confidence_score=overall_score,
            confidence_level=confidence_level,
            hallucination_status=validation_status,
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieve_times["retrieval_time_ms"],
            rerank_latency_ms=retrieve_times["rerank_time_ms"],
            generation_latency_ms=generation_time_ms,
            validation_latency_ms=validation_time_ms,
            sources=context_chunks if "Information not found" not in full_response else []
        )


# --- Tab 2: Monitoring Dashboard ---
with tab_dashboard:
    st.subheader("System Performance & Evaluation Dashboard")
    
    logs_df = get_logs()
    
    if logs_df.empty:
        st.info("No queries logged yet. Ask questions in the chatbot tab to generate analytics.")
    else:
        # Compute metrics
        total_queries = len(logs_df)
        
        # Latencies
        avg_total_lat = logs_df["total_latency_ms"].mean()
        p95_total_lat = np.percentile(logs_df["total_latency_ms"].values, 95)
        
        # Confidence
        avg_conf_score = logs_df["confidence_score"].mean()
        
        # Hallucination status
        fully_supported_count = sum(logs_df["hallucination_status"] == "FULLY_SUPPORTED")
        hallucination_rate = ((total_queries - fully_supported_count) / total_queries) * 100
        
        # Draw metric cards in grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Queries</div>
                <div class="metric-value">{total_queries}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Latency (Total)</div>
                <div class="metric-value">{avg_total_lat/1000:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">P95 Latency</div>
                <div class="metric-value">{p95_total_lat/1000:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Hallucination Rate</div>
                <div class="metric-value">{hallucination_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Draw Charts
        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("### ⏱️ Latency Distribution Breakdown (ms)")
            # Line chart of latencies over time
            latency_data = logs_df[["timestamp", "retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms", "validation_latency_ms"]].copy()
            latency_data = latency_data.set_index("timestamp").sort_index()
            st.area_chart(latency_data)
            
        with col_c2:
            st.markdown("### 🛡️ Fact Verification Status Summary")
            # Bar chart of hallucination status counts
            status_counts = logs_df["hallucination_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            st.bar_chart(data=status_counts, x="Status", y="Count", color="#667eea")
            
        # Detailed logs table
        st.markdown("### 📋 Detailed Query History Log")
        
        display_df = logs_df[[
            "timestamp", "query", "response", "confidence_level", 
            "confidence_score", "hallucination_status", "total_latency_ms"
        ]].copy()
        
        display_df.columns = [
            "Timestamp", "User Query", "Chatbot Answer", "Confidence Level", 
            "Confidence Score (%)", "Fact Check Status", "Latency (ms)"
        ]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export option
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Log History as CSV",
            data=csv,
            file_name="documind_query_history.csv",
            mime="text/csv",
        )
