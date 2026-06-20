import os
import re
import time
import sqlite3
import hashlib
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from langdetect import detect

load_dotenv()

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load folder configurations from env variables
DOCS_DIR = os.getenv("DOCS_DIR", "Docs")
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "corpus.db")

def compute_file_hash(file_path):
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_text(text):
    """Clean and normalize text, removing headers, footers, and excess spacing."""
    if not text:
        return ""
    
    # Replace common ligatures
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("“", '"').replace("”", '"')
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Heuristics for page numbers / simple headers
        if re.match(r'^(page|pg\.?)\s*\d+\s*(of\s*\d+)?$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+\s*\|\s*page.*$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^page\s*\d+\s*\|\s*.*$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', stripped):
            continue
            
        cleaned_lines.append(line)
        
    text = "\n".join(cleaned_lines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def detect_language(text):
    """Detect language of a text snippet using langdetect."""
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"

def ocr_page_worker(pdf_path, page_num):
    """Worker function for ProcessPoolExecutor to perform OCR on a specific page."""
    # Initialize pytesseract command in worker just in case
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # 0-indexed
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img)
        doc.close()
        return page_num, text
    except Exception as e:
        print(f"Error doing OCR on {pdf_path} Page {page_num}: {e}")
        return page_num, ""

def init_sqlite_db():
    """Initialize the SQLite database schema."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            pdf_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            file_path TEXT,
            file_hash TEXT,
            page_count INTEGER,
            ingested_at TEXT
        );
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            page_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id INTEGER,
            filename TEXT,
            page_number INTEGER,
            text TEXT,
            text_source TEXT,
            language TEXT,
            FOREIGN KEY(pdf_id) REFERENCES documents(pdf_id) ON DELETE CASCADE
        );
    """)
    
    # We also pre-initialize the chunks table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            pdf_id INTEGER,
            filename TEXT,
            page_number INTEGER,
            text TEXT,
            text_source TEXT,
            FOREIGN KEY(pdf_id) REFERENCES documents(pdf_id) ON DELETE CASCADE
        );
    """)
    
    conn.commit()
    conn.close()

def clean_deleted_documents(current_files):
    """Remove documents and pages from database if the files were deleted from the disk."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT pdf_id, filename FROM documents")
    db_docs = cursor.fetchall()
    
    for pdf_id, filename in db_docs:
        if filename not in current_files:
            print(f"File {filename} was deleted from disk. Purging from database...")
            # ChromaDB purge will happen in index.py during vector reconciliation
            cursor.execute("DELETE FROM documents WHERE pdf_id = ?", (pdf_id,))
            
    conn.commit()
    conn.close()

def ingest_pdfs():
    init_sqlite_db()
    
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Documents directory '{DOCS_DIR}' not found.")
        return
        
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in '{DOCS_DIR}'.")
    
    # Clean up deleted documents
    clean_deleted_documents(set(pdf_files))
    
    # Load existing docs details for hashing check
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, file_hash, pdf_id FROM documents")
    existing_docs = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    conn.close()
    
    for idx, filename in enumerate(pdf_files):
        pdf_path = os.path.join(DOCS_DIR, filename)
        file_hash = compute_file_hash(pdf_path)
        
        # Incremental check
        if filename in existing_docs:
            db_hash, pdf_id = existing_docs[filename]
            if db_hash == file_hash:
                print(f"[{idx+1}/{len(pdf_files)}] Skipping (already ingested & unchanged): {filename}")
                continue
            else:
                print(f"[{idx+1}/{len(pdf_files)}] File modified: {filename}. Re-ingesting...")
                # Purge old version from DB (pages/chunks cascade automatically)
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE pdf_id = ?", (pdf_id,))
                conn.commit()
                conn.close()
        else:
            print(f"[{idx+1}/{len(pdf_files)}] New file detected: {filename}. Ingesting...")
            
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            
            # Step 1: Add document metadata
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, file_path, file_hash, page_count, ingested_at)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, pdf_path, file_hash, num_pages, time.strftime("%Y-%m-%d %H:%M:%S")))
            new_pdf_id = cursor.lastrowid
            conn.commit()
            
            ocr_tasks = []
            extracted_pages = {}
            
            # Step 2: Separate native text extraction from OCR requirement
            for page_num in range(1, num_pages + 1):
                page = doc[page_num - 1]
                native_text = page.get_text()
                
                # Retrieve structural blocks/words
                words = page.get_text("words")
                blocks = page.get_text("blocks")
                
                # Refined OCR fallback condition:
                # If there are no native text blocks/words at all, check if images exist on the page.
                if len(words) == 0 and len(page.get_images()) > 0:
                    ocr_tasks.append(page_num)
                else:
                    cleaned = clean_text(native_text)
                    extracted_pages[page_num] = (cleaned, "native")
            
            doc.close()
            
            # Step 3: Run OCR in parallel using ProcessPoolExecutor
            if ocr_tasks:
                print(f"  Scheduling {len(ocr_tasks)} pages for parallel Tesseract OCR...")
                # Avoid allocating more workers than physical cores (capped at 4 here for resource limits)
                workers = min(os.cpu_count() or 1, 4)
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(ocr_page_worker, pdf_path, p_num) for p_num in ocr_tasks]
                    for fut in futures:
                        p_num, ocr_text = fut.result()
                        cleaned = clean_text(ocr_text)
                        extracted_pages[p_num] = (cleaned, "ocr")
            
            # Step 4: Detect languages and insert pages in a single transaction
            for p_num in sorted(extracted_pages.keys()):
                text, source = extracted_pages[p_num]
                lang = detect_language(text)
                cursor.execute("""
                    INSERT INTO pages (pdf_id, filename, page_number, text, text_source, language)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (new_pdf_id, filename, p_num, text, source, lang))
                
            conn.commit()
            conn.close()
            print(f"  Successfully processed {filename} in {time.time() - start_time:.2f} seconds.")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            
    print("Ingestion pipeline finished.")

if __name__ == "__main__":
    ingest_pdfs()
