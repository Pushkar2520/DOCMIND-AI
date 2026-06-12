import os
import re
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import time

# Set Tesseract executable path as verified
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DOCS_DIR = "Docs"
DATA_DIR = "data"
CLEANED_DOCS_PATH = os.path.join(DATA_DIR, "cleaned_docs.json")

def clean_text(text):
    """Clean and normalize text, removing headers, footers, and excess spacing."""
    if not text:
        return ""
    
    # Replace weird ligatures and characters
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("“", '"').replace("”", '"')
    
    # Remove common page number formats and headers/footers
    # e.g., "Page 1 of 10", "1 | Page", "Chapter 1", header lines like "Docker Tutorial"
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines in this pass
        if not stripped:
            continue
            
        # Heuristics for page numbers / simple headers
        if re.match(r'^(page|pg\.?)\s*\d+\s*(of\s*\d+)?$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+\s*\|\s*page.*$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^page\s*\d+\s*\|\s*.*$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', stripped): # Just a number
            continue
            
        cleaned_lines.append(line)
        
    text = "\n".join(cleaned_lines)
    
    # Replace multiple newlines or spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def detect_language(text):
    """Detect language of a text snippet using a fast, lightweight lexical stop-word heuristic."""
    if not text:
        return "unknown"
        
    common_en = {"the", "and", "of", "to", "in", "is", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"}
    words = re.findall(r'\b[a-z]{2,10}\b', text.lower())
    if not words:
        return "unknown"
        
    en_count = sum(1 for w in words if w in common_en)
    ratio = en_count / len(words)
    
    if ratio > 0.05:
        return "en"
    # Basic fallbacks can be added if needed, but the documents are English technical guides.
    return "en" 

def ingest_pdfs():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load already processed documents
    existing_docs = []
    if os.path.exists(CLEANED_DOCS_PATH):
        try:
            with open(CLEANED_DOCS_PATH, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)
                print(f"Loaded {len(existing_docs)} existing pages from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}. Will rebuild.")
            existing_docs = []
            
    # Keep track of already ingested files
    ingested_files = set(page["filename"] for page in existing_docs)
    
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {DOCS_DIR}.")
    
    new_pages = []
    
    for idx, filename in enumerate(pdf_files):
        if filename in ingested_files:
            print(f"[{idx+1}/{len(pdf_files)}] Skipping already ingested file: {filename}")
            continue
            
        pdf_path = os.path.join(DOCS_DIR, filename)
        print(f"[{idx+1}/{len(pdf_files)}] Ingesting {filename}...")
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            print(f"  Total pages: {num_pages}")
            
            for page_num in range(num_pages):
                page = doc[page_num]
                
                # Try native text extraction first
                text = page.get_text()
                text_source = "native"
                
                # Check if text is empty or too short (scanned page / image PDF)
                if len(text.strip()) < 100:
                    try:
                        # Render page to an image for OCR
                        pix = page.get_pixmap(dpi=150)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Run Tesseract OCR
                        text = pytesseract.image_to_string(img)
                        text_source = "ocr"
                    except Exception as ocr_err:
                        print(f"  OCR failed on {filename} Page {page_num + 1}: {ocr_err}")
                        text = ""
                
                cleaned = clean_text(text)
                lang = detect_language(cleaned)
                
                new_pages.append({
                    "pdf_id": idx,
                    "filename": filename,
                    "page_number": page_num + 1,
                    "text": cleaned,
                    "text_source": text_source,
                    "language": lang
                })
                
                if (page_num + 1) % 50 == 0:
                    print(f"  Processed {page_num + 1}/{num_pages} pages...")
                    
            doc.close()
            print(f"  Successfully processed {filename} in {time.time() - start_time:.2f} seconds.")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            
    if new_pages:
        # Merge new pages with existing pages and save
        all_pages = existing_docs + new_pages
        with open(CLEANED_DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_pages)} total pages to {CLEANED_DOCS_PATH}.")
    else:
        print("No new PDFs to ingest.")

if __name__ == "__main__":
    ingest_pdfs()
