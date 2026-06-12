import os
import fitz

DOCS_DIR = "Docs"
pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]

print(f"Analyzing {len(pdf_files)} PDF files...")
for idx, filename in enumerate(pdf_files):
    pdf_path = os.path.join(DOCS_DIR, filename)
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        
        # Check first 5 pages for text content
        has_text_count = 0
        for p_idx in range(min(5, num_pages)):
            text = doc[p_idx].get_text().strip()
            if len(text) > 100:
                has_text_count += 1
                
        is_scanned = has_text_count == 0
        print(f"{filename}: {num_pages} pages | Text in first few pages: {has_text_count}/5 | Scanned (no text): {is_scanned}")
        doc.close()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
