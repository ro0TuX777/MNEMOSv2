import os
import glob
import PyPDF2

SIGINT_DIR = r"C:\Users\vin\Downloads\SIGINT"
OUTPUT_DIR = r"G:\MNEMOS\benchmarks\echoframe_eval\phase10\datasets\sigint_corpus"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select a few representative PDFs
TARGET_PDFS = [
    "NSA-SIGINT-style-manual_2010.pdf",
    "Counter-signals intel techniques and procedures.pdf",
    "Redacted Annex DODM 5240.01-A(1).pdf"
]

def extract_chunks(pdf_path, chunk_size_pages=5):
    chunks = []
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        # Just grab the first few meaningful chunks to save time
        start_page = min(5, num_pages) # skip title pages
        end_page = min(start_page + chunk_size_pages, num_pages)
        
        text = ""
        for i in range(start_page, end_page):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if len(text.split()) > 100:
            chunks.append(text)
            
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        
    return chunks

def main():
    print("Extracting SIGINT PDFs...")
    pdf_files = [os.path.join(SIGINT_DIR, f) for f in TARGET_PDFS]
    
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue
            
        print(f"Processing {os.path.basename(pdf_path)}...")
        chunks = extract_chunks(pdf_path)
        
        for idx, chunk in enumerate(chunks):
            out_name = f"{os.path.basename(pdf_path)}_chunk_{idx}.md"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            # Inject governance markers to ensure we test preservation
            synthesized_context = (
                f"Source: [{os.path.basename(pdf_path)}]\n"
                f"Governance: approval_required, risk_label_HIGH\n"
                f"Config: config_sigint_strict_mode\n"
                f"Date: 2026-05-15\n"
                f"Note: This information is [CONTRADICTION] to prior reports. Do not proceed unless authorized.\n"
                f"---\n\n"
                f"{chunk}"
            )
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(synthesized_context)
                
            print(f"Saved {out_name} ({len(synthesized_context.split())} tokens)")

if __name__ == "__main__":
    main()
