from pathlib import Path
from pypdf import PdfReader
import json

PDF_DIR = Path("Dataset/pdfs")
OUTPUT_FILE = Path("chunks.json")

CHUNK_SIZE = 1500
OVERLAP = 200

chunks = []

for pdf_path in PDF_DIR.rglob("*.pdf"):
    reader = PdfReader(str(pdf_path))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    text = " ".join(text.split())

    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "id": f"{pdf_path.stem}_{chunk_id}",
                "source": str(pdf_path),
                "text": chunk_text
            })

        chunk_id += 1
        start += CHUNK_SIZE - OVERLAP

print(f"Created {len(chunks)} chunks.")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Saved chunks to: {OUTPUT_FILE}")
