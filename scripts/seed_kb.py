"""
Seed the ChromaDB knowledge base from two sources:
  1. Medical book PDF  (primary — strict RAG source)
  2. openFDA API       (drug labels supplement)

Run once before launching the app:
    python3 scripts/seed_kb.py
"""

import os
import sys
import requests
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH         = os.path.join(BASE_DIR, "db")
COLLECTION_NAME = "medical_knowledge"
MODEL_NAME      = "all-MiniLM-L6-v2"
PDF_PATH        = "/Users/geetikaa/Downloads/Medical_book.pdf"

CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks

# Drugs to supplement from openFDA
DRUGS = [
    "paracetamol", "ibuprofen", "aspirin", "metformin",
    "lisinopril", "amlodipine", "atorvastatin", "omeprazole",
    "amoxicillin", "metoprolol", "salbutamol", "prednisone",
]

OPENFDA_FIELDS = [
    "warnings", "contraindications",
    "drug_interactions", "adverse_reactions", "indications_and_usage",
]


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_pdf_chunks(pdf_path: str) -> list[tuple[str, str]]:
    """Returns list of (chunk_text, doc_id) tuples."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    print(f"📖  Opening PDF: {pdf_path}")
    doc   = fitz.open(pdf_path)
    pages = len(doc)
    print(f"    {pages} pages found.")

    chunks = []
    chunk_id = 0

    for page_num in range(pages):
        text = doc[page_num].get_text().strip()
        if not text or len(text) < 50:
            continue

        # Slide a window over the page text
        start = 0
        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if len(chunk) > 80:  # skip tiny fragments
                chunks.append((f"MEDICAL_BOOK_p{page_num+1}_{chunk_id}", chunk))
                chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"    Extracted {len(chunks)} chunks from PDF.")
    return chunks


# ── openFDA Supplement ────────────────────────────────────────────────────────

def fetch_drug_chunks(drug_name: str) -> list[tuple[str, str]]:
    url = (
        f"https://api.fda.gov/drug/label.json"
        f"?search=openfda.generic_name:{drug_name}&limit=1"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if "results" not in data:
            return []
        result = data["results"][0]
        chunks = []
        for field in OPENFDA_FIELDS:
            if field not in result:
                continue
            raw  = result[field]
            text = " ".join(raw) if isinstance(raw, list) else raw
            chunk = f"{drug_name.upper()} | {field.replace('_', ' ').title()}: {text[:600]}"
            doc_id = f"fda_{drug_name}_{field}"
            chunks.append((doc_id, chunk))
        return chunks
    except Exception as e:
        print(f"  ⚠️  Could not fetch {drug_name}: {e}")
        return []


# ── Main Seed Function ────────────────────────────────────────────────────────

def seed():
    print("\n🔄  Initialising embedding model...")
    embed_model = SentenceTransformer(MODEL_NAME)

    print("🔄  Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("   Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    all_ids, all_docs = [], []

    # ── 1. PDF ────────────────────────────────────────────────────
    if os.path.exists(PDF_PATH):
        print("\n📚  Extracting medical book...")
        pdf_chunks = extract_pdf_chunks(PDF_PATH)
        for doc_id, text in pdf_chunks:
            all_ids.append(doc_id)
            all_docs.append(text)
    else:
        print(f"⚠️  PDF not found at {PDF_PATH} — skipping book ingestion.")

    # ── 2. openFDA ────────────────────────────────────────────────
    print("\n🌐  Fetching openFDA drug data...")
    for drug in DRUGS:
        print(f"   {drug}")
        for doc_id, text in fetch_drug_chunks(drug):
            all_ids.append(doc_id)
            all_docs.append(text)

    if not all_docs:
        print("❌  No documents to embed. Exiting.")
        return

    # ── 3. Embed & Store ─────────────────────────────────────────
    print(f"\n🔄  Embedding {len(all_docs)} total chunks (this may take a few minutes)...")

    BATCH = 256
    for i in range(0, len(all_docs), BATCH):
        batch_docs = all_docs[i:i + BATCH]
        batch_ids  = all_ids[i:i + BATCH]
        embeddings = embed_model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(documents=batch_docs, embeddings=embeddings, ids=batch_ids)
        print(f"   Stored batch {i // BATCH + 1}/{(len(all_docs) - 1) // BATCH + 1}")

    print(f"\n✅  Knowledge base ready: {len(all_docs)} chunks")
    print(f"    PDF chunks : {sum(1 for i in all_ids if i.startswith('MEDICAL_BOOK'))}")
    print(f"    FDA chunks : {sum(1 for i in all_ids if i.startswith('fda_'))}")


if __name__ == "__main__":
    seed()
