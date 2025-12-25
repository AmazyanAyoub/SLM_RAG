import os
import sys
import asyncio
import time
import uuid
# from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.indexing.postgres_client import PostgresVectorDB
from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.pipeline.chunking import Chunker
from backend.ingestion.pipeline.contextual_enrichment import ContextualEnricher
from backend.models.embedding_client import embed_documents, embed_sparse

# --- CONFIGURATION ---
DATA_DIR = Path("data/pdfs")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

async def main():
    print("🚀 STARTING POSTGRES INGESTION")
    print("==================================================")
    
    # 1. Initialize DB
    try:
        db = PostgresVectorDB()
        print(f"🔌 Connected to Postgres Table: '{db.table_name}'")
    except Exception as e:
        print(f"❌ Failed to connect to Postgres: {e}")
        return

    # Initialize Enricher
    try:
        enricher = ContextualEnricher()
        print("✅ Teacher LLM Connected for Enrichment.")
    except Exception as e:
        enricher = None
        print(f"⚠️ Contextual Enricher skipped: {e}")

    # 2. Load & Chunk
    if not DATA_DIR.exists():
        print(f"⚠️ Directory {DATA_DIR} does not exist. Creating it...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"👉 Please put your PDF files in {DATA_DIR} and run this script again.")
        return

    loader = PDFLoader()
    raw_docs = []
    files = list(DATA_DIR.glob("*.pdf"))
    print(f"📂 Found {len(files)} PDF files in {DATA_DIR}")

    for file_path in files:
        try:
            text = loader.load_file(str(file_path))
            if text.strip():
                raw_docs.append({"text": text, "source": file_path.name})
                print(f"   ✅ Loaded: {file_path.name} ({len(text)} chars)")
            else:
                print(f"   ⚠️ Skipped empty file: {file_path.name}")
        except Exception as e:
            print(f"   ❌ Error loading {file_path.name}: {e}")

    if not raw_docs:
        return

    chunker = Chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in raw_docs:
        doc_chunks = chunker.chunk_text(doc["text"], metadata={"source": doc["source"]})
        for c in doc_chunks:
            c["search_content"] = c["text"]
        chunks.extend(doc_chunks)

    print(f"📦 Generated {len(chunks)} chunks.")
    # enricher = None
    # 3. Contextual Enrichment (Async)
    if enricher:
        print("   👨‍🏫 Enriching chunks (using Neighbor Window strategy)...")
        tasks = []
        window_size = 3
        
        for i, chunk in enumerate(chunks):
            start_i = max(0, i - window_size)
            end_i = min(len(chunks), i + window_size + 1)
            neighbor_text = "\n---\n".join([c["text"] for c in chunks[start_i:end_i]])
            tasks.append(enricher.enrich_chunk(chunk["text"], neighbor_text))
            
        # Run in batches
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]
            
            enriched_texts = await asyncio.gather(*batch_tasks)
            
            for j, result_text in enumerate(enriched_texts):
                batch_chunks[j]["search_content"] = result_text
            print(f"      Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...", end="\r")
        print("\n   ✅ Enrichment Complete.")
    else:
        # Fallback
        for c in chunks:
            c["search_content"] = c["text"]
            c["display_content"] = c["text"]

    # 4. Generate Embeddings (Batch Processing)
    print("🧠 Generating Embeddings (Dense + Sparse)...")
    texts = [c["search_content"] for c in chunks]
    
    start_embed = time.time()
    dense_vectors = embed_documents(texts)
    sparse_vectors = embed_sparse(texts) # Returns list of dicts {token_id: weight}
    print(f"   ✅ Embeddings generated in {time.time() - start_embed:.2f}s")

    # 5. Prepare Data for Upsert
    points = []
    for i, chunk in enumerate(chunks):
        # Create Deterministic ID (Source + Index)
        unique_str = f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_index']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        
        # Format Sparse Vector for Postgres Client
        sp_indices = list(int(k) for k in sparse_vectors[i].keys())
        sp_values = list(float(v) for v in sparse_vectors[i].values())

        points.append({
            "payload": {**chunk, "id": point_id}, # Store text & metadata in payload
            "vector": {
                "dense": dense_vectors[i],
                "sparse": {"indices": sp_indices, "values": sp_values}
            }
        })

    # 6. Upsert to Postgres
    print(f"📤 Inserting {len(points)} records into Postgres...")
    start_upsert = time.time()
    db.upsert(points)
    print(f"✅ Ingestion Complete! (Upsert took {time.time() - start_upsert:.2f}s)")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
