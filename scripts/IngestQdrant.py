import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import uuid
import time

# Add project root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.pipeline.chunking import Chunker
from backend.ingestion.pipeline.contextual_enrichment import ContextualEnricher
from backend.indexing.vector_store import VectorDBClient
from backend.models.embedding_client import embed_documents, embed_sparse
from qdrant_client.http import models

# Load Environment Variables
load_dotenv()

async def main():
    print("🚀 STARTING INGESTION PIPELINE (2025 Architecture)")
    print("============================================================")
    
    # Force Qdrant for this script
    os.environ["VECTOR_DB_PROVIDER"] = "qdrant"

    # 1. SETUP
    data_dir = Path("data/pdfs") # Ensure this matches your folder name (data/pdfs or data/raw)
    if not data_dir.exists():
        print(f"❌ Error: Directory '{data_dir}' not found.")
        return

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDFs found in {data_dir}. Please add files.")
        return

    # Initialize Components
    loader = PDFLoader()
    chunker = Chunker(strategy="docling", chunk_size=1024, chunk_overlap=200)
    
    # Initialize Enricher
    try:
        enricher = ContextualEnricher()
        print("✅ Teacher LLM Connected for Enrichment.")
    except Exception as e:
        enricher = None
        print(f"⚠️ Contextual Enricher skipped: {e}")

    # Initialize Vector DB Client
    db_client = VectorDBClient()

    # 2. PROCESS FILES
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        
        # A. LOAD
        chunks = []
        
        if chunker.strategy == "docling":
            print("   🧠 Using Docling Smart Chunking...")
            chunks = chunker.chunk(file_path=str(pdf_file), metadata={"source": pdf_file.name})
        else:
            try:
                raw_text = loader.load_file(pdf_file)
                if not raw_text:
                    print("   ⚠️ Loader returned empty text.")
                    continue
                
                # B. CHUNK
                # Pass filename as metadata "source" to avoid 'unknown' duplicates
                chunks = chunker.chunk_text(raw_text, metadata={"source": pdf_file.name})
            except Exception as e:
                print(f"   ❌ Failed to load/chunk: {e}")
                continue

        print(f"   ✂️ Generated {len(chunks)} chunks.")
        enricher = None

        # C. ENRICH (SOTA Neighbor Window)
        if enricher:
            print("   👨‍🏫 Enriching chunks (using Neighbor Window strategy)...")
            
            tasks = []
            window_size = 3  # How many chunks before/after to include
            
            for i, chunk in enumerate(chunks):
                # Calculate window indices
                start_i = max(0, i - window_size)
                end_i = min(len(chunks), i + window_size + 1)
                
                # Join the text of the neighbors to form the context
                neighbor_text = "\n---\n".join([c["text"] for c in chunks[start_i:end_i]])
                
                # Add task
                tasks.append(enricher.enrich_chunk(chunk["text"], neighbor_text))

            # Run Batches
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_chunks = chunks[i : i + batch_size]
                enriched_texts = await asyncio.gather(*batch_tasks)
                for j, res in enumerate(enriched_texts):
                    # Only store search_content if it provides new information
                    if res != batch_chunks[j]["text"]:
                        batch_chunks[j]["search_content"] = res
                print(f"      Processed {min(i + batch_size, len(chunks))}/{len(chunks)}...", end="\r")
            print("\n   ✅ Enrichment Complete.")

        # D. INDEXING (Hybrid)
        if chunks:
            print("   🧠 Generating Embeddings (Dense + Sparse)...")
            try:
                # 1. Prepare Text
                search_texts = [c.get("search_content", c["text"]) for c in chunks]
                
                # 2. Generate Embeddings
                dense_vectors = embed_documents(search_texts)
                sparse_vectors = embed_sparse(search_texts)
                
                # 3. Prepare Points for Qdrant
                points = []
                for i, chunk in enumerate(chunks):
                    source = chunk["metadata"].get("source", "unknown")
                    index = chunk["metadata"].get("chunk_index", i)
                    signature = f"{source}_{index}"
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, signature))
                    
                    # Convert Sparse to Qdrant format
                    sp_indices = list(int(k) for k in sparse_vectors[i].keys())
                    sp_values = list(float(v) for v in sparse_vectors[i].values())
                    
                    points.append(models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vectors[i],
                            "sparse": models.SparseVector(indices=sp_indices, values=sp_values)
                        },
                        payload=chunk
                    ))
                
                # 4. Upsert
                print(f"   📤 Upserting {len(points)} points to Qdrant...")
                db_client.upsert(points)
                print(f"   ✅ File '{pdf_file.name}' Fully Indexed!")
                
            except Exception as e:
                print(f"   ❌ Indexing Failed: {e}")
                continue

    print("🎉 INGESTION COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())