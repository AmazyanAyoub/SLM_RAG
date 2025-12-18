import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.pipeline.chunking import Chunker
from backend.ingestion.pipeline.contextual_enrichment import ContextualEnricher
from backend.indexing.dense_index import DenseIndexer
from backend.indexing.sparse_index import SparseIndexer

# Load Environment Variables
load_dotenv()

async def main():
    print("🚀 STARTING INGESTION PIPELINE (2025 Architecture)")
    print("============================================================")

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
    chunker = Chunker(chunk_size=1024, chunk_overlap=200)
    
    # Initialize Enricher
    try:
        enricher = ContextualEnricher()
        print("✅ Teacher LLM Connected for Enrichment.")
    except Exception as e:
        enricher = None
        print(f"⚠️ Contextual Enricher skipped: {e}")

    # Initialize BOTH Indexers
    dense_indexer = DenseIndexer()   # Creates the Point
    sparse_indexer = SparseIndexer() # Adds Keywords to the Point

    # 2. PROCESS FILES
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        
        # A. LOAD
        try:
            raw_text = loader.load_file(pdf_file)
            if not raw_text:
                print("   ⚠️ Loader returned empty text.")
                continue
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            continue

        # B. CHUNK
        # Pass filename as metadata "source" to avoid 'unknown' duplicates
        chunks = chunker.chunk_text(raw_text, metadata={"source": pdf_file.name})
        print(f"   ✂️ Generated {len(chunks)} chunks.")

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
                
                # Run parallel
                enriched_texts = await asyncio.gather(*batch_tasks)
                
                # Save results
                for j, result_text in enumerate(enriched_texts):
                    batch_chunks[j]["search_content"] = result_text
                    batch_chunks[j]["display_content"] = batch_chunks[j]["text"]
                
                print(f"      Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...", end="\r")
            
            print("\n   ✅ Enrichment Complete.")
        else:
            # Fallback
            for c in chunks:
                c["search_content"] = c["text"]
                c["display_content"] = c["text"]

        # D. INDEXING (Hybrid)
        if chunks:
            # Step 1: Dense Vectors (Upsert - Creates the Point)
            # This MUST run first to create the point ID
            print("   🧠 Step 1: Generating Dense Vectors...")
            try:
                dense_indexer.index_chunks(chunks)
            except Exception as e:
                print(f"   ❌ Dense Indexing Failed: {e}")
                continue

            # Step 2: Sparse Vectors (Update - Adds to the Point)
            # This adds the keyword data to the existing point
            print("   🧠 Step 2: Generating Sparse Vectors...")
            try:
                sparse_indexer.index_chunks(chunks)
            except Exception as e:
                print(f"   ❌ Sparse Indexing Failed: {e}")
                continue
            
            print(f"   ✅ File '{pdf_file.name}' Fully Indexed (Hybrid)!")

    print("\n" + "=" * 60)
    print("🎉 INGESTION COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())