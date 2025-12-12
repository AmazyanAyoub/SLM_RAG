# CLI to run ingestion
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.pipeline.chunking import Chunker
from backend.ingestion.pipeline.contextual_enrichment import ContextualEnricher
from backend.indexing.dense_index import DenseIndexer

# Load Environment Variables
load_dotenv()

async def main():
    print("🚀 STARTING INGESTION PIPELINE (2025 Architecture)")
    print("=" * 60)

    # 1. SETUP
    data_dir = Path("data/pdfs")
    if not data_dir.exists():
        print(f"❌ Error: Directory '{data_dir}' not found.")
        return

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDFs found. Please add files to 'data/pdfs'.")
        return

    # Initialize Components
    loader = PDFLoader()
    chunker = Chunker(chunk_size=512, chunk_overlap=100)
    
    # Only init Enricher if we have a key
    if os.getenv("OLLAMA_BASE_URL"):
        enricher = ContextualEnricher()
        print("✅ Teacher LLM (qwen3:8b) Connected for Enrichment.")
    else:
        enricher = None
        print("⚠️ No GROQ_API_KEY found. Skipping Contextual Enrichment.")

    indexer = DenseIndexer() # Loads BGE-M3 (Takes memory!)

    # 2. PROCESS FILES
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        
        # A. LOAD
        try:
            raw_text = loader.load_file(str(pdf_file))
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            continue

        # B. CHUNK
        chunks = chunker.chunk_text(raw_text, metadata={"source": pdf_file.name})
        print(f"   ✂️ Generated {len(chunks)} chunks.")

        # C. ENRICH (Phase 1.5)
        # enricher = None
        if enricher:
            print("   👨‍🏫 Enriching chunks (This may take a mom" \
            "ent)...")
            # We process in batches of 10 to avoid hitting API rate limits
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                
                # Create async tasks for the batch
                tasks = [
                    enricher.enrich_chunk(chunk["text"], raw_text) 
                    for chunk in batch
                ]
                
                # Run batch in parallel
                enriched_texts = await asyncio.gather(*tasks)
                
                # Update the chunks with the new enriched text
                for j, result_text in enumerate(enriched_texts):
                    batch[j]["search_content"] = result_text      # FOR SEARCH
                    batch[j]["display_content"] = batch[j]["text"] # FOR DISPLAY
                
                print(f"      Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...", end="\r")

                await asyncio.sleep(1)
            print("\n   ✅ Enrichment Complete.")
        else:
            # Fallback if no API key
            for c in chunks:
                c["search_content"] = c["text"]
                c["display_content"] = c["text"]

        # D. INDEX
        print("   🧠 Embedding and Indexing to Qdrant...")
        try:
            indexer.index_chunks(chunks)
            print("   ✅ File Indexed Successfully!")
        except Exception as e:
            print(f"   ❌ Indexing Failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 INGESTION COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())