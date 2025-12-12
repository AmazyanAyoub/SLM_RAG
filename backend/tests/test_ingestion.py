import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load env vars (Need GROQ_API_KEY)
load_dotenv()

# # Add project root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.pipeline.chunking import Chunker
from backend.ingestion.pipeline.contextual_enrichment import ContextualEnricher

async def test_full_ingestion_pipeline():
    print("🧪 STARTING FULL PIPELINE TEST: Load -> Chunk -> Enrich")
    print("=" * 60)
    
    # 1. SETUP PATHS
    base_dir = Path("data/pdfs")
    pdf_files = list(base_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDFs found in '{base_dir}'.")
        return
    target_pdf = pdf_files[0]

    # 2. RUN LOADER
    print(f"📂 Step 1: Loading '{target_pdf.name}'...")
    loader = PDFLoader()
    full_text = loader.load_file(str(target_pdf))
    print(f"   ✅ Loaded {len(full_text)} chars.")

    # 3. RUN CHUNKER
    print("\n✂️ Step 2: Chunking...")
    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_text(full_text, metadata={"source": target_pdf.name})
    print(f"   ✅ Generated {len(chunks)} raw chunks.")

    # 4. RUN CONTEXTUAL ENRICHMENT (The 2025 Upgrade)
    print("\n👨‍🏫 Step 3: Enriching Chunks with Teacher LLM...")
    
    if not os.getenv("GROQ_API_KEY"):
        print("   ❌ SKIPPING: GROQ_API_KEY not found in .env file.")
        return

    enricher = ContextualEnricher()
    
    # NOTE: In a real run, we enrich ALL chunks. 
    # For this test, we only enrich the first 3 to save time/tokens.
    test_chunks = chunks[:3] 
    
    for i, chunk in enumerate(test_chunks):
        print(f"   🔄 Enriching Chunk {i}...")
        
        # Call the Teacher
        original_text = chunk["text"]
        enriched_text = await enricher.enrich_chunk(original_text, full_text)
        
        # Save it back to the chunk structure
        chunk["search_content"] = enriched_text  # This is what we embed
        chunk["display_content"] = original_text # This is what we show users
        
    print(f"   ✅ Successfully enriched {len(test_chunks)} chunks.")

    # 5. FINAL INSPECTION
    print("\n🔎 COMPARISON: Raw vs. Enriched (Chunk 0)")
    print("=" * 60)
    print(f"🔴 BEFORE (Raw Chunk):\n{test_chunks[0]['display_content'][:300]}...")
    print("-" * 60)
    print(f"🟢 AFTER (Enriched for Search):\n{test_chunks[0]['search_content'][:500]}...")
    print("=" * 60)

if __name__ == "__main__":
    # We use asyncio.run because the Enricher is async
    asyncio.run(test_full_ingestion_pipeline())