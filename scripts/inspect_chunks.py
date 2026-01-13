import os
import sys
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.pipeline.chunking import Chunker

def inspect_pdf_hierarchy(pdf_path):
    print(f"🔍 Inspecting: {pdf_path}")
    
    # Initialize Chunker
    # We use a smaller chunk size (500) to force splitting of large sections
    # This helps verify that multiple chunks can share the same hierarchy path (Inheritance)
    chunker = Chunker(strategy="docling", chunk_size=500, chunk_overlap=50)
    
    try:
        chunks = chunker.chunk(file_path=pdf_path, metadata={"source": os.path.basename(pdf_path)})
    except Exception as e:
        print(f"❌ Error chunking file: {e}")
        return

    print(f"📊 Total Chunks Generated: {len(chunks)}")
    
    if not chunks:
        print("⚠️ No chunks generated.")
        return

    # 1. Group by Exact Path (Siblings)
    path_map = defaultdict(list)
    for c in chunks:
        path = c["metadata"].get("hierarchy_path", "Root")
        path_map[path].append(c)
        
    print(f"🌳 Found {len(path_map)} unique hierarchy paths.")
    
    # 2. Analyze Inheritance (Shared Paths / Siblings)
    multi_chunk_paths = {k: v for k, v in path_map.items() if len(v) > 1}
    print(f"👨‍👩‍👧‍👦 Paths with multiple chunks (Siblings): {len(multi_chunk_paths)}")
    
    # 3. Analyze Parent-Child Relationships (Prefix Matching)
    # We look for paths that are prefixes of other paths
    paths = sorted(path_map.keys())
    parents = set()
    for p in paths:
        # If path is "Chapter 1 / Section 1", parent is "Chapter 1"
        parts = p.split(' / ')
        if len(parts) > 1:
            parent_path = " / ".join(parts[:-1])
            # Check if we actually have chunks for the parent path itself
            if parent_path in path_map:
                parents.add(parent_path)

    print(f"🧬 Paths acting as Parents to others: {len(parents)}")

    print("\n--- TOP 5 PATHS BY CHUNK COUNT (Siblings) ---")
    sorted_paths = sorted(path_map.items(), key=lambda x: len(x[1]), reverse=True)
    for path, group in sorted_paths[:5]:
        print(f"[{len(group)} chunks] {path}")

    print("\n--- HIERARCHY TREE SAMPLE (First 15) ---")
    for path in paths[:15]:
        print(f"  > {path}")

if __name__ == "__main__":
    # Find a PDF
    data_dir = Path("data/pdfs")
    files = list(data_dir.glob("*.pdf"))
    if files:
        inspect_pdf_hierarchy(str(files[0]))
    else:
        print("No PDFs found in data/pdfs")
