from typing import List, Dict, Any, Optional
import os

# Import Strategies
from backend.ingestion.pipeline.chunker.RecursiveChunker import RecursiveChunker
from backend.ingestion.pipeline.chunker.DoclingChunker import DoclingChunker

class Chunker:
    def __init__(self, strategy: str = "docling", chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Controller for chunking strategies.
        
        Args:
            strategy: 'recursive' or 'docling'
            chunk_size: Target size for recursive chunker
            chunk_overlap: Overlap for recursive chunker
        """
        self.strategy = strategy.lower()
        
        if self.strategy == "recursive":
            self.processor = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif self.strategy == "docling":
            self.processor = DoclingChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError(f"❌ Unknown chunking strategy: '{strategy}'. Use 'recursive' or 'docling'.")

    def chunk(self, text: Optional[str] = None, file_path: Optional[str] = None, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Universal chunking method.
        
        Args:
            text: Raw text content (Required for 'recursive')
            file_path: Path to PDF file (Required for 'docling')
            metadata: Metadata dict to attach to chunks
        """
        if self.strategy == "recursive":
            if text is None:
                raise ValueError("❌ Strategy 'recursive' requires 'text' argument.")
            return self.processor.chunk(text, metadata)
            
        elif self.strategy == "docling":
            if not file_path:
                raise ValueError("❌ Strategy 'docling' requires 'file_path' argument.")
            return self.processor.chunk(file_path, metadata)
            
        return []

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Legacy wrapper for backward compatibility with existing ingestion scripts.
        """
        if self.strategy == "docling":
            # Try to infer file path from metadata
            source = metadata.get("source") if metadata else None
            if source and os.path.exists(source):
                return self.chunk(file_path=source, metadata=metadata)
            
            # If source is just a filename (e.g. "doc.pdf") and not full path, we can't process it with Docling
            # unless we know the directory.
            raise ValueError(f"❌ Strategy 'docling' requires a valid file path. Metadata source '{source}' not found or invalid.")
        
        return self.chunk(text=text, metadata=metadata)

# Test Block
if __name__ == "__main__":
    print("--- 🧪 TESTING RECURSIVE STRATEGY ---")
    # Dummy text to test the split
    sample_text = (
        "Artificial Intelligence is changing the world.\n\n"
        "Machine Learning is a subset of AI. Deep Learning is a subset of ML.\n"
        "We are building a RAG system using Python."
    )
    
    # 1. Test Recursive
    chunker_rec = Chunker(strategy="recursive", chunk_size=50, chunk_overlap=10)
    result_rec = chunker_rec.chunk(text=sample_text, metadata={"source": "test_doc"})
    
    print(f"✂️ Created {len(result_rec)} chunks:")
    for c in result_rec:
        print(f"   [{c['metadata']['chunk_index']}] {c['text']}")

    print("\n--- 🧪 TESTING DOCLING STRATEGY ---")
    # 2. Test Docling (Requires a real PDF)
    import glob
    pdf_files = glob.glob("data/pdfs/*.pdf")
    
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"📄 Found PDF: {test_pdf}")
        chunker_doc = Chunker(strategy="docling")
        result_doc = chunker_doc.chunk(file_path=test_pdf, metadata={"source": test_pdf})
        
        print(f"✂️ Created {len(result_doc)} chunks:")
        for c in result_doc[:3]:
            print(f"   [{c['metadata']['chunk_index']}] Path: {c['metadata'].get('hierarchy_path', 'Root')}")
            print(f"       Content: {c['text'][:100].replace(chr(10), ' ')}...")
    else:
        print("⚠️ No PDFs found in 'data/pdfs/' to test Docling strategy.")