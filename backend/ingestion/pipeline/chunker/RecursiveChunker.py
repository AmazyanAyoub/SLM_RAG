from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Initialize the Recursive Character Text Splitter.
        
        Args:
            chunk_size: The target size of each chunk (in characters/tokens approx).
            chunk_overlap: How much context to keep from the previous chunk.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Priority list for splitting: 
            # 1. Double newlines (Paragraphs)
            # 2. Single newlines (Lines)
            # 3. Periods (Sentences)
            # 4. Spaces (Words)
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits the text into chunks and attaches metadata to each.
        
        Returns:
            List of dicts: [{"text": "...", "metadata": {...}}, ...]
        """
        if not text:
            return []

        # Generate simple string chunks
        raw_chunks = self.splitter.split_text(text)
        
        structured_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_data = {
                "text": chunk_text,
                "metadata": (metadata or {}).copy()
            }
            # Add chunk-specific metadata
            chunk_data["metadata"]["chunk_index"] = i
            chunk_data["metadata"]["total_chunks"] = len(raw_chunks)
            
            structured_chunks.append(chunk_data)
            
        return structured_chunks