# PyMuPDF / Unstructured loader
import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path
import logging

# Configure logger locally
logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self):
        """
        Initialize PDF Loader using PyMuPDF (fastest/most reliable).
        """
        pass

    def load_file(self, file_path: str) -> str:
        """
        Extracts full text from a PDF file.
        Returns: A single string containing the entire document text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ PDF not found at: {path}")

        try:
            doc = fitz.open(path)
            full_text = []
            
            # Iterate over pages
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Optional: Add page markers for better context later
                # text = f"--- Page {page_num + 1} ---\n{text}" 
                full_text.append(text)
                
            logger.info(f"✅ Loaded PDF '{path.name}' with {len(doc)} pages.")
            return "\n".join(full_text)

        except Exception as e:
            logger.error(f"⚠️ Failed to load PDF {path.name}: {e}")
            raise e

# Simple Test
if __name__ == "__main__":
    # Create a dummy PDF for testing if one doesn't exist
    # (Or point this to a real PDF path on your machine)
    loader = PDFLoader()
    print("⚠️ Please point the test to a real PDF file path in the code below.")
    text = loader.load_file("data/pdfs/LIASI - Règlement d'application - 19-06-2007 - 31-12-2024.pdf")
    print(text[:500])