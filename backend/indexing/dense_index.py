# BGE-M3 embedding logic
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from backend.indexing.vector_store import VectorDBClient
from backend.core.config_loader import settings
from qdrant_client.http import models
import uuid


class DenseIndexer:
    def __init__(self):
        """
        Initialize the Embedding Model (BGE-M3) and the Database Client.
        """
        model_name = settings.retrieval.embedder_model # e.g., "BAAI/bge-m3"
        print(f"🧠 Loading Embedding Model: {model_name}...")
        
        # We use SentenceTransformer for easy local embedding
        # device="cpu" or "cuda" (if you have an NVIDIA GPU)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.db_client = VectorDBClient()
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of strings into a list of vectors.
        """
        # BGE-M3 is powerful but can be heavy. 
        # We normalize embeddings to ensure Dot Product works like Cosine Similarity.
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Takes the enriched chunks, embeds them, and upserts to Qdrant.
        """
        if not chunks:
            print("⚠️ No chunks to index.")
            return

        print(f"🚀 Embedding {len(chunks)} chunks...")
        start_time = time.time()

        # 1. Extract the text we want to SEARCH (The Enriched Content)
        # Fallback to normal 'text' if 'search_content' is missing
        search_texts = [c.get("search_content", c["text"]) for c in chunks]
        
        # 2. Generate Vectors
        vectors = self.embed_texts(search_texts)
        
        # 3. Prepare Points for Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            # Create a unique string signature for this chunk
            # We use Source Filename + Chunk Index as the signature.
            # This ensures that "Chunk 5 of manual.pdf" ALWAYS gets the same ID.
            source = chunk["metadata"].get("source", "unknown")
            index = chunk["metadata"].get("chunk_index", i)
            signature = f"{source}_{index}"

            # Generate a DETERMINISTIC UUID based on that signature
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, signature))
            
            # Metadata to store in DB (Payload)
            payload = {
                "text": chunk.get("display_content", chunk["text"]), # What we show the user
                "source": chunk["metadata"].get("source", "unknown"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "context_summary": chunk.get("search_content", "")[:200] # Optional: store preview of context
            }

            # Create the Point
            points.append(models.PointStruct(
                id=point_id,
                vector=vectors[i],
                payload=payload
            ))

        # 4. Upload to Qdrant
        print(f"📤 Uploading {len(points)} vectors to Qdrant...")
        self.db_client.upsert(points)
        
        end_time = time.time()
        print(f"✅ Indexing Complete! Time taken: {end_time - start_time:.2f}s")

# Test Block
if __name__ == "__main__":
    # Dummy data test
    indexer = DenseIndexer()
    
    dummy_chunks = [
        {
            "text": "The price of the pro plan is $20.",
            "search_content": "Context: Pricing. Content: The price of the pro plan is $20.",
            "metadata": {"source": "test", "chunk_index": 0}
        },
        {
            "text": "To reset password, click settings.",
            "search_content": "Context: Security. Content: To reset password, click settings.",
            "metadata": {"source": "test", "chunk_index": 1}
        }
    ]
    
    indexer.index_chunks(dummy_chunks)