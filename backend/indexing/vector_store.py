# Qdrant client wrapper
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.core.config_loader import settings

class VectorDBClient:
    def __init__(self):
        """
        Initialize Qdrant Client using settings from config_loader.
        """
        if not settings:
            raise ValueError("❌ Settings not loaded. Check config_loader.py")

        print(f"🔌 Connecting to Qdrant at {settings.retrieval.vector_store_host}:{settings.retrieval.vector_store_port}...")
        
        self.client = QdrantClient(
            host=settings.retrieval.vector_store_host,
            port=settings.retrieval.vector_store_port,
            api_key=settings.retrieval.vector_store_api_key,
            https=False,
        )
        self.collection_name = settings.retrieval.vector_store_collection
        self.vector_size = 1024 # Standard for BGE-M3 dense embeddings

        # Auto-create collection on startup
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Checks if collection exists; if not, creates it with Hybrid Search support.
        """
        if not self.client.collection_exists(self.collection_name):
            print(f"📦 Collection '{self.collection_name}' not found. Creating...")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                # Enable sparse vectors for Hybrid Search (BM25 equivalent)
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True, # Save RAM
                        )
                    )
                }
            )
            print(f"✅ Collection '{self.collection_name}' created successfully.")
        else:
            print(f"✅ Connected to existing collection: '{self.collection_name}'")

    def upsert(self, points: list):
        """
        Insert chunks (points) into Qdrant.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # def search(self, query_vector: list, limit: int = 5):
    #     """
    #     Standard Dense Search
    #     """
    #     return self.client.search(
    #         collection_name=self.collection_name,
    #         query_vector=query_vector,
    #         limit=limit
    #     )

    def search(self, query_vector: list, limit: int = 5):
            """
            SOTA Search using the new Query API (v1.10+)
            """
            # "query_points" is the modern replacement for "search"
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            # The new API returns an object with a .points attribute
            return results.points

# Simple test block
if __name__ == "__main__":
    try:
        db = VectorDBClient()
        print("🚀 Qdrant Connection Successful!")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")