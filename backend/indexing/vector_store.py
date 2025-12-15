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
                # FIX 1: Explicitly name the dense vector "dense"
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                # FIX 2: Rename sparse vector to "sparse" to match search logic
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True, # Save RAM, good choice
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

    def search(self, query_dense: list, query_sparse_indices: list, query_sparse_values: list, limit: int = 5):
        """
        Performs Hybrid Search (Dense + Sparse) with RRF Fusion.
        """
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense", # Now this matches FIX 1
                    limit=limit * 2, 
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse_indices,
                        values=query_sparse_values
                    ),
                    using="sparse", # Now this matches FIX 2
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), # Reciprocal Rank Fusion
            limit=limit,
            with_payload=True
        ).points

# Simple test block
if __name__ == "__main__":
    try:
        db = VectorDBClient()
        print("🚀 Qdrant Connection Successful!")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")