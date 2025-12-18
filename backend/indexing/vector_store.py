# Qdrant client wrapper
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.core.config_loader import settings
from sentence_transformers import CrossEncoder

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
        # Append _large to create a separate collection for the larger chunks
        self.collection_name = f"{settings.retrieval.vector_store_collection}_large"
        self.vector_size = 1024 # Standard for BGE-M3 dense embeddings
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)

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

    # def search(self, query_dense: list, query_sparse_indices: list, query_sparse_values: list, limit: int = 5):
    #     """
    #     Performs Hybrid Search (Dense + Sparse) with RRF Fusion.
    #     """
    #     return self.client.query_points(
    #         collection_name=self.collection_name,
    #         prefetch=[
    #             models.Prefetch(
    #                 query=query_dense,
    #                 using="dense", # Now this matches FIX 1
    #                 limit=limit * 2, 
    #             ),
    #             models.Prefetch(
    #                 query=models.SparseVector(
    #                     indices=query_sparse_indices,
    #                     values=query_sparse_values
    #                 ),
    #                 using="sparse", # Now this matches FIX 2
    #                 limit=limit * 2,
    #             ),
    #         ],
    #         query=models.FusionQuery(fusion=models.Fusion.RRF), # Reciprocal Rank Fusion
    #         limit=limit,
    #         with_payload=True
    #     ).points

    def search(self, query_text: str, query_dense: list, query_sparse_indices: list, query_sparse_values: list, limit: int = 5):
        """
        Performs Hybrid Search (Dense + Sparse) followed by Re-Ranking.
        """
        # 1. RETRIEVE CANDIDATES (Fetch more than needed, e.g., Top 15)
        # We need a pool of candidates for the judge to review.
        initial_limit = 15 
        
        candidates = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=initial_limit, 
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse_indices,
                        values=query_sparse_values
                    ),
                    using="sparse",
                    limit=initial_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=initial_limit,
            with_payload=True
        ).points

        if not candidates:
            return []

        # 2. RE-RANKING (The Judge Step)
        # Prepare pairs: [[Query, Text1], [Query, Text2], ...]
        pairs = []
        for hit in candidates:
            # Use 'search_content' if available, otherwise reconstruct from summary + text
            doc_text = hit.payload.get("search_content")
            if not doc_text:
                summary = hit.payload.get("context_summary", "")
                raw_text = hit.payload.get("text", "")
                doc_text = f"{summary}\n{raw_text}" if summary else raw_text

            pairs.append([query_text, doc_text])

        # Compute scores (The higher the score, the more relevant)
        # This returns a list of float scores, e.g., [0.98, 0.12, 0.55...]
        scores = self.reranker.predict(pairs)

        # 3. UPDATE & SORT
        for i, hit in enumerate(candidates):
            hit.score = float(scores[i]) # Overwrite the RRF score with the Re-Ranker score

        # Sort descending (Highest score first)
        candidates.sort(key=lambda x: x.score, reverse=True)

        # 4. RETURN TOP K
        return candidates[:limit]

# Simple test block
if __name__ == "__main__":
    try:
        db = VectorDBClient()
        print("🚀 Qdrant Connection Successful!")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")