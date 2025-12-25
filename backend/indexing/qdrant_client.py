from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.core.config_loader import settings
from sentence_transformers import CrossEncoder
from backend.models.embedding_client import embed_queries, embed_sparse

class QdrantVectorDB:
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
        self.collection_name = f"{settings.retrieval.vector_store_collection}_large"
        self.vector_size = 1024 
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device="cuda")

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"📦 Collection '{self.collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )
            print(f"✅ Collection '{self.collection_name}' created successfully.")
        else:
            print(f"✅ Connected to existing collection: '{self.collection_name}'")

    def upsert(self, points: list):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_text: str, limit: int = 5):
        # 0. GENERATE EMBEDDINGS
        query_dense = embed_queries([query_text])[0]
        sparse_output = embed_sparse([query_text])[0]
        query_sparse_indices = list(int(k) for k in sparse_output.keys())
        query_sparse_values = list(float(v) for v in sparse_output.values())

        # 1. RETRIEVE CANDIDATES
        initial_limit = 15 
        candidates = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=initial_limit),
                models.Prefetch(
                    query=models.SparseVector(indices=query_sparse_indices, values=query_sparse_values),
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

        # 2. RE-RANKING
        pairs = []
        for hit in candidates:
            doc_text = hit.payload.get("search_content")
            if not doc_text:
                summary = hit.payload.get("context_summary", "")
                raw_text = hit.payload.get("text", "")
                doc_text = f"{summary}\n{raw_text}" if summary else raw_text
            pairs.append([query_text, doc_text])

        scores = self.reranker.predict(pairs)
        for i, hit in enumerate(candidates):
            hit.score = float(scores[i])
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:limit]

if __name__ == "__main__":
    import uuid
    print("--- TEST: QdrantVectorDB ---")
    try:
        db = QdrantVectorDB()
        
        test_text = "Qdrant is a vector database."
        print(f"Generating embeddings for: '{test_text}'")
        
        dense_vec = embed_queries([test_text])[0]
        sparse_vec = embed_sparse([test_text])[0]
        
        # Create a Qdrant PointStruct
        point_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "sparse": models.SparseVector(
                    indices=list(int(k) for k in sparse_vec.keys()),
                    values=list(float(v) for v in sparse_vec.values())
                )
            },
            payload={"text": test_text, "search_content": test_text}
        )
        
        print("Upserting...")
        db.upsert([point])
        
        print("Searching...")
        results = db.search("vector database", limit=1)
        print(f"✅ Found: {len(results)} results.")
        if results:
            print(f"   Top result: {results[0].payload.get('text')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")