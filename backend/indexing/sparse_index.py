import uuid
import time
from typing import List, Dict, Any
from backend.indexing.vector_store import VectorDBClient
from backend.models.embedding_client import embed_sparse
from qdrant_client.http import models


class SparseIndexer:
    def __init__(self):
        """
        Initializes the Sparse Indexer.
        The Model is now handled by the shared embedding_client.
        """
        self.db_client = VectorDBClient()
        self.collection_name = self.db_client.collection_name

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        print(f"🚀 Generating Sparse Vectors for {len(chunks)} chunks...")
        start_time = time.time()

        # 1. Prepare Text (Use the 'search_content' which is the Enriched text)
        search_texts = [c.get("search_content", c["text"]) for c in chunks]
        
        # 2. Compute Vectors
        sparse_outputs = embed_sparse(search_texts)
        
        # 3. Prepare Batch Update for Qdrant
        # We use 'update_vectors' because we assume Dense vectors might already exist.
        # This prevents overwriting the Dense data.
        points_updates = []
        
        for i, chunk in enumerate(chunks):
            # Regenerate the Deterministic ID to find the correct point
            source = chunk["metadata"].get("source", "unknown")
            index = chunk["metadata"].get("chunk_index", i)
            signature = f"{source}_{index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, signature))
            
            # Convert BGE-M3 output to Qdrant Sparse Format
            # BGE-M3 output is {token_id: weight}
            sp_indices = list(int(k) for k in sparse_outputs[i].keys())
            sp_values = list(float(v) for v in sparse_outputs[i].values())

            points_updates.append(models.PointVectors(
                id=point_id,
                vector={
                    "sparse": models.SparseVector(
                        indices=sp_indices,
                        values=sp_values
                    )
                }
            ))

        # 4. Perform Update
        print(f"📤 Updating {len(points_updates)} points with sparse data...")
        
        # We use the client directly to call update_vectors
        self.db_client.client.update_vectors(
            collection_name=self.collection_name,
            points=points_updates
        )
        
        print(f"✅ Sparse Indexing Complete in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # Simple Test
    indexer = SparseIndexer()
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
    # Note: This test will fail if the collection doesn't exist or isn't configured for sparse yet.
    indexer.index_chunks(dummy_chunks)