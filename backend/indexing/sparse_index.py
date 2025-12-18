import uuid
import time
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel
from backend.indexing.vector_store import VectorDBClient
from backend.core.config_loader import settings
from qdrant_client.http import models
from huggingface_hub import snapshot_download


class SparseIndexer:
    def __init__(self):
        """
        Initializes the Sparse Indexer using BGE-M3 (Sparse Mode Only).
        """
        model_name = settings.retrieval.embedder_model # e.g. BAAI/bge-m3
        print(f"🧠 Loading Sparse Model: {model_name}...")

        # 1. SMART DOWNLOAD: Download ONLY what we need (Skip the 2GB ONNX file)
        # This prevents the "Disk Full" error
        local_path = snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.onnx", "*.onnx_data", "flax_model.msgpack", "rust_model.ot"],
            resume_download=True
        )
        
        # We load BGE-M3 but we will ONLY ask for sparse vectors
        self.model = BGEM3FlagModel(model_name_or_path=local_path, use_fp16=False)
        self.db_client = VectorDBClient()
        self.collection_name = self.db_client.collection_name

    def compute_sparse_vectors(self, texts: List[str]) -> List[Any]:
        """
        Generates ONLY sparse vectors (lexical weights) for a list of texts.
        """
        output = self.model.encode(
            texts, 
            return_dense=False, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        # Returns a list of dictionaries: [{'word_id': weight}, ...]
        return output['lexical_weights']

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        print(f"🚀 Generating Sparse Vectors for {len(chunks)} chunks...")
        start_time = time.time()

        # 1. Prepare Text (Use the 'search_content' which is the Enriched text)
        search_texts = [c.get("search_content", c["text"]) for c in chunks]
        
        # 2. Compute Vectors
        sparse_outputs = self.compute_sparse_vectors(search_texts)
        
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