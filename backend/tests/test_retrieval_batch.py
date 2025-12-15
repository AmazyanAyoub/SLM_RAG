import os
import json
import time
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.indexing.dense_index import DenseIndexer
from backend.indexing.sparse_index import SparseIndexer # <--- NEW
from backend.indexing.vector_store import VectorDBClient

# ==========================================
# 📝 DEFINE YOUR TEST QUESTIONS HERE
# ==========================================
TEST_QUESTIONS = [
    "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
    "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
    "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
    "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
    "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60% activity or more), what is the amount of the monthly income franchise (exemption) granted?",
    "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
    "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
    "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
    "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
    "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I)."
]

def run_batch_test():
    print("🧪 STARTING HYBRID RETRIEVAL TEST (10 Questions)")
    print("============================================================")

    # 1. Initialize System (Both Indexers)
    try:
        dense_indexer = DenseIndexer()
        sparse_indexer = SparseIndexer()
        client = VectorDBClient()
        print("✅ Hybrid System Initialized.")
    except Exception as e:
        print(f"❌ System Init Failed: {e}")
        return

    results_log = []
    total_score = 0
    total_retrieved = 0

    # 2. Loop Through Questions
    start_time = time.time()
    
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"🔍 [{i+1}/10] Querying: {question[:50]}...", end="\r")
        
        # A. Generate Dense Vector (Meaning)
        query_dense = dense_indexer.embed_texts([question])[0]
        
        # B. Generate Sparse Vector (Keywords)
        # Note: sparse_indexer returns a list of dicts, we need the first one
        query_sparse_output = sparse_indexer.compute_sparse_vectors([question])[0]
        query_sparse_indices = [int(k) for k in query_sparse_output.keys()]
        query_sparse_values = [float(v) for v in query_sparse_output.values()]
        
        # C. Hybrid Search (Dense + Sparse)
        search_hits = client.search(
            query_dense=query_dense,
            query_sparse_indices=query_sparse_indices,
            query_sparse_values=query_sparse_values,
            limit=5
        )
        
        # Format Hits for JSON
        hits_data = []
        for hit in search_hits:
            hits_data.append({
                "score": float(hit.score), # This is now an RRF Score (Fusion), not just Cosine
                "source": hit.payload.get("source", "Unknown"),
                "text_preview": hit.payload.get("text", ""), # First 300 chars
                "search_content": hit.payload.get("context_summary", ""), # The Smart Context
                "chunk_index": hit.payload.get("chunk_index")
            })
            
            # Track stats
            total_score += hit.score
            total_retrieved += 1

        # Add to Log
        results_log.append({
            "question_id": i + 1,
            "question": question,
            "top_match_score": hits_data[0]["score"] if hits_data else 0,
            "retrieved_chunks": hits_data
        })

    print(f"\n✅ Completed 10 queries in {time.time() - start_time:.2f} seconds.")

    # 3. Save to JSON
    output_dir = "data/eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/hybrid_retrieval_benchmark_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "timestamp": timestamp,
                "mode": "Hybrid (Dense + Sparse)",
                "avg_score": total_score / total_retrieved if total_retrieved else 0
            },
            "results": results_log
        }, f, indent=4, ensure_ascii=False)

    print(f"📂 Results saved to: {filename}")
    
    # 4. Print Summary
    print("-" * 60)
    print("📊 HYBRID SEARCH SUMMARY")
    print("Note: Scores are now RRF (Rank Fusion), so they might look lower/different than raw Cosine.")
    print("Check the JSON file to see if the retrieved text actually answers the question!")
    print("-" * 60)

if __name__ == "__main__":
    run_batch_test()