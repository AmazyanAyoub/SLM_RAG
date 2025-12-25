import sys
import os
import time
import re
from pathlib import Path
import json
from datetime import datetime

# Add project root to path to ensure imports work
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.indexing.vector_store import VectorDBClient
from backend.core.config_loader import settings
from backend.indexing.postgres_client import PostgresVectorDB
from sentence_transformers import CrossEncoder


import psycopg2
from psycopg2 import sql



def clean_reasoning(text: str) -> str:
    """Removes <think> blocks and returns clean answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

queries = [
        # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---

        # "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
        # "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
        # "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
        # "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
        # "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60%|activity or more), what is the amount of the monthly income franchise (exemption) granted?",
        # "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
        # "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
        # "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
        # "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
        # "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I).",

        # # # # --- 10 Short Questions (5 Simple, 5 Hard) ---

        # # # # Simple
        # "According to Article 2, paragraph 1, what is the base monthly maintenance amount (prestation mensuelle de base) for a single person before any multiplication factor is applied?",
        # "Under Article 12, what is the maximum duration for which provisional financial aid (aide financière provisoire) can be granted?",
        # "According to Article 40, which entity is responsible for financing the furniture and current stewardship of social aid premises provided by communes?",
        # "What is the minimum validity duration of the control document established by the office for emergency aid applicants, according to Article 30?",
        # "According to Article 3, paragraph 3, how frequently is the payment of rent systematically controlled?",

        # # # Hard
        # "According to Article 9, paragraph 13, what is the maximum reimbursement amount for transport costs outside the canton related to the employment evaluation internship?",
        # "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
        # "According to Article 20, paragraph 4(e), what is the daily food allowance granted during leaves for a person staying in an establishment outside the canton of Geneva?",
        # "Based on Article 16, what is the standard maximum duration for ordinary financial aid granted to a person exercising an independent lucrative activity (excluding cases of incapacity)?",
        # "According to Article 5, paragraph 4, what is the maximum annual amount granted for household and family aid (aide ménagère et familiale)?",

        # # --- 10 Long Questions (5 Simple, 5 Hard) ---

        # # Simple
        # "List the four cumulative conditions required for a person to request aid for the management of periodic income under Article 23.",
        # "Describe the four specific items or services that constitute 'Prestations d'aide d'urgence' (emergency aid) provided in kind, as detailed in Article 24, paragraph 1.",
        # "According to Article 19, paragraph 2(a), list the specific monthly maintenance allowance amounts (forfait d'entretien) for exceptional financial aid for a household of 1, 2, 3, and 4 persons.",
        # "List the five cumulative conditions set out in Article 17A that a foreigner without a residence permit must meet to benefit from return aid (aide au retour).",
        # "According to Article 41, which specific regulation was abrogated by the entry into force of the current RIASI regulation?",

        # # Hard
        # "Explain the conditions and limits for the coverage of health insurance premiums under Article 4, specifically regarding the 'prime cantonale de référence' and the exception for new beneficiaries whose premiums exceed this reference.",
        # "Detail the specific categories of expenses related to children's activities that can be reimbursed under Article 9, paragraph 12, including the maximum annual amount for holiday camps (camps de vacances).",
        # "Compare the calculation of the monthly maintenance allowance for a beneficiary in a 'Communauté de majeurs' (Article 10) versus a beneficiary in 'Cohabitation' (Article 11) regarding how the base amount is determined.",
        # "Describe the process for the 'Allocation d'indépendant' under Article 23I, specifically focusing on the repayment conditions if the project is abandoned versus if the activity continues after 12 months.",
        # "According to Article 35, what are the specific financial consequences on the monthly maintenance allowance and situational benefits in cases of 'manquement aux devoirs' (breach of duties) versus 'manquement grave' (serious breach)?",


        # "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
        # "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
        # "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
        # "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
        # "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",

        "Calculate the monthly maintenance allowance for a household of 7 people using the base amount and the coefficients provided in Article 2 (including the add-on for people beyond 5).",
        "According to Article 5, calculate the monthly diet allocation (allocation de régime) for a household where 3 people require a medical diet, applying the coefficient from Article 2.",
        "Using the table in Article 3, calculate the maximum recognized rent for a family group consisting of a couple and 5 children.",
        "Calculate the total combined Moving and Installation allowances (Article 9) for a family of 4 people.",
        "Calculate the total monthly income franchise (Article 8) for a household where one person works at 100% |and another works at 50%.",
        
        "A beneficiary works full-time (100% |activity) for a whole year. According to Article 8, calculate the total annual amount of the income franchise they are entitled to.",
        "A single parent sends their 3 children to holiday camps during the summer. According to the limits set in Article 9, what is the maximum total reimbursement the family can receive for these camps in a single calendar year?",
        "Under Article 5, a parent requires childcare for 50 hours in a specific month to attend a professional insertion program. Using the hourly rate provided, calculate the total cost coverage for that month.",
        "A young adult (20 years old) student lives with their parents and is eligible for the integration supplement. According to Article 7, calculate the total amount of this supplement they generate for the family over a period of 6 months.",
        "A single person is receiving exceptional financial aid under Article 19. Calculate the total monthly sum of their maximum rent coverage limit plus their pocket money allowance.",

        "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
        "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
        "A beneficiary works at 80%| activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
        "A family of 5 people (Couple + 3 children) moves into a new apartment. They claim the maximum Moving Allowance and the maximum Installation Allowance under Article 9. Calculate the grand total of these one-off benefits.",
        "Calculate the difference in the Maximum Rent allowance (Article 3) between a 'Couple with 2 children' and a 'Couple with 4 children'. Return the difference as a positive number."
    ]

def merge_text_chunks(chunks, overlap_hint=200):
    """
    Merges a list of text chunks by removing the overlapping text at the boundaries.
    """
    if not chunks:
        return ""
    
    merged = chunks[0]
    
    for next_chunk in chunks[1:]:
        # We look for the overlap in the start of the next_chunk
        # We assume the overlap is roughly 'overlap_hint' size, but we scan a bit wider
        scan_len = min(len(next_chunk), overlap_hint + 50) 
        prefix = next_chunk[:scan_len]
        
        # We scan the end of the current 'merged' text for this prefix
        # We look for the longest matching suffix
        best_overlap_len = 0
        
        # We try to match from largest possible overlap down to a reasonable minimum (e.g. 10 chars)
        # to avoid false positives on small words like "the ".
        for i in range(scan_len, 10, -1):
            candidate = prefix[:i]
            if merged.endswith(candidate):
                best_overlap_len = i
                break
        
        if best_overlap_len > 0:
            # Append only the non-overlapping part
            merged += next_chunk[best_overlap_len:]
        else:
            # Fallback: Just join with a newline if no overlap found
            merged += "\n" + next_chunk
            
    return merged

def retrieve_window_context(db_client, query, top_k=3, window_size=2, reranker=CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=4096, device="cuda")):
    """
    FALLBACK METHOD: Uses Postgres Primary Key 'id' to find neighbors.
    Assumes chunks were inserted sequentially (ID 100 -> ID 101 -> ID 102).
    """
    print(f"   🛠️  Using HACKY ID-based Window Retrieval (Size={window_size})...")
    
    # 1. Standard Vector Search
    initial_results = db_client.search(query, limit=top_k)
    if not initial_results:
        return [], []

    # 2. Collect IDs to fetch
    # We need to map {Original_ID: [List of Neighbor IDs]}
    id_map = {} 
    all_ids_to_fetch = []

    for hit in initial_results:
        # We assume the Vector DB 'id' matches the Postgres Primary Key 'id'
        # If your vector store uses UUIDs but Postgres uses Ints, this might fail.
        try:
            center_id = int(hit.id) 
        except (TypeError, ValueError):
            print(f"   ⚠️ Skipping hit: ID '{hit.id}' is not an integer.")
            continue
            
        # Calculate the range: center_id to center_id + window_size
        neighbor_ids = list(range(center_id, center_id + window_size + 1))
        
        id_map[center_id] = neighbor_ids
        all_ids_to_fetch.extend(neighbor_ids)

    # Remove duplicates
    all_ids_to_fetch = list(set(all_ids_to_fetch))

    if not all_ids_to_fetch:
        print("   ❌ No valid integer IDs found in vector results.")
        return [], []

    # 3. Query Postgres for these IDs
    table_name = settings.retrieval.vector_store_collection # "slm_rag_contextual_V3"
    fetched_content_map = {} # ID -> Content

    query_sql = sql.SQL("SELECT id, content, metadata->>'source' FROM {} WHERE id IN ({})").format(
        sql.Identifier(table_name),
        sql.SQL(', ').join(sql.Placeholder() for _ in all_ids_to_fetch)
    )

    try:
        with psycopg2.connect(
            host=settings.retrieval.postgres_host,
            user=settings.retrieval.postgres_user,
            password=settings.retrieval.postgres_password or "mysecretpassword",
            dbname=settings.retrieval.postgres_db,
            port=settings.retrieval.postgres_port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, all_ids_to_fetch)
                rows = cur.fetchall()
                for r in rows:
                    # r[0]=id, r[1]=content, r[2]=source
                    fetched_content_map[r[0]] = (r[1], r[2])
                    
    except Exception as e:
        print(f"   ❌ DB Fetch Error: {e}")
        return [], []

    # 4. Reconstruct Contexts
    final_contexts = []
    final_sources = []

    for hit in initial_results:
        try:
            center_id = int(hit.id)
            target_ids = id_map.get(center_id, [])
            
            # Sort IDs to ensure text order is correct
            target_ids.sort()
            
            merged_text = []
            detected_source = "Unknown"
            
            for tid in target_ids:
                if tid in fetched_content_map:
                    content, source = fetched_content_map[tid]
                    merged_text.append(content)
                    if source: detected_source = source
            
            if merged_text:
                clean_text = merge_text_chunks(merged_text, overlap_hint=200)
                final_contexts.append(clean_text)
                final_sources.append(detected_source)
            else:
                # Fallback if ID fetch failed
                final_contexts.append(hit.payload.get("search_content", ""))
                final_sources.append(hit.payload.get("metadata", {}).get("source", "Unknown"))
                
        except:
            continue

    # 5. Re-Ranking (NEW)
    if reranker and final_contexts:
        print("   ⚖️  Re-ranking merged contexts...")
        start_rerank = time.time()
        
        # Prepare pairs for the Cross-Encoder: [Query, Document Text]
        pairs = [[query, doc_text] for doc_text in final_contexts]
        
        try:
            scores = reranker.predict(pairs)
            
            # Zip everything together: (Score, Context, Source)
            scored_results = list(zip(scores, final_contexts, final_sources))
            
            # Sort descending by Score
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Unzip back into lists
            final_contexts = [x[1] for x in scored_results]
            final_sources = [x[2] for x in scored_results]
            
            print(f"   📊 Re-ranking took {time.time() - start_rerank:.4f}s")
        except Exception as e:
            print(f"   ⚠️ Re-ranking failed: {e}. Returning original order.")

    return final_contexts, final_sources


def main():
    print("🚀 STARTING WINDOW RETRIEVAL INSPECTION (NO GENERATION)")
    print("==================================================")

    # 1. Setup Evaluation Directory
    LOG_DIR = Path("data/eval_results")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Vector DB
    try:
        db_client = VectorDBClient()
        print(f"✅ Connected to Provider: {db_client.provider.upper()}")
    except Exception as e:
        print(f"❌ Failed to connect to VectorDB: {e}")
        return

    # 3. Define Queries (Ensure your 'queries' list is defined)
    # queries = [...]

    results_log = []
    start_total = time.time()

    # 4. Execute Search Loop
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] ❓ Query: '{query}'")
        try:
            start_time = time.time()
            
            # --- USE WINDOW RETRIEVAL (Top 4 matches + 3 neighbors) ---
            # This captures the exact list of chunks the model WOULD receive
            context_blocks, source_names = retrieve_window_context(db_client, query, top_k=5, window_size=5)
            
            elapsed = time.time() - start_time
            print(f"⏱️  Window Retrieval: {elapsed:.4f}s | Context Blocks Built: {len(context_blocks)}")
            
            if not context_blocks:
                print("⚠️  No documents found.")
                
            # Log the exact text blocks for inspection
            retrieved_data = []
            for idx, (txt, src) in enumerate(zip(context_blocks, source_names)):
                retrieved_data.append({
                    "block_id": idx + 1,
                    "source": src,
                    "content": txt  # Saving the full text to check truthfulness
                })

            # D. LOG RESULT
            results_log.append({
                "id": i + 1,
                "question": query,
                "retrieval_method": "Window (Top 4 + Window 3)",
                "time_taken": round(elapsed, 4),
                "retrieved_contexts": retrieved_data # <--- This is what we check
            })

        except Exception as e:
            print(f"❌ Failed for query '{query}': {e}")
            results_log.append({
                "id": i + 1, 
                "question": query, 
                "error": str(e)
            })

    # 5. SAVE JSON
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"eval_window_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "mode": "Window Retrieval Inspection (No LLM)",
            "total_questions": len(queries),
            "total_time_seconds": round(total_time, 2)
        },
        "results": results_log
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

if __name__ == "__main__":
    main()