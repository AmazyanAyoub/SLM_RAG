import sys
import os
import time
import re
from pathlib import Path
import json
from datetime import datetime

# Add project root to path to ensure imports work
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.indexing.vector_store import VectorDBClient
from langchain_core.output_parsers import StrOutputParser
from backend.indexing.vector_store import VectorDBClient
from backend.core.config_loader import settings
from sentence_transformers import CrossEncoder

import psycopg2
from psycopg2 import sql


def clean_reasoning(text: str) -> str:
    """Removes <think> blocks and returns clean answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

queries = [
        # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---

        "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
        "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
        "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
        "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
        "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60%|activity or more), what is the amount of the monthly income franchise (exemption) granted?",
        "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
        "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
        "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
        "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
        "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I).",

        # # # --- 10 Short Questions (5 Simple, 5 Hard) ---

        # # # Simple
        "According to Article 2, paragraph 1, what is the base monthly maintenance amount (prestation mensuelle de base) for a single person before any multiplication factor is applied?",
        "Under Article 12, what is the maximum duration for which provisional financial aid (aide financière provisoire) can be granted?",
        "According to Article 40, which entity is responsible for financing the furniture and current stewardship of social aid premises provided by communes?",
        "What is the minimum validity duration of the control document established by the office for emergency aid applicants, according to Article 30?",
        "According to Article 3, paragraph 3, how frequently is the payment of rent systematically controlled?",

        # # Hard
        "According to Article 9, paragraph 13, what is the maximum reimbursement amount for transport costs outside the canton related to the employment evaluation internship?",
        "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
        "According to Article 20, paragraph 4(e), what is the daily food allowance granted during leaves for a person staying in an establishment outside the canton of Geneva?",
        "Based on Article 16, what is the standard maximum duration for ordinary financial aid granted to a person exercising an independent lucrative activity (excluding cases of incapacity)?",
        "According to Article 5, paragraph 4, what is the maximum annual amount granted for household and family aid (aide ménagère et familiale)?",

        # --- 10 Long Questions (5 Simple, 5 Hard) ---

        # Simple
        "List the four cumulative conditions required for a person to request aid for the management of periodic income under Article 23.",
        "Describe the four specific items or services that constitute 'Prestations d'aide d'urgence' (emergency aid) provided in kind, as detailed in Article 24, paragraph 1.",
        "According to Article 19, paragraph 2(a), list the specific monthly maintenance allowance amounts (forfait d'entretien) for exceptional financial aid for a household of 1, 2, 3, and 4 persons.",
        "List the five cumulative conditions set out in Article 17A that a foreigner without a residence permit must meet to benefit from return aid (aide au retour).",
        "According to Article 41, which specific regulation was abrogated by the entry into force of the current RIASI regulation?",

        # Hard
        "Explain the conditions and limits for the coverage of health insurance premiums under Article 4, specifically regarding the 'prime cantonale de référence' and the exception for new beneficiaries whose premiums exceed this reference.",
        "Detail the specific categories of expenses related to children's activities that can be reimbursed under Article 9, paragraph 12, including the maximum annual amount for holiday camps (camps de vacances).",
        "Compare the calculation of the monthly maintenance allowance for a beneficiary in a 'Communauté de majeurs' (Article 10) versus a beneficiary in 'Cohabitation' (Article 11) regarding how the base amount is determined.",
        "Describe the process for the 'Allocation d'indépendant' under Article 23I, specifically focusing on the repayment conditions if the project is abandoned versus if the activity continues after 12 months.",
        "According to Article 35, what are the specific financial consequences on the monthly maintenance allowance and situational benefits in cases of 'manquement aux devoirs' (breach of duties) versus 'manquement grave' (serious breach)?",


        "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
        "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
        "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
        "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
        "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",

        # "Calculate the monthly maintenance allowance for a household of 7 people using the base amount and the coefficients provided in Article 2 (including the add-on for people beyond 5).",
        # "According to Article 5, calculate the monthly diet allocation (allocation de régime) for a household where 3 people require a medical diet, applying the coefficient from Article 2.",
        # "Using the table in Article 3, calculate the maximum recognized rent for a family group consisting of a couple and 5 children.",
        # "Calculate the total combined Moving and Installation allowances (Article 9) for a family of 4 people.",
        # "Calculate the total monthly income franchise (Article 8) for a household where one person works at 100% |and another works at 50%.",
        
        # "A beneficiary works full-time (100% |activity) for a whole year. According to Article 8, calculate the total annual amount of the income franchise they are entitled to.",
        # "A single parent sends their 3 children to holiday camps during the summer. According to the limits set in Article 9, what is the maximum total reimbursement the family can receive for these camps in a single calendar year?",
        # "Under Article 5, a parent requires childcare for 50 hours in a specific month to attend a professional insertion program. Using the hourly rate provided, calculate the total cost coverage for that month.",
        # "A young adult (20 years old) student lives with their parents and is eligible for the integration supplement. According to Article 7, calculate the total amount of this supplement they generate for the family over a period of 6 months.",
        # "A single person is receiving exceptional financial aid under Article 19. Calculate the total monthly sum of their maximum rent coverage limit plus their pocket money allowance.",

        # "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
        # "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
        # "A beneficiary works at 80%| activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
        # "A family of 5 people (Couple + 3 children) moves into a new apartment. They claim the maximum Moving Allowance and the maximum Installation Allowance under Article 9. Calculate the grand total of these one-off benefits.",
        # "Calculate the difference in the Maximum Rent allowance (Article 3) between a 'Couple with 2 children' and a 'Couple with 4 children'. Return the difference as a positive number."
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
    print("🚀 STARTING VECTOR SEARCH + GENERATION TEST")
    print("==================================================")

    LOG_DIR = Path("data/gen_results")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Vector DB
    # The factory will automatically pick 'qdrant' or 'postgres' based on 
    # the VECTOR_DB_PROVIDER environment variable or default.
    try:
        db_client = VectorDBClient()
        print(f"✅ Connected to Provider: {db_client.provider.upper()}")
    except Exception as e:
        print(f"❌ Failed to connect to VectorDB: {e}")
        return

    # 2. Initialize Ollama (Generation)
    print("⚙️ Initializing Ollama (qwen3:8b)...")
    try:
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=8192,
            seed=42,
            top_p=0.9,
            top_k=40,
            keep_alive="30m"
        )
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        return

    # 3. Define Prompt Template
# 3. Define Prompt Template (STRICT VERSION)
    # prompt = ChatPromptTemplate.from_template(
    #     """
    #     You are a strict data extraction assistant. Your ONLY job is to extract facts from the provided text.

    #     <context>
    #     {context}
    #     </context>

    #     Question: {question}

    #     **STRICT RULES:**
    #     1. **Evidence First:** You must find the EXACT sentence or table row in the <context> that answers the question.
    #     2. **No Context = No Answer:** If the specific number, rate, or condition asked for is NOT written in the <context>, you must output exactly: "I do not know."
    #     3. **Do Not Guess:** Do not try to derive, estimate, or use outside knowledge. If the text cuts off before the answer, say "I do not know."
    #     4. **Do Not Multiply:** If a number is given for a "family of 4", copy that number directly. Do not multiply it.

    #     **OUTPUT FORMAT:**
    #     - <think> First, copy the exact quote from the text that contains the answer. If you cannot find a quote, write "No quote found." </think>
    #     - Then, provide the final answer.
    #     """ 
    # )

    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
            """
            You are a precise and helpful document assistant.

            ### CORE INSTRUCTIONS:
            1. **Answer strictly** based on the provided **Context Blocks** below. Do not use outside knowledge.
            2. **HANDLE SPLIT TEXT (CRITICAL):**
            - The context is provided as a sequence of continuous text chunks (Block 1, Block 2, etc.).
            - **Text may be cut off:** A sentence, list, or paragraph ending in Block N often continues immediately in Block N+1.
            - **Stitch mentally:** If Block 1 ends abruptly (e.g., with a colon `:`, a hyphen `-`, or mid-sentence), read the start of Block 2 to complete the thought.
            - Treat the blocks as a single continuous document, not separate snippets.
            3. **Synthesis:** If the answer requires combining facts from Block 1 and Block 3, merge them into a coherent response.
            4. **Fallback:** If the answer is not found in the context, state clearly: "The provided documents do not contain this information."

            ### CONTEXT BLOCKS:
            {context}

            ### USER QUESTION:
            {question}

            ### ANSWER:
        """
    )
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()

    # 4. Define Queries


    # 5. Execute Search & Generation Loop
#     results_log = []
#     start_total = time.time()

#     for i, query in enumerate(queries):
#         print(f"\n[{i+1}/{len(queries)}] ❓ Query: '{query}'")
#         try:
#             start_time = time.time()
            
#             # A. RETRIEVE
#             results = db_client.search(query, limit=7)
#             elapsed = time.time() - start_time
            
#             print(f"⏱️  Retrieval: {elapsed:.4f}s | Found: {len(results)} docs")
            
#             if not results:
#                 print("⚠️  No documents found.")
#                 results_log.append({
#                     "id": i + 1, "question": query, "cleaned_answer": "I don't know (No documents found).",
#                     "raw_response_snippet": "", "sources": [], "retrieval_score_top1": 0,
#                     "time_taken": round(time.time() - start_time, 2)
#                 })
#                 continue

#             # B. FORMAT CONTEXT
#             context_text = ""
#             sources = []
#             print(f"📄 Top Sources:")
            
#             for j, hit in enumerate(results):
#                 # Extract Payload Data
#                 payload = hit.payload
                
#                 # Handle different schema structures (Qdrant vs Postgres ingestions)
#                 # 1. Content
#                 content = payload.get("search_content") or payload.get("text", "")
                
#                 # 2. Source
#                 # Check top-level (Qdrant style) or nested metadata (Postgres style)
#                 meta = payload.get("metadata", {})
#                 if isinstance(meta, dict) and "source" in meta:
#                     source = meta["source"]
#                 else:
#                     source = payload.get("source", "Unknown")
#                 sources.append(source)
                    
#                 print(f"   {j+1}. {source} (Score: {hit.score:.4f})")
#                 context_text += f"\n---\nSource: {source}\n{content}\n"

#             # C. GENERATE
#             print("🤖 Generating Answer...", end="", flush=True)
#             gen_start = time.time()
            
#             raw_response = chain.invoke({"context": context_text, "question": query})
#             final_answer = clean_reasoning(raw_response)
            
#             gen_time = time.time() - gen_start
#             print(f" Done in {gen_time:.2f}s")
            
#             print(f"\n💡 ANSWER:\n{final_answer}")
#             print("-" * 60)

#             # D. LOG RESULT
#             results_log.append({
#                 "id": i + 1,
#                 "question": query,
#                 "cleaned_answer": final_answer,
#                 "raw_response_snippet": raw_response[:200] + "...",
#                 "sources": list(set(sources)),
#                 "retrieval_score_top1": results[0].score if results else 0,
#                 "time_taken": round(time.time() - start_time, 2)
#             })

#         except Exception as e:
#             print(f"❌ Failed for query '{query}': {e}")
#             results_log.append({
#                 "id": i + 1, "question": query, "cleaned_answer": f"Error: {str(e)}",
#                 "raw_response_snippet": "", "sources": [], "retrieval_score_top1": 0,
#                 "time_taken": round(time.time() - start_time, 2)
#             })

#     # 6. SAVE JSON
#     total_time = time.time() - start_total
#     print("\n" + "=" * 60)
    
#     output_file = LOG_DIR / f"simple_search_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
#     final_report = {
#         "meta": {
#             "timestamp": datetime.now().isoformat(),
#             "model_name": "qwen3:8b",
#             "mode": f"Simple Search ({db_client.provider.upper()})",
#             "total_questions": len(queries),
#             "total_time_seconds": round(total_time, 2)
#         },
#         "results": results_log
#     }

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(final_report, f, indent=4, ensure_ascii=False)

#     print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

# if __name__ == "__main__":
#     main()



    results_log = []
    start_total = time.time()

    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] ❓ Query: '{query}'")
        try:
            start_time = time.time()
            
            # --- USE WINDOW RETRIEVAL (Top 3 matches + next 2 neighbors) ---
            context_blocks, source_names = retrieve_window_context(db_client, query, top_k=5, window_size=3)
            
            elapsed = time.time() - start_time
            print(f"⏱️  Window Retrieval: {elapsed:.4f}s | Contexts Built: {len(context_blocks)}")
            
            if not context_blocks:
                print("⚠️  No documents found.")
                continue

            # B. FORMAT CONTEXT
            context_text = ""
            for idx, (txt, src) in enumerate(zip(context_blocks, source_names)):
                context_text += f"\n---\nSource: {src}\n{txt}\n"

            # C. GENERATE
            print("🤖 Generating Answer...", end="", flush=True)
            gen_start = time.time()
            
            raw_response = chain.invoke({"context": context_text, "question": query})
            final_answer = clean_reasoning(raw_response)
            
            gen_time = time.time() - gen_start
            print(f" Done in {gen_time:.2f}s")
            
            print(f"\n💡 ANSWER:\n{final_answer}")
            print("-" * 60)

            # D. LOG RESULT
            results_log.append({
                "id": i + 1,
                "question": query,
                "cleaned_answer": final_answer,
                "raw_response_snippet": raw_response[:200] + "...",
                "sources": list(set(source_names)),
                "time_taken": round(time.time() - start_time, 2)
            })

        except Exception as e:
            print(f"❌ Failed for query '{query}': {e}")
            results_log.append({
                "id": i + 1, "question": query, "cleaned_answer": f"Error: {str(e)}",
                "raw_response_snippet": "", "sources": [], "retrieval_score_top1": 0,
                "time_taken": round(time.time() - start_time, 2)
            })

    # 6. SAVE JSON
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"window_search_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model_name": "qwen3:8b",
            "mode": f"Window Retrieval (Metadata) + Universal Prompt",
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

# import sys
# import os
# import time
# import re
# from pathlib import Path
# import json
# from datetime import datetime
# import psycopg2
# from psycopg2 import sql

# # Add project root to path to ensure imports work
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from backend.indexing.vector_store import VectorDBClient
# from backend.core.config_loader import settings 

# # ==========================================
# # 🛠️ HELPER: WINDOW RETRIEVAL LOGIC
# # ==========================================

# def get_direct_db_connection():
#     """Opens a direct connection to Postgres for fetching neighbor chunks."""
#     return psycopg2.connect(
#         host=settings.retrieval.postgres_host,
#         user=settings.retrieval.postgres_user,
#         password=settings.retrieval.postgres_password,
#         dbname=settings.retrieval.postgres_db,
#         port=settings.retrieval.postgres_port
#     )

# def retrieve_window_context(db_client, query, top_k=3, window_size=2):
#     """
#     Performs vector search, then fetches neighbor chunks (next N IDs) 
#     to fix 'cut-off' tables and lists.
#     """
#     # 1. Standard Vector Search
#     initial_results = db_client.search(query, limit=top_k)
#     if not initial_results:
#         return [], []

#     # 2. Calculate IDs to fetch (Matches + Neighbors)
#     ids_to_fetch = set()
#     for hit in initial_results:
#         ids_to_fetch.add(hit.id)
#         # Fetch NEXT 'window_size' chunks (e.g., ID+1, ID+2)
#         # We assume IDs are sequential (bigserial)
#         for i in range(1, window_size + 1):
#             ids_to_fetch.add(hit.id + i)

#     # 3. Fetch Content from DB
#     table_name = settings.retrieval.vector_store_collection
#     fetched_rows = {}
    
#     try:
#         with get_direct_db_connection() as conn:
#             with conn.cursor() as cur:
#                 # Select ID, Content, and Source Metadata
#                 # Note: Adjust 'metadata' key access based on your DB schema if needed
#                 query_sql = sql.SQL("SELECT id, content, metadata FROM {table} WHERE id IN %s").format(
#                     table=sql.Identifier(table_name)
#                 )
#                 cur.execute(query_sql, (tuple(ids_to_fetch),))
#                 rows = cur.fetchall()
                
#                 for r in rows:
#                     fetched_rows[r[0]] = {
#                         "id": r[0],
#                         "content": r[1],
#                         "metadata": r[2] if r[2] else {}
#                     }
#     except Exception as e:
#         print(f"⚠️ Window fetch failed: {e}. Falling back to standard results.")
#         # Fallback: Just use vector results if DB fetch fails
#         contexts = [h.payload.get("search_content") or h.payload.get("text", "") for h in initial_results]
#         sources = [h.payload.get("metadata", {}).get("source", "Unknown") for h in initial_results]
#         return contexts, sources

#     # 4. Merge Sequential Chunks
#     # Group IDs: [100, 101, 102] -> One Context Block
#     sorted_ids = sorted(list(fetched_rows.keys()))
#     if not sorted_ids:
#         return [], []

#     groups = []
#     current_group = [sorted_ids[0]]
    
#     for i in range(1, len(sorted_ids)):
#         prev_id = sorted_ids[i-1]
#         curr_id = sorted_ids[i]
        
#         # Logic: If ID is sequential AND Source is the same -> Merge
#         prev_source = fetched_rows[prev_id]["metadata"].get("source", "Unknown")
#         curr_source = fetched_rows[curr_id]["metadata"].get("source", "Unknown")

#         if (curr_id == prev_id + 1) and (prev_source == curr_source):
#             current_group.append(curr_id)
#         else:
#             groups.append(current_group)
#             current_group = [curr_id]
#     groups.append(current_group)

#     # 5. Format Output
#     final_contexts = []
#     final_sources = []

#     for group in groups:
#         # Join text with newlines
#         merged_text = "\n".join([fetched_rows[gid]["content"] for gid in group])
#         # Get source from first chunk
#         source = fetched_rows[group[0]]["metadata"].get("source", "Unknown")
        
#         final_contexts.append(merged_text)
#         final_sources.append(source)

#     return final_contexts, final_sources

# # ==========================================
# # 🚀 MAIN SCRIPT
# # ==========================================

# def clean_reasoning(text: str) -> str:
#     """Removes <think> blocks and returns clean answer."""
#     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#     return cleaned.strip()

# def main():
#     print("🚀 STARTING VECTOR SEARCH + WINDOW RETRIEVAL + GENERATION")
#     print("=========================================================")

#     LOG_DIR = Path("data/gen_results")
#     LOG_DIR.mkdir(parents=True, exist_ok=True)

#     # 1. Initialize Vector DB
#     try:
#         db_client = VectorDBClient()
#         print(f"✅ Connected to Provider: {db_client.provider.upper()}")
#     except Exception as e:
#         print(f"❌ Failed to connect to VectorDB: {e}")
#         return

#     # 2. Initialize Ollama
#     print("⚙️ Initializing Ollama (qwen3:8b)...")
#     try:
#         llm = ChatOllama(
#             model="qwen3:8b",
#             base_url="http://localhost:11434",
#             temperature=0.1,
#             num_ctx=8192, 
#             keep_alive="30m"
#         )
#     except Exception as e:
#         print(f"❌ Failed to connect to Ollama: {e}")
#         return

#     # 3. UNIVERSAL LOGIC PROMPT
#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are an advanced analytical AI assistant. Your task is to answer the user's question using ONLY the provided context.

#         <context>
#         {context}
#         </context>

#         Question: {question}

#         **CORE OPERATING RULES:**
#         1. **Strict Context Adherence:** Use ONLY the information provided in the <context> tags. If the answer is not explicitly present or derivable from the context, state "I do not know."
#         2. **Citation:** Cite the specific section, table, or source title for every key fact used.
#         3. **Zero Hallucination:** Do not import outside knowledge or make up values.

#         **UNIVERSAL LOGIC & CALCULATION GUIDELINES:**
        
#         1. **"Unit vs. Group" Logic:**
#         - If the text provides a specific value, multiplier, or coefficient for a **defined group size, category, or tier** (e.g., "Family of 4", "Level 3 User"), use that specific value directly.
#         - **DO NOT** multiply a group-specific rate by the number of units in that group unless the text explicitly states "per unit" (e.g., "per person") *in addition* to the group rate.
#         - *Rule:* If a table entry already accounts for the group size (e.g., a coefficient), do not multiply by the count again.

#         2. **Range & Threshold Logic:**
#         - If a rule applies to values **"greater than" or "equal to"** a limit (≥ X), apply that rule to **ANY** specific value that meets the condition.
#         - *Example:* If a rule says "Orders > $100 get free shipping," an order of $500 also gets free shipping. Do not reject it because the specific number "$500" isn't listed.

#         3. **Dependency & Cross-Reference Logic:**
#         - If a section refers to another part of the document for specific details (e.g., "see Table A for rates"), actively check if that referenced content exists elsewhere in your provided context (it may have been retrieved in a nearby chunk).

#         **OUTPUT FORMAT:**
#         - For calculations, you MUST enclose your step-by-step logic in <think> tags.
#         - Explicitly state the formula you are extracting from the text before inserting numbers.
#         - The final part of your message must be the clear, direct answer.
#         """
#     )

#     chain = prompt | llm | StrOutputParser()

#     # 4. Queries (Short + Hard + Calculations)
#     queries = [
#         # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---

#         "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
#         "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
#         "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
#         "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
#         "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60%|activity or more), what is the amount of the monthly income franchise (exemption) granted?",
#         "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
#         "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
#         "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
#         "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
#         "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I).",

#         # # # --- 10 Short Questions (5 Simple, 5 Hard) ---

#         # # # Simple
#         "According to Article 2, paragraph 1, what is the base monthly maintenance amount (prestation mensuelle de base) for a single person before any multiplication factor is applied?",
#         "Under Article 12, what is the maximum duration for which provisional financial aid (aide financière provisoire) can be granted?",
#         "According to Article 40, which entity is responsible for financing the furniture and current stewardship of social aid premises provided by communes?",
#         "What is the minimum validity duration of the control document established by the office for emergency aid applicants, according to Article 30?",
#         "According to Article 3, paragraph 3, how frequently is the payment of rent systematically controlled?",

#         # Hard
#         "According to Article 9, paragraph 13, what is the maximum reimbursement amount for transport costs outside the canton related to the employment evaluation internship?",
#         "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
#         "According to Article 20, paragraph 4(e), what is the daily food allowance granted during leaves for a person staying in an establishment outside the canton of Geneva?",
#         "Based on Article 16, what is the standard maximum duration for ordinary financial aid granted to a person exercising an independent lucrative activity (excluding cases of incapacity)?",
#         "According to Article 5, paragraph 4, what is the maximum annual amount granted for household and family aid (aide ménagère et familiale)?",

#         # # --- 10 Long Questions (5 Simple, 5 Hard) ---

#         # # Simple
#         "List the four cumulative conditions required for a person to request aid for the management of periodic income under Article 23.",
#         "Describe the four specific items or services that constitute 'Prestations d'aide d'urgence' (emergency aid) provided in kind, as detailed in Article 24, paragraph 1.",
#         "According to Article 19, paragraph 2(a), list the specific monthly maintenance allowance amounts (forfait d'entretien) for exceptional financial aid for a household of 1, 2, 3, and 4 persons.",
#         "List the five cumulative conditions set out in Article 17A that a foreigner without a residence permit must meet to benefit from return aid (aide au retour).",
#         "According to Article 41, which specific regulation was abrogated by the entry into force of the current RIASI regulation?",

#         # Hard
#         "Explain the conditions and limits for the coverage of health insurance premiums under Article 4, specifically regarding the 'prime cantonale de référence' and the exception for new beneficiaries whose premiums exceed this reference.",
#         "Detail the specific categories of expenses related to children's activities that can be reimbursed under Article 9, paragraph 12, including the maximum annual amount for holiday camps (camps de vacances).",
#         "Compare the calculation of the monthly maintenance allowance for a beneficiary in a 'Communauté de majeurs' (Article 10) versus a beneficiary in 'Cohabitation' (Article 11) regarding how the base amount is determined.",
#         "Describe the process for the 'Allocation d'indépendant' under Article 23I, specifically focusing on the repayment conditions if the project is abandoned versus if the activity continues after 12 months.",
#         "According to Article 35, what are the specific financial consequences on the monthly maintenance allowance and situational benefits in cases of 'manquement aux devoirs' (breach of duties) versus 'manquement grave' (serious breach)?",


#         "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
#         "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
#         "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
#         "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
#         "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",

#         # "Calculate the monthly maintenance allowance for a household of 7 people using the base amount and the coefficients provided in Article 2 (including the add-on for people beyond 5).",
#         # "According to Article 5, calculate the monthly diet allocation (allocation de régime) for a household where 3 people require a medical diet, applying the coefficient from Article 2.",
#         # "Using the table in Article 3, calculate the maximum recognized rent for a family group consisting of a couple and 5 children.",
#         # "Calculate the total combined Moving and Installation allowances (Article 9) for a family of 4 people.",
#         # "Calculate the total monthly income franchise (Article 8) for a household where one person works at 100% |and another works at 50%.",
        
#         # "A beneficiary works full-time (100% |activity) for a whole year. According to Article 8, calculate the total annual amount of the income franchise they are entitled to.",
#         # "A single parent sends their 3 children to holiday camps during the summer. According to the limits set in Article 9, what is the maximum total reimbursement the family can receive for these camps in a single calendar year?",
#         # "Under Article 5, a parent requires childcare for 50 hours in a specific month to attend a professional insertion program. Using the hourly rate provided, calculate the total cost coverage for that month.",
#         # "A young adult (20 years old) student lives with their parents and is eligible for the integration supplement. According to Article 7, calculate the total amount of this supplement they generate for the family over a period of 6 months.",
#         # "A single person is receiving exceptional financial aid under Article 19. Calculate the total monthly sum of their maximum rent coverage limit plus their pocket money allowance.",

#         # # "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
#         # "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
#         # "A beneficiary works at 80%| activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
#         # "A family of 5 people (Couple + 3 children) moves into a new apartment. They claim the maximum Moving Allowance and the maximum Installation Allowance under Article 9. Calculate the grand total of these one-off benefits.",
#         # "Calculate the difference in the Maximum Rent allowance (Article 3) between a 'Couple with 2 children' and a 'Couple with 4 children'. Return the difference as a positive number."
#     ]

#     # 5. Execute Loop
#     results_log = []
#     start_total = time.time()

#     for i, query in enumerate(queries):
#         print(f"\n[{i+1}/{len(queries)}] ❓ Query: '{query}'")
#         try:
#             start_time = time.time()
            
#             # --- UPDATED RETRIEVAL: Window Search ---
#             # Fetch Top 3 matches + their Next 2 neighbors (Chunks 1024, Overlap 200)
#             context_blocks, source_names = retrieve_window_context(db_client, query, top_k=3, window_size=2)
            
#             elapsed = time.time() - start_time
#             print(f"⏱️  Window Retrieval: {elapsed:.4f}s | Context Blocks: {len(context_blocks)}")
            
#             if not context_blocks:
#                 print("⚠️  No documents found.")
#                 continue

#             # Format Context
#             formatted_context = ""
#             for idx, (txt, src) in enumerate(zip(context_blocks, source_names)):
#                 formatted_context += f"\n---\nSource: {src}\n{txt}\n"

#             # Generate
#             print("🤖 Generating Answer...", end="", flush=True)
#             gen_start = time.time()
            
#             raw_response = chain.invoke({"context": formatted_context, "question": query})
#             final_answer = clean_reasoning(raw_response)
            
#             gen_time = time.time() - gen_start
#             print(f" Done in {gen_time:.2f}s")
#             print(f"\n💡 ANSWER:\n{final_answer}")
#             print("-" * 60)

#             # Log
#             results_log.append({
#                 "id": i + 1,
#                 "question": query,
#                 "cleaned_answer": final_answer,
#                 "raw_response_snippet": raw_response[:200] + "...",
#                 "sources": list(set(source_names)),
#                 "time_taken": round(time.time() - start_time, 2)
#             })

#         except Exception as e:
#             print(f"❌ Failed for query '{query}': {e}")
#             results_log.append({"id": i+1, "question": query, "cleaned_answer": f"Error: {e}"})

#     # 6. Save
#     output_file = LOG_DIR / f"window_search_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     final_report = {
#         "meta": {"timestamp": datetime.now().isoformat(), "mode": "Window Retrieval + Universal Prompt"},
#         "results": results_log
#     }
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(final_report, f, indent=4, ensure_ascii=False)
#     print(f"💾 Saved to: {output_file}")

# if __name__ == "__main__":
#     main()