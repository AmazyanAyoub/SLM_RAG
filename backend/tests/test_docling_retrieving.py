# import time
# import json
# import re
# import os
# from pathlib import Path
# from datetime import datetime

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from qdrant_client.http import models # Needed for filtering

# from backend.indexing.vector_store import VectorDBClient

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# # OLLAMA_BASE_URL = "http://18.132.143.112:14528"
# # MODEL_NAME = "qwen3:4b-instruct-2507-fp16" 
# MODEL_NAME = "qwen3:8b" 


# # The 10 Specific Benchmark Questions
# queries = [
#         # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---
#         # "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
#         # "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
#         # "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
#         # "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
#         # "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60%|activity or more), what is the amount of the monthly income franchise (exemption) granted?",
#         # "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
#         # "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
#         # "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
#         # "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
#         # "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I)."
#         "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
#         "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
#         "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
#         "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
#         "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",
# ]

# LOG_DIR = Path("data/gen_results")
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# def clean_reasoning(text: str) -> str:
#     """Removes <think> blocks and returns clean answer."""
#     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#     return cleaned.strip()

# def fetch_full_section(client, collection_name, hierarchy_path):
#     """
#     Retrieves ALL chunks that share the specific hierarchy_path
#     and merges them back into one complete text block.
#     """
#     if not hierarchy_path:
#         return None

#     # Scroll through Qdrant to find all chunks with this path
#     # We use scroll to ensure we get them all (though usually < 10)
#     response, _ = client.scroll(
#         collection_name=collection_name,
#         scroll_filter=models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key="metadata.hierarchy_path",
#                     match=models.MatchValue(value=hierarchy_path)
#                 )
#             ]
#         ),
#         limit=50, # Assumption: No single section is split into > 50 chunks
#         with_payload=True,
#         with_vectors=False
#     )
    
#     if not response:
#         return None

#     # Sort by chunk_index to ensure correct reading order
#     # Default to 0 if key is missing
#     sorted_chunks = sorted(response, key=lambda x: x.payload.get("chunk_index", 0))
    
#     # Merge Texts
#     full_text = "\n".join([c.payload.get("text", "") for c in sorted_chunks])
    
#     return full_text

# def run_benchmark():
#     print(f"\n🧪 STARTING HYBRID BENCHMARK (Model: {MODEL_NAME})")
#     print(f"📊 Total Questions: {len(queries)}")
#     print("============================================================")

#     # 1. SETUP (Run Once)
#     print("⚙️ Initializing Hybrid RAG components...")
    
#     # Force Qdrant
#     os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
#     print("👉 Forcing Provider: QDRANT")

#     try:
#         # Initialize Wrapper
#         db_wrapper = VectorDBClient()
        
#         # ACCESS INTERNAL CLIENTS (Need direct access for the 'scroll' operation)
#         # Structure: VectorDBClient -> QdrantVectorDB -> QdrantClient
#         qdrant_backend = db_wrapper.client
#         raw_client = qdrant_backend.client 
#         collection_name = qdrant_backend.collection_name
        
#         llm = ChatOllama(
#             model=MODEL_NAME,
#             base_url="http://localhost:11434",
#             temperature=0.1,
#             num_ctx=4096,
#             keep_alive="5m"
#         )
#         print("✅ Components Ready.")
#     except Exception as e:
#         print(f"❌ Init Failed: {e}")
#         return

#     # Prompt Template
#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are a precise and faithful assistant. Answer the user's question using ONLY the context provided below.
        
#         <context>
#         {context}
#         </context>

#         Question: {question}
        
#         Instructions:
#         1. Use ONLY the information from the context. If the answer is not present, say "I do not know".
#         2. Cite the Source for every key fact or number you use.
#         3. Do not hallucinate or make up numbers.

#         IMPORTANT: If you generate internal reasoning/thinking, enclose it in <think> tags.
#         The final part of your message must be the direct answer.
#         """
#     )
    
#     results_log = []
    
#     # 2. BATCH LOOP
#     start_total = time.time()
    
#     for i, question in enumerate(queries):
#         print(f"\n🔍 Processing [{i+1}/{len(queries)}]: {question[:50]}...")
#         q_start = time.time()
        
#         # A. HYBRID RETRIEVE
#         search_hits = db_wrapper.search(query_text=question, limit=5)
        
#         # B. CONTEXT RE-ASSEMBLY (The New Logic)
#         processed_paths = set()
#         final_context_blocks = []
#         sources = []

#         if not search_hits:
#             print("   🔴 No context found.")
#             final_answer = "I don't know (No documents found)."
#             raw_response = ""
#         else:
#             for hit in search_hits:
#                 # 1. Get the path of this hit
#                 path = hit.payload.get("hierarchy_path")
#                 src = hit.payload.get("source", "Unknown")
#                 sources.append(src)

#                 # 2. Check if we already processed this section
#                 if path and path in processed_paths:
#                     continue # Skip, we already fetched the full section
                
#                 if path:
#                     # 3. New Section Found: Fetch ALL its siblings
#                     full_section_text = fetch_full_section(raw_client, collection_name, path)
#                     if full_section_text:
#                         final_context_blocks.append(f"Source: {src} (Section: {path})\n{full_section_text}")
#                         processed_paths.add(path)
#                     else:
#                         # Fallback if fetch fails (shouldn't happen)
#                         final_context_blocks.append(f"Source: {src}\n{hit.payload.get('text', '')}")
#                 else:
#                     # 4. Fallback for chunks with no path
#                     final_context_blocks.append(f"Source: {src}\n{hit.payload.get('text', '')}")

#             # Join all full sections
#             context_text = "\n\n---\n\n".join(final_context_blocks)

#             # C. GENERATE
#             try:
#                 chain = prompt | llm | StrOutputParser()
#                 raw_response = chain.invoke({"context": context_text, "question": question})
#                 final_answer = clean_reasoning(raw_response)
#                 print(f"   ✅ Answered in {time.time() - q_start:.2f}s")
#             except Exception as e:
#                 print(f"   ❌ Error: {e}")
#                 final_answer = f"Error: {str(e)}"
#                 raw_response = ""

#         # D. LOG
#         results_log.append({
#             "id": i + 1,
#             "question": question,
#             "cleaned_answer": final_answer,
#             "raw_response_snippet": raw_response[:200] + "...",
#             "sources": list(set(sources)),
#             "retrieval_score_top1": search_hits[0].score if search_hits else 0,
#             "time_taken": round(time.time() - q_start, 2)
#         })

#     # 3. SAVE JSON
#     total_time = time.time() - start_total
#     print("\n" + "=" * 60)
    
#     output_file = LOG_DIR / f"hybrid_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
#     final_report = {
#         "meta": {
#             "timestamp": datetime.now().isoformat(),
#             "model_name": MODEL_NAME,
#             "mode": "Hybrid (Dense + Sparse) + Context Re-Assembly",
#             "vector_db": "Qdrant",
#             "total_questions": len(queries),
#             "total_time_seconds": round(total_time, 2)
#         },
#         "results": results_log
#     }

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(final_report, f, indent=4, ensure_ascii=False)

#     print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

# if __name__ == "__main__":
#     run_benchmark()


import time
import json
import re
import os
from pathlib import Path
from datetime import datetime
from qdrant_client.http import models

from backend.indexing.vector_store import VectorDBClient

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
queries = [
        "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
        "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
        "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
        "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
        "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",
]

LOG_DIR = Path("data/eval_results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def fetch_full_section(client, collection_name, hierarchy_path):
    """
    Retrieves ALL chunks that share the specific hierarchy_path
    and merges them back into one complete text block.
    """
    if not hierarchy_path:
        return None

    # Scroll through Qdrant to find all chunks with this path
    response, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.hierarchy_path", # Corrected Key
                    match=models.MatchValue(value=hierarchy_path)
                )
            ]
        ),
        limit=50, 
        with_payload=True,
        with_vectors=False
    )
    
    if not response:
        return None

    # Sort by chunk_index to ensure correct reading order
    sorted_chunks = sorted(response, key=lambda x: x.payload.get("chunk_index", 0))
    
    # Merge Texts
    full_text = "\n".join([c.payload.get("text", "") for c in sorted_chunks])
    
    return full_text

def run_retrieval_only():
    print(f"\n🧪 STARTING RETRIEVAL AUDIT (No LLM)")
    print(f"📊 Total Questions: {len(queries)}")
    print("============================================================")

    # 1. SETUP
    print("⚙️ Initializing Qdrant Client...")
    
    # Force Qdrant
    os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
    print("👉 Forcing Provider: QDRANT")

    try:
        # Initialize Wrapper
        db_wrapper = VectorDBClient()
        
        # ACCESS INTERNAL CLIENTS
        qdrant_backend = db_wrapper.client
        raw_client = qdrant_backend.client 
        collection_name = qdrant_backend.collection_name
        
        print("✅ Client Ready.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return
    
    results_log = []
    
    # 2. BATCH LOOP
    start_total = time.time()
    
    for i, question in enumerate(queries):
        print(f"\n🔍 Processing [{i+1}/{len(queries)}]: {question[:50]}...")
        q_start = time.time()
        
        # A. HYBRID RETRIEVE
        search_hits = db_wrapper.search(query_text=question, limit=5)
        
        # B. CONTEXT RE-ASSEMBLY
        processed_paths = set()
        retrieved_contexts = []
        sources = []

        if not search_hits:
            print("   🔴 No hits found.")
        else:
            for hit in search_hits:
                # 1. Get Metadata
                path = hit.payload.get("hierarchy_path")
                src = hit.payload.get("source", "Unknown")
                sources.append(src)

                # 2. Check if we already processed this section
                if path and path in processed_paths:
                    continue 
                
                content_block = ""
                context_type = ""

                if path:
                    # 3. Fetch Siblings
                    full_section_text = fetch_full_section(raw_client, collection_name, path)
                    if full_section_text:
                        content_block = full_section_text
                        context_type = "Full Section (Merged)"
                        processed_paths.add(path)
                    else:
                        # Fallback
                        content_block = hit.payload.get('text', '')
                        context_type = "Single Chunk (Fallback)"
                else:
                    # 4. No Path
                    content_block = hit.payload.get('text', '')
                    context_type = "Single Chunk (No Path)"

                # Save the raw text block
                retrieved_contexts.append({
                    "source": src,
                    "type": context_type,
                    "path": path,
                    "content": content_block
                })

        elapsed = time.time() - q_start
        print(f"   ✅ Retrieved {len(retrieved_contexts)} blocks in {elapsed:.2f}s")

        # C. LOG
        results_log.append({
            "id": i + 1,
            "question": question,
            "retrieved_contexts": retrieved_contexts,
            "top1_score": search_hits[0].score if search_hits else 0,
            "time_taken": round(elapsed, 2)
        })

    # 3. SAVE JSON
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"retrieval_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "mode": "Retrieval Only (Context Re-Assembly)",
            "vector_db": "Qdrant",
            "total_questions": len(queries),
            "total_time_seconds": round(total_time, 2)
        },
        "results": results_log
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"💾 Results saved to:\n   {output_file}")

if __name__ == "__main__":
    run_retrieval_only()