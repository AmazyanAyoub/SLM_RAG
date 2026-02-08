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
# queries = [
#         # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---

#         "Quelle est le revenu à prendre en compte pour 1 personne ayant un revenu de 2384.- frs ?"

#         # "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
#         # "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
#         # "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
#         # "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
#         # "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60%|activity or more), what is the amount of the monthly income franchise (exemption) granted?",
#         # "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
#         # "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
#         # "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
#         # "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
#         # "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I).",

#         # # # # --- 10 Short Questions (5 Simple, 5 Hard) ---

#         # # # # Simple
#         # "According to Article 2, paragraph 1, what is the base monthly maintenance amount (prestation mensuelle de base) for a single person before any multiplication factor is applied?",
#         # "Under Article 12, what is the maximum duration for which provisional financial aid (aide financière provisoire) can be granted?",
#         # "According to Article 40, which entity is responsible for financing the furniture and current stewardship of social aid premises provided by communes?",
#         # "What is the minimum validity duration of the control document established by the office for emergency aid applicants, according to Article 30?",
#         # "According to Article 3, paragraph 3, how frequently is the payment of rent systematically controlled?",

#         # # # Hard
#         # "According to Article 9, paragraph 13, what is the maximum reimbursement amount for transport costs outside the canton related to the employment evaluation internship?",
#         # "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
#         # "According to Article 20, paragraph 4(e), what is the daily food allowance granted during leaves for a person staying in an establishment outside the canton of Geneva?",
#         # "Based on Article 16, what is the standard maximum duration for ordinary financial aid granted to a person exercising an independent lucrative activity (excluding cases of incapacity)?",
#         # "According to Article 5, paragraph 4, what is the maximum annual amount granted for household and family aid (aide ménagère et familiale)?",

#         # # --- 10 Long Questions (5 Simple, 5 Hard) ---

#         # # Simple
#         # "List the four cumulative conditions required for a person to request aid for the management of periodic income under Article 23.",
#         # "Describe the four specific items or services that constitute 'Prestations d'aide d'urgence' (emergency aid) provided in kind, as detailed in Article 24, paragraph 1.",
#         # "According to Article 19, paragraph 2(a), list the specific monthly maintenance allowance amounts (forfait d'entretien) for exceptional financial aid for a household of 1, 2, 3, and 4 persons.",
#         # "List the five cumulative conditions set out in Article 17A that a foreigner without a residence permit must meet to benefit from return aid (aide au retour).",
#         # "According to Article 41, which specific regulation was abrogated by the entry into force of the current RIASI regulation?",

#         # # Hard
#         # "Explain the conditions and limits for the coverage of health insurance premiums under Article 4, specifically regarding the 'prime cantonale de référence' and the exception for new beneficiaries whose premiums exceed this reference.",
#         # "Detail the specific categories of expenses related to children's activities that can be reimbursed under Article 9, paragraph 12, including the maximum annual amount for holiday camps (camps de vacances).",
#         # "Compare the calculation of the monthly maintenance allowance for a beneficiary in a 'Communauté de majeurs' (Article 10) versus a beneficiary in 'Cohabitation' (Article 11) regarding how the base amount is determined.",
#         # "Describe the process for the 'Allocation d'indépendant' under Article 23I, specifically focusing on the repayment conditions if the project is abandoned versus if the activity continues after 12 months.",
#         # "According to Article 35, what are the specific financial consequences on the monthly maintenance allowance and situational benefits in cases of 'manquement aux devoirs' (breach of duties) versus 'manquement grave' (serious breach)?",


#         # "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
#         # "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
#         # "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
#         # "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
#         # "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",

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

#         # "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
#         # "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
#         # "A beneficiary works at 80%| activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
#         # "A family of 5 people (Couple + 3 children) moves into a new apartment. They claim the maximum Moving Allowance and the maximum Installation Allowance under Article 9. Calculate the grand total of these one-off benefits.",
#         # "Calculate the difference in the Maximum Rent allowance (Article 3) between a 'Couple with 2 children' and a 'Couple with 4 children'. Return the difference as a positive number."
#     ]


queries = [
    "Quel est le montant du forfait d'entretien pour une personne ?",
    "Quel est le montant du forfait d'entretien pour une famille de 3 personnes ?",
    "Quel est le montant du forfait d'entretien ?",
    # "Quels sont les changements de montant au niveau du forfait d'entretien entre l'ancienne loi LIASI et la nouvelle loi LASLP ?",
    # "Quel est dans la LASLP le montant du forfait d'entretien pour une famille de 7 personnes ?",
    # "Quel est dans la RIASI le montant du forfait d'entretien pour une famille de 7 personnes ?",
    "Quelle est la limite de fortune pour un couple sans enfant ?",
    "Quelle est la limite de fortune pour un couple avec 2 enfants ?",
    "Quelle est la limite de fortune pour un couple avec 1 enfant ?",
    "Quelle est la limite de fortune pour une personne mineure ?",
    "Quel est le forfait qui couvre les frais de repas scolaires ?",
    "Quel est le montant de la franchise sur le revenu pour un revenu mensuel de 5000 frs ?",
    "Quel est le montant de la franchise sur le revenu pour un revenu mensuel net de 5000 frs ?",

    "Quel est le montant de la franchise sur le revenu pour un revenu mensuel brut de 5000 frs ?",
    "Quels est le délai de prise en charge des frais dentaires après approbation du médecin-dentiste conseil ?",
    "Quel est le montant de l'aide d'urgence pour une famille de 5 personnes ?",
    "Soit un dossier avec une seule personne aidée ayant un loyer mensuel de 2245 frs, une allocation logement mensuelle de 100 frs et des frais mensuels de garde-meubles de 500 frs.\nQuel est le montant mensuel total du loyer pris en charge ?",
    "Quel est le montant du forfait d'entretien pour 1 personne étudiant en haute école ?",
    "Quel est le montant du forfait d'entretien pour 1 personne suivant une formation dans le but d'obtenir le brevet fédéral ?",
    "Quelle est la franchise sur le revenu pour 1 personne ayant un salaire de 2384.- frs ?",
    "Quelle est le revenu à prendre en compte pour 1 personne ayant un revenu de 2384.- frs ?",
    "Quelle est le revenu à prendre en compte pour 1 personne ayant un revenu de 250.- frs ?",
    "Quelle est la durée d'aide financière maximale pour les indépendants ?",

    "De combien de temps peut être prolongée la durée d'aide financière pour les indépendants ayant un certificat médical ?",
    "Soit un dossier avec une seule personne bénéficiaire qui est majeure.\nQuelle est la franchise d'apprentissage à appliquer pour cette personne ?",
    "Quel est le montant du forfait pour dépenses personnelles des personnes hiospitalisées en clinique ou à l'hopital ?",
    # "Entre le nouveau règlement RASLP et l'ancien règlement RIASI, quel est le changement en ce qui concerne les frais liés à une activité non rémunérée ?",
    "Quel est le taux de réduction du forfait d'entretien à appliquer en cas de faute grave ?",
    # "Résume moi la LASLP",
    # "Que signifie LASLP ?",
    # "Que signifie RASLP ?",
    "Quel est le loyer maximum pris en charge pour une personne ?",
    "Quel est le loyer maximum pris en charge pour une famille composée d'une personne sans enfants à charge ?",
    "Quel est le loyer maximum pris en charge pour une famille de 5 personnes ?",
    "Le salaire d'apprentissage d'un enfant de 17 ans est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    "Le salaire d'apprentissage d'un enfant de 17 ans en 1ère année est de 1000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    "Le salaire d'apprentissage d'un enfant de 27 ans en 1ère année est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    "Le salaire d'apprentissage d'un enfant de 17 ans en 1ère année est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    "Le salaire d'apprentissage d'un enfant de 22 ans en 4ème année est de 1000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    "Le groupe familial est composé de 2 enfants en apprentissage, chacun en 3ème année.\nLes 2 enfants ont 19 ans et touchent chacun un salaire de 500 frs.\nQuelle est la franchise globale ?",
    "Le groupe familial est composé de 2 enfants en apprentissage, chacun en 3ème année.\nLes 2 enfants ont 19 ans et touchent chacun un salaire de 800 frs.\nQuelle est la franchise globale ?"
]

LOG_DIR = Path("data/retrieval_audit")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def fetch_full_section(client, collection_name, hierarchy_path):
    """
    Retrieves ALL chunks that share the specific hierarchy_path.
    """
    if not hierarchy_path:
        return None

    # Scroll through Qdrant
    response, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    # ✅ CORRECT: Access nested metadata field
                    key="metadata.hierarchy_path", 
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

    # ✅ CORRECT: Sort by nested chunk_index
    sorted_chunks = sorted(
        response, 
        key=lambda x: x.payload.get("metadata", {}).get("chunk_index", 0)
    )
    
    # Merge Texts
    full_text = "\n".join([c.payload.get("text", "") for c in sorted_chunks])
    
    return full_text

def run_retrieval_only():
    print(f"\n🧪 STARTING RETRIEVAL AUDIT (No LLM)")
    print(f"📊 Total Questions: {len(queries)}")
    print("============================================================")

    # 1. SETUP
    print("⚙️ Initializing Qdrant Client...")
    os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
    print("👉 Forcing Provider: QDRANT")

    try:
        db_wrapper = VectorDBClient()
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
        
        # A. SEARCH
        search_hits = db_wrapper.search(query_text=question, limit=5)
        
        # B. CONTEXT PROCESSING
        processed_paths = set()
        retrieved_contexts = []

        if not search_hits:
            print("   🔴 No hits found.")
        else:
            for hit in search_hits:
                # ---------------------------------------------------------
                # ✅ FIX IS HERE: Extract from 'metadata' sub-dictionary
                # ---------------------------------------------------------
                payload = hit.payload
                meta = payload.get("metadata", {}) # Safety access
                
                path = meta.get("hierarchy_path")
                src = meta.get("source", "Unknown")
                # ---------------------------------------------------------

                # 2. Check overlap
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
                        content_block = payload.get('text', '')
                        context_type = "Single Chunk (Fallback)"
                else:
                    # 4. No Path
                    content_block = payload.get('text', '')
                    context_type = "Single Chunk (No Path)"

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

    # 3. SAVE
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"retrieval_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "mode": "Retrieval Only (Fixed Metadata)",
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