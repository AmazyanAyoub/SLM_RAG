import time
import json
import re
from pathlib import Path
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.indexing.dense_index import DenseIndexer
from backend.indexing.sparse_index import SparseIndexer # <--- NEW IMPORT
from backend.indexing.vector_store import VectorDBClient

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
OLLAMA_BASE_URL = "http://18.132.143.112:14528"
# MODEL_NAME = "qwen3:4b-instruct-2507-fp16" 
MODEL_NAME = "qwen3:8b" 


# The 10 Specific Benchmark Questions
# TEST_QUESTIONS = [
#     "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
#     "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
#     "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
#     "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
#     "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60% activity or more), what is the amount of the monthly income franchise (exemption) granted?",
#     "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
#     "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
#     "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
#     "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
#     "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I)."
# ]

# TEST_QUESTIONS = [
#     # --- 10 Short Questions (5 Simple, 5 Hard) ---

#     # Simple
#     "According to Article 2, paragraph 1, what is the base monthly maintenance amount (prestation mensuelle de base) for a single person before any multiplication factor is applied?",
#     "Under Article 12, what is the maximum duration for which provisional financial aid (aide financière provisoire) can be granted?",
#     "According to Article 40, which entity is responsible for financing the furniture and current stewardship of social aid premises provided by communes?",
#     "What is the minimum validity duration of the control document established by the office for emergency aid applicants, according to Article 30?",
#     "According to Article 3, paragraph 3, how frequently is the payment of rent systematically controlled?",

#     # Hard
#     "According to Article 9, paragraph 13, what is the maximum reimbursement amount for transport costs outside the canton related to the employment evaluation internship?",
#     "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
#     "According to Article 20, paragraph 4(e), what is the daily food allowance granted during leaves for a person staying in an establishment outside the canton of Geneva?",
#     "Based on Article 16, what is the standard maximum duration for ordinary financial aid granted to a person exercising an independent lucrative activity (excluding cases of incapacity)?",
#     "According to Article 5, paragraph 4, what is the maximum annual amount granted for household and family aid (aide ménagère et familiale)?",

#     # --- 10 Long Questions (5 Simple, 5 Hard) ---

#     # Simple
#     "List the four cumulative conditions required for a person to request aid for the management of periodic income under Article 23.",
#     "Describe the four specific items or services that constitute 'Prestations d'aide d'urgence' (emergency aid) provided in kind, as detailed in Article 24, paragraph 1.",
#     "According to Article 19, paragraph 2(a), list the specific monthly maintenance allowance amounts (forfait d'entretien) for exceptional financial aid for a household of 1, 2, 3, and 4 persons.",
#     "List the five cumulative conditions set out in Article 17A that a foreigner without a residence permit must meet to benefit from return aid (aide au retour).",
#     "According to Article 41, which specific regulation was abrogated by the entry into force of the current RIASI regulation?",

#     # Hard
#     "Explain the conditions and limits for the coverage of health insurance premiums under Article 4, specifically regarding the 'prime cantonale de référence' and the exception for new beneficiaries whose premiums exceed this reference.",
#     "Detail the specific categories of expenses related to children's activities that can be reimbursed under Article 9, paragraph 12, including the maximum annual amount for holiday camps (camps de vacances).",
#     "Compare the calculation of the monthly maintenance allowance for a beneficiary in a 'Communauté de majeurs' (Article 10) versus a beneficiary in 'Cohabitation' (Article 11) regarding how the base amount is determined.",
#     "Describe the process for the 'Allocation d'indépendant' under Article 23I, specifically focusing on the repayment conditions if the project is abandoned versus if the activity continues after 12 months.",
#     "According to Article 35, what are the specific financial consequences on the monthly maintenance allowance and situational benefits in cases of 'manquement aux devoirs' (breach of duties) versus 'manquement grave' (serious breach)?"
# ]

math_questions = [
    # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---

    # Short - Very Easy
    # "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
    # "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
    # "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
    # "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
    # "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",

    # # Short - Very Hard
    # "Calculate the monthly maintenance allowance for a household of 7 people using the base amount and the coefficients provided in Article 2 (including the add-on for people beyond 5).",
    # "According to Article 5, calculate the monthly diet allocation (allocation de régime) for a household where 3 people require a medical diet, applying the coefficient from Article 2.",
    # "Using the table in Article 3, calculate the maximum recognized rent for a family group consisting of a couple and 5 children.",
    # "Calculate the total combined Moving and Installation allowances (Article 9) for a family of 4 people.",
    # "Calculate the total monthly income franchise (Article 8) for a household where one person works at 100% and another works at 50%.",

    # --- LONG QUESTIONS (5 Very Easy, 5 Very Hard) ---

    # Long - Very Easy
    # "A beneficiary works full-time (100% activity) for a whole year. According to Article 8, calculate the total annual amount of the income franchise they are entitled to.",
    # "A single parent sends their 3 children to holiday camps during the summer. According to the limits set in Article 9, what is the maximum total reimbursement the family can receive for these camps in a single calendar year?",
    # "Under Article 5, a parent requires childcare for 50 hours in a specific month to attend a professional insertion program. Using the hourly rate provided, calculate the total cost coverage for that month.",
    # "A young adult (20 years old) student lives with their parents and is eligible for the integration supplement. According to Article 7, calculate the total amount of this supplement they generate for the family over a period of 6 months.",
    # "A single person is receiving exceptional financial aid under Article 19. Calculate the total monthly sum of their maximum rent coverage limit plus their pocket money allowance.",

    # # Long - Very Hard
    # "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
    "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
    # "A beneficiary works at 80% activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
    # "A family of 5 people (Couple + 3 children) moves into a new apartment. They claim the maximum Moving Allowance and the maximum Installation Allowance under Article 9. Calculate the grand total of these one-off benefits.",
    # "Calculate the difference in the Maximum Rent allowance (Article 3) between a 'Couple with 2 children' and a 'Couple with 4 children'. Return the difference as a positive number."
]

LOG_DIR = Path("data/gen_results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def clean_reasoning(text: str) -> str:
    """Removes <think> blocks and returns clean answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def run_benchmark():
    print(f"\n🧪 STARTING HYBRID BENCHMARK (Model: {MODEL_NAME})")
    print(f"📊 Total Questions: {len(math_questions)}")
    print("============================================================")

    # 1. SETUP (Run Once)
    print("⚙️ Initializing Hybrid RAG components...")
    try:
        dense_indexer = DenseIndexer()
        sparse_indexer = SparseIndexer() # <--- NEW
        client = VectorDBClient()
        
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=2048,
            keep_alive="5m"
        )
        print("✅ Components Ready.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a precise and faithful assistant. Answer the user's question using ONLY the context provided below.
        
        <context>
        {context}
        </context>

        Question: {question}
        
        Instructions:
        1. Use ONLY the information from the context. If the answer is not present, say "I do not know".
        2. Cite the Source for every key fact or number you use.
        3. Do not hallucinate or make up numbers.

        IMPORTANT: If you generate internal reasoning/thinking, enclose it in <think> tags.
        The final part of your message must be the direct answer.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    results_log = []
    
    # 2. BATCH LOOP
    start_total = time.time()
    
    for i, question in enumerate(math_questions):
        print(f"\n🔍 Processing [{i+1}/{len(math_questions)}]: {question[:50]}...")
        q_start = time.time()
        
        # A. HYBRID RETRIEVE
        # 1. Dense Vector
        query_dense = dense_indexer.embed_texts([question])[0]
        
        # 2. Sparse Vector
        sparse_output = sparse_indexer.compute_sparse_vectors([question])[0]
        query_indices = [int(k) for k in sparse_output.keys()]
        query_values = [float(v) for v in sparse_output.values()]
        
        # 3. Hybrid Search
        search_hits = client.search(
            query_text=question,
            query_dense=query_dense,
            query_sparse_indices=query_indices,
            query_sparse_values=query_values,
            limit=5
        )
        
        context_text = ""
        sources = []
        for hit in search_hits:
            text = hit.payload.get("text", "")
            # Use 'search_content' (summary) if available for better context context
            # smart_context = hit.payload.get("search_content", text) 
            src = hit.payload.get("source", "Unknown")
            sources.append(src)
            context_text += f"\n---\nSource: {src}\n{text}\n"

        # B. GENERATE
        try:
            if not search_hits:
                print("   🔴 No context found.")
                final_answer = "I don't know (No documents found)."
                raw_response = ""
            else:
                raw_response = chain.invoke({"context": context_text, "question": question})
                final_answer = clean_reasoning(raw_response)
                print(f"   ✅ Answered in {time.time() - q_start:.2f}s")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            final_answer = f"Error: {str(e)}"
            raw_response = ""

        # C. LOG
        results_log.append({
            "id": i + 1,
            "question": question,
            "cleaned_answer": final_answer,
            "raw_response_snippet": raw_response[:200] + "...",
            "sources": list(set(sources)),
            "retrieval_score_top1": search_hits[0].score if search_hits else 0, # RRF Score
            "time_taken": round(time.time() - q_start, 2)
        })

    # 3. SAVE JSON
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"hybrid_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "mode": "Hybrid (Dense + Sparse)",
            "total_questions": len(math_questions),
            "total_time_seconds": round(total_time, 2)
        },
        "results": results_log
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

if __name__ == "__main__":
    run_benchmark()