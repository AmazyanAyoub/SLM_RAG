import os
import json
import time
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.indexing.vector_store import VectorDBClient

# ==========================================
# 📝 DEFINE YOUR TEST QUESTIONS HERE
# ==========================================
queries = [
    # --- SHORT QUESTIONS (5 Very Easy, 5 Very Hard) ---

    "According to Article 1, what is the fortune limit (asset limit) for a couple to be eligible for financial aid?",
    # "List the specific needs that the basic monthly maintenance allowance (forfait mensuel) is intended to cover, as detailed in Article 2, paragraph 2.",
    # "Based on Article 3, what is the maximum recognized rent amount for a family group composed of one or two persons and two children?",
    # "Under Article 5, what are the criteria and the maximum reimbursement amount for participating in the costs of a temporary stay for a child (visitation rights)?",
    # "According to Article 8, if a beneficiary works between 104 and 121 hours per month (60%|activity or more), what is the amount of the monthly income franchise (exemption) granted?",
    # "What are the conditions for the reimbursement of orthodontic treatment costs, and is it available to adults? (Reference Article 9).",
    # "What is the maximum duration of the 'stage d'évaluation à l'emploi' (employment evaluation internship), and how many days per week must it be attended? (Reference Article 23E).",
    # "For persons receiving emergency aid (rejected asylum seekers), what is the daily financial amount allocated for food, and what happens to this amount if the person adopts delinquent behavior? (Reference Articles 29B and 29C).",
    # "At what annual income level does the state require a financial contribution from parents (married couple) for an adult beneficiary who is not considered a 'young adult' (Art. 37)? (Reference Article 38).",
    # "What is the maximum amount of the 'allocation d'indépendant' (self-employment allowance), and is this amount a grant or a reimbursable loan? (Reference Article 23I)."

    # # # --- 10 Short Questions (5 Simple, 5 Hard) ---

    # # # Simple
    # "According to Article 2, paragraph 1, what is the base monthly maintenance amount (prestation mensuelle de base) for a single person before any multiplication factor is applied?",
    # "Under Article 12, what is the maximum duration for which provisional financial aid (aide financière provisoire) can be granted?",
    # "According to Article 40, which entity is responsible for financing the furniture and current stewardship of social aid premises provided by communes?",
    # "What is the minimum validity duration of the control document established by the office for emergency aid applicants, according to Article 30?",
    # "According to Article 3, paragraph 3, how frequently is the payment of rent systematically controlled?",

    # # Hard
    # "According to Article 9, paragraph 13, what is the maximum reimbursement amount for transport costs outside the canton related to the employment evaluation internship?",
    # "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
    # "According to Article 20, paragraph 4(e), what is the daily food allowance granted during leaves for a person staying in an establishment outside the canton of Geneva?",
    # "Based on Article 16, what is the standard maximum duration for ordinary financial aid granted to a person exercising an independent lucrative activity (excluding cases of incapacity)?",
    # "According to Article 5, paragraph 4, what is the maximum annual amount granted for household and family aid (aide ménagère et familiale)?",

    # # # --- 10 Long Questions (5 Simple, 5 Hard) ---

    # # # Simple
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
    # "According to Article 35, what are the specific financial consequences on the monthly maintenance allowance and situational benefits in cases of 'manquement aux devoirs' (breach of duties) versus 'manquement grave' (serious breach)?"


    # "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
    # "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
    # "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
    # "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
    # "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",

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

    # # "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
    # "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
    # "A beneficiary works at 80%| activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
    # "A family of 5 people (Couple + 3 children) moves into a new apartment. They claim the maximum Moving Allowance and the maximum Installation Allowance under Article 9. Calculate the grand total of these one-off benefits.",
    # "Calculate the difference in the Maximum Rent allowance (Article 3) between a 'Couple with 2 children' and a 'Couple with 4 children'. Return the difference as a positive number."
]


def run_batch_test():
    print("🧪 STARTING HYBRID RETRIEVAL TEST (10 Questions)")
    print("============================================================")

    # 1. Initialize System (Both Indexers)
    try:
        client = VectorDBClient()
        print(f"✅ Hybrid System Initialized (Provider: {client.provider.upper()}).")
    except Exception as e:
        print(f"❌ System Init Failed: {e}")
        return

    results_log = []
    total_score = 0
    total_retrieved = 0

    # 2. Loop Through Questions
    start_time = time.time()
    
    for i, question in enumerate(queries):
        print(f"🔍 [{i+1}/10] Querying: {question[:50]}...", end="\r")
        
        # A. Hybrid Search (Dense + Sparse)
        search_hits = client.search(
            query_text=question,
            limit=2
        )
        
        # Format Hits for JSON
        hits_data = []
        for hit in search_hits:
            hits_data.append({
                "score": float(hit.score), # This is now an RRF Score (Fusion), not just Cosine
                "source": hit.payload.get("source", "Unknown"),
                "text_preview": hit.payload.get("text", ""), # First 300 chars
                "search_content": hit.payload.get("search_content") or hit.payload.get("context_summary", ""), # The Smart Context
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