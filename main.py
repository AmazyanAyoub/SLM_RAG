import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path to ensure imports work correctly
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.langgraph_flow.graph_builder import app

# Directory for saving results
LOG_DIR = Path("data/gen_results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 📝 DEFINE YOUR QUESTIONS HERE
# ------------------------------------------------------------------
Questions = [
        "Under Article 19, paragraph 2(e), what is the maximum monthly rent coverage amount for persons receiving exceptional financial aid?",
        "According to Article 19, paragraph 2(a), list the specific monthly maintenance allowance amounts (forfait d'entretien) for exceptional financial aid for a household of 1, 2, 3, and 4 persons.",
        "A family consists of a couple and 4 children (total 6 people). Calculate their total monthly entitlement sum including the Base Maintenance (Article 2) and the Maximum Rent (Article 3).",
        "A household of 4 people (couple + 2 children) are all prescribed a specific medical diet. According to Article 5, which applies the Article 2 coefficients to the base diet allowance, calculate the total monthly diet allocation for this entire family.",
        "A beneficiary works at 80%| activity (generating a franchise) and has a dependent child aged 16 who attends school (generating an integration supplement). Calculate the total monthly sum of these two specific incentive benefits based on Articles 7 and 8.",
]

def main():
    print(f"\n🧪 STARTING AGENTIC BENCHMARK")
    print(f"📊 Total Questions: {len(Questions)}")
    print("==========================================")

    results_log = []
    start_total = time.time()

    for i, query in enumerate(Questions):
        print(f"\n🔍 Processing [{i+1}/{len(Questions)}]: {query}...")
        q_start = time.time()

        # Initialize State
        initial_state = {
            "question": query,
            "attempts": 0,
            "steps": [],
            "documents": [],
            "generation": None,
            "classification": None,
            "error": None,
            "grade": None
        }

        try:
            # Run the graph
            result = app.invoke(initial_state)
            
            # Extract Data
            final_answer = result.get("generation")
            steps = result.get("steps", [])
            documents = result.get("documents", [])
            
            # Extract Sources & Scores
            sources = []
            top_score = 0.0
            
            if documents:
                for doc in documents:
                    # Handle Document object or dict
                    meta = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
                    src = meta.get("source", "Unknown")
                    sources.append(src)
                    
                    score = meta.get("score", 0.0)
                    if score > top_score:
                        top_score = score
            
            sources = list(set(sources))
            time_taken = round(time.time() - q_start, 2)

            print(f"   ✅ Finished in {time_taken}s | Steps: {steps}")

            # Log Result
            results_log.append({
                "id": i + 1,
                "question": query,
                "final_answer": final_answer,
                "steps_taken": steps,
                "sources": sources,
                "retrieval_score_top1": top_score,
                "time_taken": time_taken
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results_log.append({
                "id": i + 1,
                "question": query,
                "error": str(e),
                "time_taken": round(time.time() - q_start, 2)
            })

    # Save JSON
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"agent_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "mode": "Agentic Graph (LangGraph)",
            "total_questions": len(Questions),
            "total_time_seconds": round(total_time, 2)
        },
        "results": results_log
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

if __name__ == "__main__":
    main()