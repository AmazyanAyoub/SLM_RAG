import time
import json
import re
from pathlib import Path
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.indexing.dense_index import DenseIndexer
from backend.indexing.vector_store import VectorDBClient

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
OLLAMA_BASE_URL = "http://18.132.143.112:14528"
MODEL_NAME = "qwen3:8b" 

# The 10 Specific Benchmark Questions
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

LOG_DIR = Path("data/gen_results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def clean_reasoning(text: str) -> str:
    """Removes <think> blocks and returns clean answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def run_benchmark():
    print(f"\n🧪 STARTING BENCHMARK (Model: {MODEL_NAME})")
    print(f"📊 Total Questions: {len(TEST_QUESTIONS)}")
    print("=" * 60)

    # 1. SETUP (Run Once)
    print("⚙️ Initializing RAG components...")
    try:
        indexer = DenseIndexer()
        client = VectorDBClient()
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            keep_alive="5m"
        )
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Answer the user's question using ONLY the context below.
        
        <context>
        {context}
        </context>

        Question: {question}
        
        IMPORTANT: If you generate internal reasoning/thinking, enclose it in <think> tags.
        However, the final part of your message must be the direct answer.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    results_log = []
    
    # 2. BATCH LOOP
    start_total = time.time()
    
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n🔍 Processing [{i+1}/{len(TEST_QUESTIONS)}]: {question[:50]}...")
        q_start = time.time()
        
        # A. RETRIEVE
        query_vector = indexer.embed_texts([question])[0]
        search_hits = client.search(query_vector, limit=3)
        
        context_text = ""
        sources = []
        for hit in search_hits:
            text = hit.payload.get("text", "")
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
            "raw_response_snippet": raw_response[:200] + "...", # Save space
            "sources": list(set(sources)), # Unique sources
            "retrieval_score_top1": search_hits[0].score if search_hits else 0,
            "time_taken": round(time.time() - q_start, 2)
        })

    # 3. SAVE JSON
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "total_questions": len(TEST_QUESTIONS),
            "total_time_seconds": round(total_time, 2)
        },
        "results": results_log
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

if __name__ == "__main__":
    run_benchmark()