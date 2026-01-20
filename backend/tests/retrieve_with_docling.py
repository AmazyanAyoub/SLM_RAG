import time
import json
import re
import os
from pathlib import Path
from datetime import datetime
from qdrant_client.http import models

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.indexing.vector_store import VectorDBClient

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_NAME = "qwen3:4b"  # As requested

queries = [
        "According to Article 2, calculate the monthly maintenance amount (forfait d'entretien) for a couple (2 people) by applying the multiplier of 1.53 to the base amount.",
        "Using the rate in Article 5, what is the total cost coverage for 30 hours of childcare (frais de garde)?",
        "According to Article 9, calculate the total annual maximum reimbursement for holiday camps (camps de vacances) for a family with 2 children.",
        "Calculate the total installation allowance (frais d'installation) for a single person as defined in Article 9.",
        "According to Article 19, what is the combined monthly amount for 'pocket money' and 'clothing' for a single adult beneficiary?",
]

LOG_DIR = Path("data/gen_results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def clean_reasoning(text: str) -> str:
    """Removes <think> blocks and returns clean answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def fetch_full_section(client, collection_name, hierarchy_path):
    """
    Retrieves ALL chunks that share the specific hierarchy_path
    and merges them back into one complete text block.
    """
    if not hierarchy_path:
        return None

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

def run_benchmark():
    print(f"\n🧪 STARTING GENERATION BENCHMARK (Model: {MODEL_NAME})")
    print(f"📊 Total Questions: {len(queries)}")
    print("============================================================")

    # 1. SETUP
    print("⚙️ Initializing Components...")
    os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
    print("👉 Forcing Provider: QDRANT")

    try:
        # DB Client
        db_wrapper = VectorDBClient()
        qdrant_backend = db_wrapper.client
        raw_client = qdrant_backend.client 
        collection_name = qdrant_backend.collection_name
        
        # LLM Client
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=4096,
            keep_alive="5m"
        )
        print("✅ Components Ready.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 2. PROMPT TEMPLATE
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
    
    results_log = []
    
    # 3. BATCH LOOP
    start_total = time.time()
    
    for i, question in enumerate(queries):
        print(f"\n🔍 Processing [{i+1}/{len(queries)}]: {question[:50]}...")
        q_start = time.time()
        
        # A. RETRIEVE
        search_hits = db_wrapper.search(query_text=question, limit=5)
        
        # B. CONTEXT RE-ASSEMBLY
        processed_paths = set()
        final_context_blocks = []
        sources = []

        if not search_hits:
            print("   🔴 No context found.")
            context_text = ""
        else:
            for idx, hit in enumerate(search_hits):
                # Extract Metadata correctly
                payload = hit.payload
                meta = payload.get("metadata", {})
                
                path = meta.get("hierarchy_path")
                src = meta.get("source", "Unknown")
                sources.append(src)

                # Check duplication
                if path and path in processed_paths:
                    continue 
                
                block_text = ""
                
                if path:
                    # Fetch Siblings (Full Section)
                    full_section_text = fetch_full_section(raw_client, collection_name, path)
                    if full_section_text:
                        block_text = f"SOURCE: {src} (SECTION: {path})\n{full_section_text}"
                        processed_paths.add(path)
                    else:
                        block_text = f"SOURCE: {src}\n{payload.get('text', '')}"
                else:
                    block_text = f"SOURCE: {src}\n{payload.get('text', '')}"

                final_context_blocks.append(f"--- BLOCK {idx+1} ---\n{block_text}")

            context_text = "\n\n".join(final_context_blocks)

        # C. GENERATE
        try:
            if not context_text:
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

        # D. LOG
        results_log.append({
            "id": i + 1,
            "question": question,
            "cleaned_answer": final_answer,
            "sources": list(set(sources)),
            "retrieval_score_top1": search_hits[0].score if search_hits else 0,
            "time_taken": round(time.time() - q_start, 2)
        })

    # 4. SAVE
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    
    output_file = LOG_DIR / f"generation_qwen3-4b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "mode": "Hybrid + Context Re-Assembly",
            "vector_db": "Qdrant",
            "total_questions": len(queries),
            "total_time_seconds": round(total_time, 2)
        },
        "results": results_log
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"💾 Benchmark Complete! Results saved to:\n   {output_file}")

if __name__ == "__main__":
    run_benchmark()