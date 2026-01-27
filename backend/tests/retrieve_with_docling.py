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
# MODEL_NAME = "qwen3:4b-instruct-2507-fp16"
MODEL_NAME = "qwen3:4b"

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
    # "Quel est le montant de la franchise sur le revenu pour un revenu mensuel brut de 5000 frs ?",
    # "Quels est le délai de prise en charge des frais dentaires après approbation du médecin-dentiste conseil ?",
    # "Quel est le montant de l'aide d'urgence pour une famille de 5 personnes ?",
    # "Soit un dossier avec une seule personne aidée ayant un loyer mensuel de 2245 frs, une allocation logement mensuelle de 100 frs et des frais mensuels de garde-meubles de 500 frs.\nQuel est le montant mensuel total du loyer pris en charge ?",
    # "Quel est le montant du forfait d'entretien pour 1 personne étudiant en haute école ?",
    # "Quel est le montant du forfait d'entretien pour 1 personne suivant une formation dans le but d'obtenir le brevet fédéral ?",
    # "Quelle est la franchise sur le revenu pour 1 personne ayant un salaire de 2384.- frs ?",
    # "Quelle est le revenu à prendre en compte pour 1 personne ayant un revenu de 2384.- frs ?",
    # "Quelle est le revenu à prendre en compte pour 1 personne ayant un revenu de 250.- frs ?",
    # "Quelle est la durée d'aide financière maximale pour les indépendants ?",
    # "De combien de temps peut être prolongée la durée d'aide financière pour les indépendants ayant un certificat médical ?",
    # "Soit un dossier avec une seule personne bénéficiaire qui est majeure.\nQuelle est la franchise d'apprentissage à appliquer pour cette personne ?",
    # "Quel est le montant du forfait pour dépenses personnelles des personnes hiospitalisées en clinique ou à l'hopital ?",
    # # "Entre le nouveau règlement RASLP et l'ancien règlement RIASI, quel est le changement en ce qui concerne les frais liés à une activité non rémunérée ?",
    # "Quel est le taux de réduction du forfait d'entretien à appliquer en cas de faute grave ?",
    # # "Résume moi la LASLP",
    # # "Que signifie LASLP ?",
    # # "Que signifie RASLP ?",
    # "Quel est le loyer maximum pris en charge pour une personne ?",
    # "Quel est le loyer maximum pris en charge pour une famille composée d'une personne sans enfants à charge ?",
    # "Quel est le loyer maximum pris en charge pour une famille de 5 personnes ?",
    # "Le salaire d'apprentissage d'un enfant de 17 ans est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    # "Le salaire d'apprentissage d'un enfant de 17 ans en 1ère année est de 1000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    # "Le salaire d'apprentissage d'un enfant de 27 ans en 1ère année est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    # "Le salaire d'apprentissage d'un enfant de 17 ans en 1ère année est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    # "Le salaire d'apprentissage d'un enfant de 22 ans en 4ème année est de 1000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
    # "Le groupe familial est composé de 2 enfants en apprentissage, chacun en 3ème année.\nLes 2 enfants ont 19 ans et touchent chacun un salaire de 500 frs.\nQuelle est la franchise globale ?",
    # "Le groupe familial est composé de 2 enfants en apprentissage, chacun en 3ème année.\nLes 2 enfants ont 19 ans et touchent chacun un salaire de 800 frs.\nQuelle est la franchise globale ?"
]

LOG_DIR = Path("data/gen_results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def clean_reasoning(text: str) -> str:
    """Removes <think> blocks and returns clean answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def fetch_full_section_qdrant(client, collection_name, hierarchy_path):
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


def fetch_full_section(pg_backend, table_name, hierarchy_path):
    """
    Retrieves ALL chunks that share the specific hierarchy_path from Postgres.
    """
    if not hierarchy_path:
        return None

    # Query to find all chunks with the matching hierarchy_path in the JSONB metadata
    query_sql = f"""
        SELECT content, (metadata->>'chunk_index')::int as idx
        FROM {table_name}
        WHERE metadata->>'hierarchy_path' = %s
        ORDER BY idx ASC;
    """
    
    try:
        # Use the existing connection from your PostgresVectorDB instance
        with pg_backend.conn.cursor() as cur:
            cur.execute(query_sql, (hierarchy_path,))
            rows = cur.fetchall()
            
            if not rows:
                return None
            
            # Merge text from all parts of the section
            full_text = "\n".join([row[0] for row in rows])
            return full_text
    except Exception as e:
        print(f"   ⚠️ SQL Fetch Error: {e}")
        return None

def run_benchmark():
    print(f"\n🧪 STARTING GENERATION BENCHMARK (Model: {MODEL_NAME})")
    print(f"📊 Total Questions: {len(queries)}")
    print("============================================================")

    # 1. SETUP
    print("⚙️ Initializing Components...")
    # os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
    # print("👉 Forcing Provider: QDRANT")

    try:
        # DB Client
        # db_wrapper = VectorDBClient()
        # qdrant_backend = db_wrapper.client
        # raw_client = qdrant_backend.client 
        # collection_name = qdrant_backend.collection_name

        db_wrapper = VectorDBClient()
            # Access the Postgres database client directly
        pg_backend = db_wrapper.client 
        table_name = pg_backend.table_name
    
        # LLM Client
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_ctx=4096,
            # keep_alive="10min"
            timeout=120.0
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
        search_hits = db_wrapper.search(query_text=question, limit=3)
        
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
                    full_section_text = fetch_full_section(pg_backend, table_name, path)
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
                print(len(context_text))
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