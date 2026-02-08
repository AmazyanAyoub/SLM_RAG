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
MODEL_NAME = "qwen3:4b-instruct-2507-fp16"
# MODEL_NAME = "qwen3:8b"
# MODEL_NAME = "qwen3:8b-q8_0"
# MODEL_NAME = "initium/law_model"
# MODEL_NAME = "mistral-nemo"


queries = [
    "Quel est le montant du forfait d'entretien pour une personne ?",
    "Quel est le montant du forfait d'entretien pour une famille de 3 personnes ?",
    "Quel est le montant du forfait d'entretien ?",
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
    "Quel est le taux de réduction du forfait d'entretien à appliquer en cas de faute grave ?",
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


# queries = [
#     "Quel est le montant de l'aide d'urgence pour une famille de 5 personnes ?",
#     "Quel est le montant du forfait d'entretien pour 1 personne étudiant en haute école ?",
#     # "Soit un dossier avec une seule personne bénéficiaire qui est majeure.\nQuelle est la franchise d'apprentissage à appliquer pour cette personne ?",
#     # "Quel est le montant du forfait pour dépenses personnelles des personnes hiospitalisées en clinique ou à l'hopital ?",
#     # "Quel est le loyer maximum pris en charge pour une famille de 5 personnes ?",
#     "Le salaire d'apprentissage d'un enfant de 17 ans est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
#     "Le salaire d'apprentissage d'un enfant de 27 ans en 1ère année est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
#     # "Le salaire d'apprentissage d'un enfant de 17 ans en 1ère année est de 2000 frs.\nQuelle est la franchise à appliquer sur ce salaire ?",
# ]

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
            base_url="http://18.132.143.112:11435",
            temperature=0.1,
            num_ctx=4096,
            seed=42,
            # keep_alive="10min"
            timeout=120.0
        )
        print("✅ Components Ready.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 2. PROMPT TEMPLATE
    # PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    #         """
    #         You are a precise and helpful document assistant.

    #         ### CORE INSTRUCTIONS:
    #         1. **Answer strictly** based on the provided **Context Blocks** below. Do not use outside knowledge.
    #         2. **HANDLE SPLIT TEXT (CRITICAL):**
    #         - The context is provided as a sequence of continuous text chunks (Block 1, Block 2, etc.).
    #         - **Text may be cut off:** A sentence, list, or paragraph ending in Block N often continues immediately in Block N+1.
    #         - **Stitch mentally:** If Block 1 ends abruptly (e.g., with a colon `:`, a hyphen `-`, or mid-sentence), read the start of Block 2 to complete the thought.
    #         - Treat the blocks as a single continuous document, not separate snippets.
    #         3. **Synthesis:** If the answer requires combining facts from Block 1 and Block 3, merge them into a coherent response.
    #         4. **Fallback:** If the answer is not found in the context, state clearly: "The provided documents do not contain this information."

    #         ### CONTEXT BLOCKS:
    #         {context}

    #         ### USER QUESTION:
    #         {question}

    #         ### ANSWER:
    #         """
    # )

    # PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    #     """
    #     You are an expert legal assistant specialized in interpreting regulatory documents. Your goal is to answer the user's question with extreme precision using ONLY the provided Context Blocks.

    #     ### CORE REASONING PROTOCOL (MUST FOLLOW):
    #     1. **Hierarchy of Rules:**
    #     - Look for **Specific Rules** that override **General Rules**. (e.g., A rule for "Students" overrides a rule for "General Population").
    #     - Look for **Exceptions** or **Special Conditions** (e.g., "Sauf si...", "Except for...", "Pour les mineurs...").
    #     - If a specific case is mentioned in the question (e.g., "Hospitalized", "Apprentice", "Minor"), YOU MUST prioritize the text sections explicitly addressing that case over general tables.

    #     2. **Handling Definitions & Calculations:**
    #     - **Zero Assumptions:** Do not assume variables (like age or family size) unless stated in the question. If the answer depends on a missing variable, state the rule (e.g., "The amount depends on X: if X is A, then 100; if X is B, then 200").
    #     - **Math:** If the text provides a formula or percentage (e.g., "70% of base amount"), perform the calculation explicitly in your response.
    #     - **Caps/Limits:** Always check if a calculated amount is subject to a maximum cap (ceiling) mentioned in a nearby or related block.

    #     3. **Context stitching:**
    #     - The Context Blocks are segments of a larger legal text. If a sentence ends abruptly in Block N, it continues in Block N+1. Read across blocks to capture full clauses.

    #     ### RESPONSE FORMAT:
    #     - **Direct Answer:** Start with the specific number, rate, or rule.
    #     - **Reasoning:** Briefly cite the logic or calculation (e.g., "Base amount 1000 + 15% franchise = 1150").
    #     - **Conditionality:** If the answer is ambiguous based on the text, explain the variables (e.g., "It is 100 for adults, but 50 for minors").
    #     - **Negative Constraint:** If the answer is truly not in the text, output: "The provided documents do not contain this information."

    #     ### CONTEXT BLOCKS:
    #     {context}

    #     ### USER QUESTION:
    #     {question}

    #     ### ANSWER:
    #     """
    # )

#     PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
#     """
#     You are an **Expert Legal Algorithm** specialized in strict regulatory interpretation.
#     Your task is not just to "read" the text, but to **execute the logic** of the law to answer the user's question.

#     ### EXECUTION PROTOCOL (MANDATORY STEPS):

#     **STEP 1: SEMANTIC VERIFICATION (Avoid Wrong Categories)**
#     - **Exact Match:** Ensure the text you cite matches the specific legal term in the question.
#       - *Example:* If the user asks for "Aide d'urgence" (Emergency Aid), DO NOT cite amounts for "Argent de poche" (Pocket Money) or "Forfait d'entretien" (Maintenance) unless explicitly linked in the text.
#     - **Unit Check:** Verify units. If the text says "francs par jour" (per day), do not report it as a monthly total unless you explicitly calculate it (x30) and state the assumption.

#     **STEP 2: ELIGIBILITY CHECK (The "Gatekeeper" Rule)**
#     - **Prerequisites First:** Before applying ANY formula, check the definitions/conditions at the start of the article.
#       - *Example:* If a rule says "For apprentices up to 25 years old", and the user asks about a "27-year-old apprentice", you must STOP and report that the rule does not apply.
#     - **Status Priority:** Check if the user's subject has a special status (Student, Hospitalized, Independent, Minor). Specific status rules **ALWAYS override** general population rules.

#     **STEP 3: HIERARCHY & CALCULATION**
#     - **Specific > General:** If Block 1 has a general "Maintenance Fee" and Block 2 has a "Maintenance Fee for Students," and the user asks about a Student, IGNORE Block 1.
#     - **The "Global Cap" Scan:** After calculating a number (e.g., a franchise or rent), you must mentally scan ALL provided blocks for a **"Plafond" (Ceiling)**, **"Limite"**, or **"Maximum"** that applies to that category.
#       - *Rule:* `Final_Answer = MIN(Calculated_Amount, Cap_Amount)`

#     **STEP 4: VARIABLE DECLARATION (Zero Assumptions)**
#     - **Do NOT Guess:** If a variable is missing (e.g., family composition for a "family of 5"), do NOT assume "1 adult + 4 kids".
#     - **Output Logic:** Instead, define the scenarios.
#       - *Correct format:* "The amount depends on composition: Case A (2 adults, 3 kids) is X; Case B (1 adult, 4 kids) is Y."

#     ### RESPONSE FORMAT:
#     1. **Direct Answer:** The specific number, "Not Applicable", or "Depends on variables".
#     2. **Legal Basis:** "According to Article X..." (if available).
#     3. **Step-by-Step Logic:** Show the formula: `Base (1000) + Modifier (50%) = 1500`.
#     4. **Constraints Applied:** "Capped at 1200 as per Article Y."
#     5. **Fallback:** If the exact answer is not in the text, write: "The provided documents do not contain this information."

#     ### CONTEXT BLOCKS:
#     {context}

#     ### USER QUESTION:
#     {question}

#     ### ANSWER:
#     """
# )
    
#     PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
#     """
#     You are an analytical document assistant. Your goal is to answer the user's question by extracting and **executing the logic** found in the Context Blocks.

#     ### 🧠 LOGIC PROTOCOL (Apply these abstract rules to ANY document):

#     1. **FORMULA EXECUTION (The "Hidden Math" Rule):**
#        - Legal texts often define a *method* rather than a specific number.
#        - If the user asks for a "Countable Amount" or "Net Value" and the text defines a **Deduction**, **Exemption**, or **Percentage** rule, you MUST perform the calculation.
#        - *Logic:* `Answer = Input Value - Applicable Deduction/Exemption`.
#        - Do not say "Not found" just because the specific number (e.g., 2384) isn't written in the text. Apply the rule to the number.

#     2. **SEMANTIC DISTINCTION (The "Wrong Category" Rule):**
#        - Distinguish between a **Main Benefit** and a **Subsidiary Allowance**.
#        - If the user asks for the "Amount of Aid," look for the *primary* rate table. Do not answer with a smaller sub-allowance (like "pocket money" or "clothing allowance") just because it shares a keyword.

#     3. **GLOBAL CONSTRAINTS (The "Ceiling" Rule):**
#        - A calculation is never finished until you check for **Limits**.
#        - After applying a formula, scan the text for a **"Maximum"**, **"Ceiling"**, or **"Cap"** (Plafond/Limite) that applies to the family or category.
#        - *Logic:* `Final Answer = MIN(Calculated Amount, Cap Amount)`.

#     4. **EXTRAPOLATION (The "Family Size" Rule):**
#        - If a reference table stops at size N (e.g., 4 people) but provides a rule for "each additional person," you **MUST calculate** the value for size N+1, N+2, etc.
#        - Do not say "Not found" for a family of 5; calculate it using the "base + increment" logic provided.

#     5. **ELIGIBILITY GATING (The "Age/Status" Rule):**
#        - Before applying a rule, check the **header definitions** for age or status limits (e.g., "Up to 25 years").
#        - If the subject in the question exceeds the limit (e.g., 27 years old), the specific rule **DOES NOT APPLY** (Result is usually 0 or None).

#     ### RESPONSE FORMAT:
#     - **Direct Answer:** The specific number or "Not Applicable".
#     - **Logic Used:** Briefly state the formula: "Base (X) + Increment (Y) = Z" or "Income (A) - Deduction (B) = C".
#     - **Constraint:** Mention if a cap was hit or an age limit blocked the rule.

#     ### CONTEXT BLOCKS:
#     {context}

#     ### USER QUESTION:
#     {question}

#     ### ANSWER:
#     """
# )

#     PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
#     """
#     <system_role>
#     You are an expert Legal Calculation Engine. Your goal is to extract logic from the Context and execute it to answer the User Question.
#     You are a strict rule-follower. You do not summarize; you calculate.
#     </system_role>

#     <context_data>
#     {context}
#     </context_data>

#     <execution_algorithm>
#     To answer the question, you MUST strictly follow this 6-step algorithm. Failure to follow the order will result in a penalty.

#     1. **SCOPE CHECK (The "Wrong Book" Rule):**
#        - Legal texts often contain different regimes (e.g., "Standard/General" vs. "Emergency/Asylum").
#        - **RULE:** If the User Question is standard, you MUST IGNORE values found in sections labeled "Emergency" (Urgence), "Asylum" (Asile), or "Exceptional Measures".
#        - *Logic:* General Question = General Section Data Only. Do not cross-contaminate limits.

#     2. **ENTITY MAPPING (The "Synonym" Rule):**
#        - Legal terms often vary. You must map the User's terms to the Text's definitions.
#        - **Common Mappings:**
#          - "Minor" (Mineur) → Treat as "Dependent Child" (Enfant à charge).
#          - "Hospitalized" → Treat as "Therapeutic Institution" (Séjour thérapeutique).
#          - "Family of N" → Treat as "Head of Household + (N-1) Dependents" (unless text defines otherwise).

#     3. **SPECIFICITY PRIORITY (The "Exception" Rule):**
#        - **RULE:** Specific status overrides general status.
#        - **ACTION:** Before applying a "General Adult" rate, scan specifically for the User's attributes (e.g., "Student", "Apprentice", "Intern").
#        - If a specific reduction or rule exists for that status (e.g., "Students get 70%"), you MUST apply it.

#     4. **INPUT VALIDATION (The "Gross vs Net" Rule):**
#        - Check the variable types.
#        - **RULE:** If the text formula requires "Net Income" and the user provides "Gross Income", do NOT calculate using the Gross number. Unless a conversion formula exists in the text, return "Not Applicable".

#     5. **FORMULA EXECUTION (The "Math" Rule):**
#        - If the text defines a method (e.g., "Base + Increment" or "Income - Deduction"), you MUST perform the math.
#        - **Family Extrapolation:** If a table stops at size N (e.g., 4 people) but implies a rule for "each additional person," calculate the value for the User's requested size (e.g., 5).

#     6. **GLOBAL CONSTRAINTS (The "Ceiling" Rule):**
#        - A calculation is never finished until you check for Limits.
#        - **RULE:** Scan the text for "Maximum," "Ceiling," or "Plafond" that applies to the specific category (Family, Rent, Aid).
#        - *Logic:* Final Answer = MIN(Calculated Amount, Cap Amount).
#     </execution_algorithm>

#     <response_format>
#     You MUST use this exact format. Do not speak in paragraphs.

#     STEP 1 - REASONING:
#     - Scope Identified: [e.g., Standard Regime (ignored Emergency section)]
#     - Entity/Status Mapped: [e.g., Mapped "Minor" to "Child"]
#     - Formula Identified: [Cite the rule/text]
#     - Eligibility Check: [Pass/Fail based on age/status]
#     - Calculation Steps: [Show the math: e.g., Base 500 + Extra 100 = 600]
#     - Ceiling Check: [Is 600 > Cap 550? Yes, so result is 550]

#     STEP 2 - FINAL ANSWER:
#     [The final number or "Not Applicable"]
#     </response_format>

#     <user_question>
#     {question}
#     </user_question>
#     """
# )

    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    <system_role>
    You are an expert Legal Calculation Engine. Your goal is to extract logic from the Context and execute it to answer the User Question.
    You are a strict rule-follower. You do not summarize; you calculate.
    </system_role>

    <context_data>
    {context}
    </context_data>

    <execution_algorithm>
    To answer the question, you MUST strictly follow this 6-step algorithm. Failure to follow the order will result in a penalty.

    1. **SCOPE CHECK (The "Wrong Book" Rule):**
       - Legal texts often contain different regimes (e.g., "Standard/General" vs. "Emergency/Asylum").
       - **RULE:** If the User Question is standard, you MUST IGNORE values found in sections labeled "Emergency" (Urgence), "Asylum" (Asile), or "Exceptional Measures" unless the user explicitly asks for them.
       - *Logic:* General Question = General Section Data Only. Do not cross-contaminate limits.

    2. **ENTITY MAPPING (The "Synonym" Rule):**
       - Legal terms often vary. You must map the User's terms to the Text's definitions.
       - **Common Mappings:**
         - "Minor" (Mineur) → Treat as "Dependent Child" (Enfant à charge).
         - "Hospitalized" → Treat as "Therapeutic Institution" (Séjour thérapeutique).
         - "Family of N" → Treat as "Head of Household + (N-1) Dependents".

    3. **SPECIFICITY PRIORITY (The "Exception" Rule):**
       - **RULE:** Specific status overrides general status.
       - **ACTION:** Before applying a "General Adult" rate, scan specifically for the User's attributes (e.g., "Student", "Apprentice", "Intern").
       - If a specific reduction or rule exists for that status (e.g., "Students get 70%"), you MUST apply it.

    4. **INPUT VALIDATION (The "Gross vs Net" Rule):**
       - Check the variable types.
       - **RULE:** If the text formula requires "Net Income" and the user provides "Gross Income", do NOT calculate using the Gross number. Unless a conversion formula exists in the text, return "Not Applicable".

    5. **FORMULA EXECUTION (The "Math" Rule):**
       - If the text defines a method (e.g., "Base + Increment" or "Income - Deduction"), you MUST perform the math.
       - **Family Extrapolation:** If a table stops at size N (e.g., 4 people) but implies a rule for "each additional person," calculate the value for the User's requested size (e.g., 5).

    6. **GLOBAL CONSTRAINTS (The "Ceiling" Rule):**
       - A calculation is never finished until you check for Limits.
       - **RULE:** Scan the ENTIRE text for "Maximum," "Ceiling," or "Plafond" that applies to the specific category (Family, Rent, Aid).
       - *Logic:* Final Answer = MIN(Calculated Amount, Cap Amount).
    </execution_algorithm>

    <examples>
    Here are examples of how to apply the logic correctly:

    *Example 1: The "Scope" Logic (Standard vs Emergency)*
    User: "What is the franchise for an adult?"
    Context: [Art 14: Adult Franchise = 300 frs] [Art 67 (Emergency): Adult Franchise = 1250 frs]
    Reasoning: User did NOT ask for Emergency Aid. Ignore Art 67. Use Art 14.
    Final Answer: 300

    *Example 2: The "Exception" Logic (Student vs Adult)*
    User: "What is the maintenance for a Student?"
    Context: [Art 5: Base Maintenance = 1000] [Art 40: Student = 70% of Base]
    Reasoning: User is "Student". Specific Rule (Art 40) overrides General Rule (Art 5).
    Calculation: 1000 * 0.70 = 700.
    Final Answer: 700

    *Example 3: The "Cap" Logic (Formula vs Ceiling)*
    User: "Franchise for apprentice salary of 2000?"
    Context: [Art 15: Franchise = 1300] [Art 16: Family Cap = 1200]
    Reasoning: Calculated 1300. Found Global Cap 1200 in Art 16. Cap is lower.
    Final Answer: 1200
    </examples>

    <response_format>
    You MUST use this exact format. Do not speak in paragraphs.

    STEP 1 - REASONING:
    - Scope Identified: [e.g., Standard Regime (ignored Emergency section)]
    - Entity/Status Mapped: [e.g., Mapped "Minor" to "Child"]
    - Formula Identified: [Cite the rule/text]
    - Eligibility Check: [Pass/Fail based on age/status]
    - Calculation Steps: [Show the math: e.g., Base 500 + Extra 100 = 600]
    - Ceiling Check: [Is 600 > Cap 550? Yes, so result is 550]

    STEP 2 - FINAL ANSWER:
    [The final number or "Not Applicable"]
    </response_format>

    <user_question>
    {question}
    </user_question>
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

    output_file = LOG_DIR / f"generation_qwen3_4b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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