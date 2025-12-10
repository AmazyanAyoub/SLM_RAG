# 8B RAG Architecture – State of the Art Pipeline (2025 Edition)

## Phase 0 – Define the Playground

1. **Pick the stack**
   - **SLM (Student):** `Llama 3.1 8B Instruct` (or Qwen 2.5 7B / Mistral 8B).
   - **Teacher LLM:** GPT-4o / Claude 3.5 Sonnet / DeepSeek-V3 (for data generation/grading).
   - **Embedder:** `BGE-M3` (Dense) + `ColBERTv2` (Late Interaction) OR `Splade` (Sparse).
   - **Vector Store:** Qdrant / Weaviate / Milvus (Must support hybrid/multi-vector).
   - **Orchestrator:** **LangGraph** (Essential for cyclic agentic loops).
   - **Memory:** Mem0 (or custom vector store for user preferences).

2. **Define the domain**
   - **Data:** PDFs, HTML, Notion, Tickets.
   - **Query Types:** FAQ, Troubleshooting, Analytical, Global Summary.
   - **Success Criteria:** Grounded answers, self-correction loops, zero hallucinations.

---

## Phase 1 – Data, Contextual Indexing & Memory (The 2025 Upgrade)

3. **Ingest & Contextualize (The "Anthropic" Method)**
   - Extract text + metadata.
   - **Contextual Retrieval Step:**
     - Before chunking, pass the *entire document* to the Teacher LLM.
     - **Prompt:** "Give a 1-sentence summary of what this document is about and its key entities."
     - Prepend this summary to *every single chunk* derived from that doc.
     - *Result:* Instead of "The fee is 5%", the chunk becomes "In the 2024 Visa Card Agreement for Enterprise users, the transaction fee is 5%."

4. **Chunking Strategy**
   - Sliding window (512 tokens, 10-20% overlap).
   - **Storage Schema:**
     - `search_content`: Contextual summary + chunk text (Used for embedding).
     - `display_content`: Original raw chunk text (Used for LLM context).
     - `metadata`: `doc_id`, `section`, `created_at`, `source`.

5. **Build the Retrieval Stack**
   - **Dense Index:** Compute embeddings on `search_content` using BGE-M3.
   - **Late Interaction Index:** Compute ColBERTv2 tokens (via RAGatouille) for superior reranking performance without the slowness of Cross-Encoders.

---

## Phase 2 – User Memory Layer (Personalization)

6. **Implement Long-Term Memory (Mem0)**
   - Create a separate vector store for "User Facts".
   - **Write Path:** Analyze user chat history to extract preferences (e.g., "User is on Enterprise Plan", "User prefers Python code").
   - **Read Path:** Retrieve relevant user facts before the main search.
   - **Filter Injection:** Apply metadata filters to the main retrieval based on memory (e.g., `filter: { plan: "enterprise" }`).

---

## Phase 3 – Agentic Controller (LangGraph Implementation)

7. **The "Supervisor" Node**
   - **Input:** User Query + User Memory.
   - **Logic:** Classify intent.
     - `needs_retrieval`: Yes/No.
     - `complexity`: Simple (Vector) vs. Global (GraphRAG).

8. **The Retrieval Agent (Cyclic Loop)**
   - **Node A: Retrieve**
     - Fetch top-k chunks using Hybrid Search (Dense + ColBERT).
   - **Node B: Grade Documents**
     - Use a lightweight fine-tuned 8B (or small prompt) to score relevance.
     - *Decision:* Are at least 2 chunks relevant?
   - **Conditional Edge (The Loop):**
     - **IF** `relevance == No` AND `attempts < 3`:
       - Trigger **Query Rewriter** node (Reformulate query based on failure).
       - Loop back to **Node A**.
     - **IF** `relevance == Yes`:
       - Proceed to Generation.

---

## Phase 4 – GraphRAG for Complex Corpora

9. **Knowledge Graph Construction (Offline)**
   - Run IE (Information Extraction) with Teacher LLM to find Entities & Relations.
   - **Community Detection:** Cluster nodes using the Leiden algorithm.
   - **Summarization:** Generate summaries for every community cluster.

10. **Global Search Strategy**
    - For queries classified as "Global" (e.g., "Summarize the major updates in 2024"):
      - **Map-Reduce:**
        - Retrieve relevant community summaries.
        - Generate partial answers.
        - Collapse into a final global answer.
    - *Critical:* Do not use standard vector chunks for these queries.

---

## Phase 5 – Generation & Self-Correction (Self-RAG)

11. **Draft Generation**
    - **Input:** Original Query + Rewritten Query + Validated Chunks + Graph Summaries.
    - **Model:** Llama 3.1 8B.
    - **Prompt:** "Using the provided context, answer the query. Cite sources as [doc_id]."

12. **Hallucination Grader (Safety Valve)**
    - **Node:** Check the generated answer against the chunks.
    - **Prompt:** "Does the answer contain information NOT present in the chunks?"
    - **Edge:**
      - **IF** `Hallucination == Yes`: Retry generation with penalty instruction ("You hallucinated X, remove it.").
      - **IF** `Hallucination == No`: Stream final answer to user.

---

## Phase 6 – Distillation & Fine-Tuning (The 8B Specialist)

13. **Dataset Generation**
    - Log `(Query, Retrieved Context, Teacher Answer, Reasoning Trace)` during usage.
    - Accumulate ~1000 high-quality examples where the Teacher LLM corrected the Student.

14. **Fine-Tune the 8B Model**
    - **Objective:** Train the 8B model to imitate the Teacher's reasoning and citation behavior.
    - **Method:** LoRA (Low-Rank Adaptation).
    - **Tasks:**
      1. Relevance Grading (for Node B).
      2. Faithful Generation (for Final Answer).

---

## Phase 7 – Infrastructure & Deployment

15. **Inference Optimization**
    - **Engine:** vLLM or Ollama.
    - **Feature:** Enable **Prefix Caching** (KV Cache).
      - *Why:* System prompts and document contexts are heavy; caching reduces Time-To-First-Token (TTFT) by ~50%.

16. **Continuous Evaluation**
    - **Tools:** Ragas / DeepEval.
    - **Metrics:** Context Precision, Faithfulness, Answer Relevance.
    - **CI/CD:** Run evaluation suite on every pipeline change.

> **Final Outcome:** An 8B model that uses "Brain (Teacher) + Tools (Graph/Vector) + Memory (Mem0)" to outperform raw GPT-4 on domain-specific tasks.