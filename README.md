# 2025 RAG Documentation
# SLM RAG project README - Project Overview

## 1. Purpose

This project implements a Retrieval-Augmented Generation (RAG) system where a **small Groq-hosted 8B model** behaves as strongly as possible for domain-specific tasks by combining:

- Strong retrieval (BGE/GTE embeddings + Chroma),
- Hybrid search and reranking,
- Self-RAG style control (query classification, dynamic retrieval, reflection),
- Optional GraphRAG for complex corpora,
- Distillation from a stronger 70B teacher model,
- Continuous evaluation and self-improvement.

The final user-facing interface is a **Streamlit app** that lets users upload documents, query the knowledge base, and inspect evaluation / distillation runs.

---

## 2. Core Components

- **Student LLM (Groq)**
  - Default: `llama-3.1-8b-instant` (fast, small).
- **Teacher LLM (Groq)**
  - Default: `llama-3.3-70b-versatile` (strong general model, used for distillation and evaluation).
- **Embeddings**
  - Local HuggingFace models:
    - `bge-large` (default) or
    - `gte-large` (optional).
- **Vector Store**
  - `Chroma` for dense retrieval (stored locally under `data/indices/chroma`).
- **Frameworks**
  - **LangChain** + **LangGraph** for orchestration and graph-based workflows.
  - **Streamlit** for the frontend (chat UI, document upload, evaluation dashboards).
  - Standard Python stack for ingestion, indexing, and evaluation.

---

## 3. High-Level Flow

1. **Ingestion & Indexing**
   - Load PDFs / web pages / docs.
   - Clean and chunk text (sliding window).
   - Embed chunks with BGE or GTE.
   - Store in Chroma with metadata.

2. **Retrieval**
   - Hybrid retrieval (BM25 later + dense via Chroma).
   - Optional reranking (e.g. monoT5/BGE-reranker).

3. **RAG Generation (Student 8B)**
   - Query classification (needs retrieval? factoid vs analytical vs global).
   - Dynamic retrieval plan (k, mode).
   - Build context + prompt.
   - Generate answer with `llama-3.1-8b-instant` via Groq.
   - Reflection/self-check pass to reduce hallucinations.

4. **GraphRAG (Optional)**
   - Build a knowledge graph and community summaries.
   - Use graph-based retrieval for global / multi-doc queries.

5. **Teacher Distillation**
   - Run the same pipeline with `llama-3.3-70b-versatile` as teacher.
   - Collect (query, context, teacher answer [+ rationale]) pairs.
   - Fine-tune or LoRA the 8B student to imitate teacher behavior.

6. **Evaluation & Self-Improvement**
   - Evaluation set with gold answers and sources.
   - Compare student vs teacher performance.
   - Use failures to generate new distillation data and improve the student.

---

## 4. Frontend (Streamlit)

- **Chat with RAG**: ask questions, get grounded answers with citations.
- **Document Upload**: add new PDFs/docs to the knowledge base.
- **Evaluation Dashboard**: visualize metrics (accuracy, hallucination rate, citation quality).
- **Admin / Distillation Controls**: run teacher RAG, build datasets, trigger student fine-tuning (later).

---