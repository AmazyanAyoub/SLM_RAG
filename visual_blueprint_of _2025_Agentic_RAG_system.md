[ PDF DOCUMENT ]
       │
       ▼
┌───────────────────────┐
│   Step 1: LOAD        │
│  (PyMuPDF / Loader)   │
└──────────┬────────────┘
           │ Raw Text
           ▼
┌────────────────────────────────────────────────────────┐
│  Step 2: CONTEXTUAL ENRICHMENT (Phase 1.5)             │
│  (Teacher LLM: "Summary of this doc is...")            │
└──────────┬─────────────────────────────────────────────┘
           │ <Global Context> + <Raw Text>
           ▼
┌───────────────────────┐
│   Step 3: CHUNK       │
│   (Sliding Window)    │
└──────────┬────────────┘
           │ [Chunk 1], [Chunk 2], ...
           │
     ┌─────┴──────────────────────────────┐
     ▼                                    ▼
┌──────────────────┐             ┌──────────────────┐
│  DENSE EMBED     │             │  SPARSE INDEX    │
│  (BGE-M3 Model)  │             │  (ColBERT /      │
│                  │             │   RAGatouille)   │
└────────┬─────────┘             └────────┬─────────┘
         │ Vectors                        │ Tokens
         │                                │
         ▼                                ▼
    ┌──────────────────────────────────────────┐
    │  QDRANT VECTOR STORE                     │
    │  (Collection: `slm_rag_contextual`)      │
    └──────────────────────────────────────────┘


USER QUERY: "Why is my bill higher?"
       │
       ▼
┌───────────────────────────────────────┐
│  MEMORY NODE (Mem0)                   │
│  "User is on Enterprise Plan..."      │
└──────────────┬────────────────────────┘
               │ Query + User Context
               ▼
┌───────────────────────────────────────┐
│  SUPERVISOR AGENT (The Brain)         │
│  "Does this need Graph or Vector?"    │
└──────┬───────────────────────┬────────┘
       │ (Simple Fact)         │ (Global Summary)
       │                       │
       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  VECTOR AGENT    │    │   GRAPH AGENT    │
└──────┬───────────┘    └──────┬───────────┘
       │                       │
       │ <──[LOOP START]       │
       ▼                       │
  [RETRIEVE]                   │
       │                       │
       ▼                       │
  [GRADE DOCS] ──(No) ──┐      │
       │                │      │
     (Yes)           [REWRITE] │
       │                │      │
       ▼                │      │
  [GENERATE] ◄──────────┘      │
       │                       │
       ▼                       │
[HALLUCINATION CHECK]          │
       │ (Fail)                │
       └─(Loop Back)           │
       │                       │
       ▼ (Pass)                ▼
┌───────────────────────────────────────┐
│        FINAL ANSWER TO USER           │
└───────────────────────────────────────┘



