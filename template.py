import os
from pathlib import Path

# ==========================================
# 2025 AGENTIC RAG ARCHITECTURE (FINAL)
# ==========================================
PROJECT_STRUCTURE = {
    "backend": {
        "app": {
            "__init__.py": "",
            "main.py": "# FastAPI entry point\n",
            "settings.py": "# Pydantic settings management\n",
            "dependencies.py": "# Dependency injection (DB, Clients)\n",
        },
        "core": {
            "__init__.py": "",
            "logging.py": "# Structured logging setup\n",
            "config_loader.py": "# YAML/Env loader\n",
            "prompts.py": "# CENTRALIZED PROMPTS (Teacher, Grader, etc.)\n",
            "utils.py": "# Async helpers, decorators\n",
        },
        "memory": {
            "__init__.py": "",
            "mem0_client.py": "# Client for Mem0 / User Facts\n",
            "user_preferences.py": "# Logic to read/write user state\n",
        },
        "ingestion": {
            "__init__.py": "",
            "loaders": {
                "__init__.py": "",
                "pdf_loader.py": "# PyMuPDF / Unstructured loader\n",
                "web_loader.py": "# Firecrawl / Beautiful Soup loader\n",
            },
            "pipeline": {
                "__init__.py": "",
                "cleaning.py": "# Text normalization\n",
                "chunking.py": "# Sliding window chunker\n",
                "contextual_enrichment.py": "# PHASE 1.5: Teacher LLM summarizer for chunks\n",
            },
        },
        "indexing": {
            "__init__.py": "",
            "vector_store.py": "# Qdrant client wrapper\n",
            "dense_index.py": "# BGE-M3 embedding logic\n",
            "colbert_index.py": "# RAGatouille / ColBERTv2 logic\n",
            "sparse_index.py": "# Splade / BM25 fallback (optional)\n",
        },
        "graph_rag": {
            "__init__.py": "",
            "graph_builder.py": "# Entity extraction & Graph construction\n",
            "community_detection.py": "# Leiden algorithm for clustering\n",
            "graph_search.py": "# Map-Reduce global search logic\n",
        },
        "langgraph_flow": {
            "__init__.py": "",
            "state.py": "# GraphState (query, docs, attempts, hallu_score)\n",
            "graph_builder.py": "# Main StateGraph definition (Edges & Routing)\n",
            "nodes": {
                "__init__.py": "",
                "supervisor.py": "# ROUTER: Decides Vector vs Graph vs Web\n",
                "retrieve.py": "# TOOL: Runs Hybrid Retrieval\n",
                "grade_documents.py": "# AGENT: 8B model checks relevance\n",
                "rewrite_query.py": "# LOOP: Fixes bad queries\n",
                "generate.py": "# GENERATOR: Final answer synthesis\n",
                "hallucination_check.py": "# SELF-CORRECTION: Checks answer vs docs\n",
            },
        },
        "models": {
            "__init__.py": "",
            "llm_factory.py": "# Returns Student (Ollama) or Teacher (Groq) client\n",
            "embeddings.py": "# Embedding model wrapper\n",
        },
        "distillation": {
            "__init__.py": "",
            "logger.py": "# Logs traces for training data\n",
            "dataset_builder.py": "# Converts logs to Preference Datasets\n",
            "fine_tuner.py": "# LoRA training script wrapper\n",
        },
        "evaluation": {
            "__init__.py": "",
            "ragas_runner.py": "# DeepEval / Ragas integration\n",
            "benchmarks.py": "# Test set definitions\n",
        },
        "tests": {
            "__init__.py": "",
            "test_ingestion.py": "",
            "test_retrieval.py": "",
            "test_agentic_flow.py": "# End-to-end graph tests\n",
        },
    },
    "frontend": {
        "streamlit_app.py": "# Main UI entry point\n",
        "components": {
            "chat_interface.py": "# Chat bubbles & history\n",
            "sidebar.py": "# Settings & Debug controls\n",
        },
        "pages": {
            "1_Chat_RAG.py": "",
            "2_Doc_Upload.py": "",
            "3_Evaluation.py": "",
        },
    },
    "configs": {
        "base.yaml": "# Main config (Copy the one I gave you here)\n",
        ".env.example": "OPENAI_API_KEY=\nGROQ_API_KEY=\nQDRANT_API_KEY=\n",
    },
    "data": {
        "raw": {},
        "processed": {},
        "indices": {
             "qdrant": {},
             "colbert": {},
             "graph": {},
        },
    },
    "scripts": {
        "ingest.py": "# CLI to run ingestion\n",
        "build_graph.py": "# CLI to build GraphRAG\n",
        "train_student.py": "# CLI to start LoRA fine-tuning\n",
    },
    "README.md": "# 2025 RAG Documentation\n",
    "requirements.txt": "langgraph\nlangchain\nqdrant-client\nragatouille\nolama\nstreamlit\n",
    "docker-compose.yaml": "\n",
}


def create_structure(base_path: Path, structure: dict) -> None:
    """
    Recursively create directories and files.
    SAFETY CHECK: If a file exists and is not empty, it renames the OLD file
    to 'filename_backup.ext' before creating the new template file.
    """
    for name, content in structure.items():
        path = base_path / name

        if isinstance(content, dict):
            # Directory
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            # File
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists and has content
            if path.exists() and path.stat().st_size > 0:
                # Create a backup name (e.g., main_backup.py)
                backup_name = f"{path.stem}_backup{path.suffix}"
                backup_path = path.parent / backup_name
                
                # Check if the backup already exists to avoid overwriting IT
                counter = 1
                while backup_path.exists():
                    backup_name = f"{path.stem}_backup_{counter}{path.suffix}"
                    backup_path = path.parent / backup_name
                    counter += 1
                
                # Rename the existing file to the backup name
                path.rename(backup_path)
                print(f"⚠️  Existing code found in '{name}'. Renamed to '{backup_name}'.")

            # Write the new template file (if it was empty, it just overwrites)
            path.write_text(content or "", encoding="utf-8")


def main():
    project_root = Path(".")
    project_root.mkdir(exist_ok=True)
    create_structure(project_root, PROJECT_STRUCTURE)
    print(f"\n✅ 2025 SOTA Project structure created under: {project_root.resolve()}")
    print("👉 Any existing files with code were renamed with '_backup' suffix.")


if __name__ == "__main__":
    main()