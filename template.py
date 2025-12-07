from pathlib import Path

PROJECT_STRUCTURE = {
    "backend": {
        "app": {
            "__init__.py": "",
            "main.py": "# Entry point for backend API\n",
            "settings.py": "# Settings and configuration\n",
            "dependencies.py": "# Dependency wiring (DB, clients, etc.)\n",
        },
        "core": {
            "__init__.py": "",
            "logging.py": "# Logging configuration\n",
            "config_loader.py": "# Load YAML/ENV configs\n",
            "utils.py": "# Shared helper functions\n",
        },
        "ingestion": {
            "__init__.py": "",
            "loaders": {
                "__init__.py": "",
                "pdf_loader.py": "# PDF loader\n",
                "web_loader.py": "# Web page loader\n",
                "generic_loader.py": "# Generic loader\n",
            },
            "preprocessing": {
                "__init__.py": "",
                "cleaning.py": "# Text cleaning\n",
                "metadata_enrichment.py": "# Add metadata\n",
            },
        },
        "indexing": {
            "__init__.py": "",
            "chunking.py": "# Chunking logic\n",
            "embeddings_bge_gte.py": "# BGE/GTE embedding client\n",
            "dense_index.py": "# Dense index setup\n",
            "sparse_index.py": "# Sparse/BM25 index setup\n",
            "hybrid_index.py": "# Hybrid index (dense + sparse)\n",
        },
        "retrieval": {
            "__init__.py": "",
            "retriever_hybrid.py": "# Hybrid retriever\n",
            "reranker_monoT5_bge.py": "# monoT5/BGE reranker\n",
            "query_rewriter.py": "# Query rewriting\n",
            "context_builder.py": "# Build context for generation\n",
        },
        "graph_rag": {
            "__init__.py": "",
            "graph_builder.py": "# Build knowledge graph\n",
            "community_detection.py": "# Community detection\n",
            "graph_store.py": "# Graph store abstraction\n",
            "graph_retrieval.py": "# Graph-based retrieval\n",
        },
        "controller": {
            "__init__.py": "",
            "query_classifier.py": "# Classify query type\n",
            "retrieval_planner.py": "# Plan retrieval strategy\n",
            "self_rag_controller.py": "# Self-RAG orchestration\n",
            "reflection_loop.py": "# Reflection / self-check loop\n",
            "routing.py": "# Route requests through pipeline\n",
        },
        "models": {
            "__init__.py": "",
            "llama3_8b_client.py": "# 8B model client\n",
            "teacher_llm_client.py": "# Teacher LLM client\n",
            "embedding_client.py": "# Embedding model client\n",
        },
        "distillation": {
            "__init__.py": "",
            "dataset_builder.py": "# Build distillation datasets\n",
            "teacher_runner.py": "# Run teacher RAG system\n",
            "student_trainer.py": "# Train student (8B) model\n",
            "drag_kard_utils.py": "# DRAG/KARD helpers\n",
        },
        "evaluation": {
            "__init__.py": "",
            "metrics.py": "# Metrics for RAG quality\n",
            "eval_runner.py": "# Run evaluation suites\n",
            "reports.py": "# Generate eval reports\n",
        },
        "langgraph_flow": {
            "__init__.py": "",
            "state.py": "# Shared state definition\n",
            "nodes": {
                "__init__.py": "",
                "node_query_classify.py": "# Node: classify query\n",
                "node_retrieve.py": "# Node: retrieval\n",
                "node_graph_rag.py": "# Node: GraphRAG\n",
                "node_generate.py": "# Node: generation\n",
                "node_reflect.py": "# Node: reflection\n",
                "node_eval_logging.py": "# Node: eval/logging\n",
            },
            "graph_builder.py": "# Build LangGraph graph\n",
        },
        "tests": {
            "__init__.py": "",
            "test_retrieval.py": "# Tests for retrieval\n",
            "test_graph_rag.py": "# Tests for GraphRAG\n",
            "test_controller.py": "# Tests for controller logic\n",
            "test_end_to_end.py": "# End-to-end tests\n",
        },
    },
    "frontend": {
        "streamlit_app.py": "# Main Streamlit entry point\n",
        "pages": {
            "1_Chat_RAG.py": "# Chat with RAG\n",
            "2_Doc_Upload.py": "# Document upload page\n",
            "3_Evaluation_Dashboard.py": "# Evaluation dashboard\n",
            "4_Admin_Teacher_Distillation.py": "# Admin / distillation controls\n",
        },
    },
    "configs": {
        "base.yaml": "# Base configuration\n",
        "dev.yaml": "# Dev overrides\n",
        "prod.yaml": "# Prod overrides\n",
        "logging.yaml": "# Logging configuration\n",
    },
    "data": {
        "raw": {},
        "processed": {},
        "chunks": {},
        "indices": {},
        "graphs": {},
        "eval_sets": {},
    },
    "scripts": {
        "ingest_data.py": "# Script: ingest documents\n",
        "build_indices.py": "# Script: build indices\n",
        "build_graph.py": "# Script: build knowledge graph\n",
        "run_teacher_rag.py": "# Script: run teacher RAG\n",
        "run_distillation.py": "# Script: run distillation training\n",
        "run_eval.py": "# Script: run evaluation suite\n",
    },
    "notebooks": {
        "exploration_ingestion.ipynb": "",
        "retrieval_tuning.ipynb": "",
        "graph_rag_prototyping.ipynb": "",
        "distillation_experiments.ipynb": "",
    },
    "infra": {
        "docker": {
            "Dockerfile.backend": "# Backend Dockerfile\n",
            "Dockerfile.streamlit": "# Streamlit Dockerfile\n",
            "docker-compose.yaml": "# docker-compose for local dev\n",
        },
        "k8s": {
            "backend_deployment.yaml": "# K8s deployment for backend\n",
            "backend_service.yaml": "# K8s service for backend\n",
            "frontend_deployment.yaml": "# K8s deployment for frontend\n",
            "frontend_service.yaml": "# K8s service for frontend\n",
        },
    },
    ".env.example": "# Example environment variables\n",
    "pyproject.toml": "# Project configuration (optional)\n",
    "requirements.txt": "# Python dependencies\n",
    "README.md": "# RAG project README\n",
}


def create_structure(base_path: Path, structure: dict) -> None:
    """
    Recursively create directories and files based on a nested dict structure.
    - Keys are names (files or directories).
    - Values:
        - dict  -> directory (recursively processed)
        - str   -> file content (created as file)
        - None  -> empty file
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
            if not path.exists():
                path.write_text(content or "", encoding="utf-8")


def main():
    project_root = Path(".")
    project_root.mkdir(exist_ok=True)
    create_structure(project_root, PROJECT_STRUCTURE)
    print(f"Project structure created under: {project_root.resolve()}")


if __name__ == "__main__":
    main()
