# Embedding model client
from functools import lru_cache
from typing import List, Literal, Dict, Any, Tuple
import os

from sentence_transformers import SentenceTransformer

# Map our simple names -> real HF IDs
EMBEDDING_NAME_TO_HF_ID: Dict[str, str] = {
    "bge-large": "BAAI/bge-large-en-v1.5",  # recommended BGE version for retrieval :contentReference[oaicite:1]{index=1}
    "gte-large": "thenlper/gte-large",
    "bge-m3": "BAAI/bge-m3"
}


def _get_embedding_choice() -> Tuple[str, str]:
    """
    Returns (hf_model_id, short_name) for the embedding model.
    Controlled by EMBEDDING_MODEL_NAME env var (bge-large | gte-large).
    """
    short_name = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")
    hf_id = EMBEDDING_NAME_TO_HF_ID.get(short_name)
    if hf_id is None:
        raise ValueError(
            f"Unsupported EMBEDDING_MODEL_NAME={short_name!r}. "
            f"Allowed: {list(EMBEDDING_NAME_TO_HF_ID.keys())}"
        )
    return hf_id, short_name


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    hf_id, _ = _get_embedding_choice()
    # Lazy-load & cache the model in memory
    return SentenceTransformer(hf_id)


def _apply_bge_query_prefix(texts: List[str]) -> List[str]:
    # For BGE, official guidance is to prefix queries to get best retrieval performance :contentReference[oaicite:2]{index=2}
    instruction = "Represent this sentence for searching relevant passages: "
    return [instruction + t for t in texts]


def embed_queries(texts: List[str]) -> List[List[float]]:
    """
    Embed user queries for retrieval.
    """
    if not texts:
        return []

    model = get_sentence_transformer()
    _, short_name = _get_embedding_choice()

    processed = texts

    if short_name == "bge-large":
        processed = _apply_bge_query_prefix(texts)

    embeddings = model.encode(
        processed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed document chunks for indexing.
    """
    if not texts:
        return []

    model = get_sentence_transformer()
    # For documents we usually don't add the BGE query prefix; straight encoding works well.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()
