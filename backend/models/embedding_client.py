# Embedding model client
from functools import lru_cache
from typing import List, Literal, Dict, Any, Tuple
import os

from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download

from backend.core.config_loader import settings

# Map our simple names -> real HF IDs
EMBEDDING_NAME_TO_HF_ID: Dict[str, str] = {
    "bge-large": "BAAI/bge-large-en-v1.5",  # recommended BGE version for retrieval :contentReference[oaicite:1]{index=1}
    "gte-large": "thenlper/gte-large",
    "bge-m3": "BAAI/bge-m3"
}


def _get_embedding_choice() -> Tuple[str, str]:
    """
    Returns (hf_model_id, short_name) for the embedding model.
    Controlled by settings or EMBEDDING_MODEL_NAME env var.
    """
    # Prioritize settings, then env var, then default
    if settings and settings.retrieval and settings.retrieval.embedder_model:
        short_name = settings.retrieval.embedder_model
    else:
        short_name = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")
        
    hf_id = EMBEDDING_NAME_TO_HF_ID.get(short_name)
    if hf_id is None:
        # Fallback: if the name isn't in our map, maybe it's a direct ID or we default?
        # For safety, let's raise error to avoid silent failures or mismatch
        raise ValueError(
            f"Unsupported embedding model name={short_name!r}. "
            f"Allowed: {list(EMBEDDING_NAME_TO_HF_ID.keys())}"
        )
    return hf_id, short_name


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    hf_id, _ = _get_embedding_choice()
    # Lazy-load & cache the model in memory
    return SentenceTransformer(hf_id, device="cuda")


@lru_cache(maxsize=1)
def get_sparse_model() -> BGEM3FlagModel:
    """
    Loads the BGE-M3 model specifically for sparse encoding.
    """
    hf_id, short_name = _get_embedding_choice()
    
    # We enforce BGE-M3 or compatible for sparse. 
    # If a different model is chosen but sparse is requested, this might need adjustment.
    # For now, we assume the user intends to use BGE-M3 features if asking for sparse.
    if short_name != "bge-m3":
        print(f"⚠️ Warning: Sparse embedding requested but model is {short_name}. Loading BGE-M3 for sparse.")
        hf_id = EMBEDDING_NAME_TO_HF_ID["bge-m3"]

    print(f"🧠 Loading Sparse Model: {hf_id}...")
    
    # SMART DOWNLOAD: Download ONLY what we need (Skip the 2GB ONNX file)
    local_path = snapshot_download(
        repo_id=hf_id,
        ignore_patterns=["*.onnx", "*.onnx_data", "flax_model.msgpack", "rust_model.ot", "pytorch_model.bin"],
        resume_download=True
    )
    
    return BGEM3FlagModel(model_name_or_path=local_path, use_fp16=True, device="cuda")


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


def embed_sparse(texts: List[str]) -> List[Dict[str, float]]:
    """
    Generates sparse vectors (lexical weights) for a list of texts using BGE-M3.
    """
    if not texts:
        return []
        
    model = get_sparse_model()
    output = model.encode(
        texts, 
        return_dense=False, 
        return_sparse=True, 
        return_colbert_vecs=False
    )
    # Returns a list of dictionaries: [{'word_id': weight}, ...]
    return output['lexical_weights']
