from typing import List, Callable, Optional
import re


DEFAULT_ENCODING = "cl100k_base"


def _get_tokenizer(encoding_name: str = DEFAULT_ENCODING):
    """
    Best-effort tokenizer getter. Uses tiktoken if available; falls back to None.
    """
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding(DEFAULT_ENCODING)


def _token_len_fn(encoding) -> Callable[[str], int]:
    if encoding is None:
        # Fallback: approximate length by whitespace tokens
        return lambda text: len(text.split())
    return lambda text: len(encoding.encode(text))


def _split_sentences(text: str) -> List[str]:
    # Simple regex-based sentence splitter; avoids heavyweight deps.
    parts = re.split(r"(?<=[\.!\?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p and p.strip()]


def _semantic_units(text: str, max_len: int, length_fn: Callable[[str], int]) -> List[str]:
    """
    Split text into semantically meaningful units (paragraphs -> sentences) that
    are at most max_len tokens (or shorter). Longer units are left for the
    downstream splitter to handle at token level.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p and p.strip()]
    units: List[str] = []

    for para in paragraphs:
        if length_fn(para) <= max_len:
            units.append(para)
            continue

        sentences = _split_sentences(para)
        for sent in sentences:
            if sent and length_fn(sent) > 0:
                units.append(sent)

    if not units and text.strip():
        units.append(text.strip())

    return units


def _split_long_unit(unit: str, chunk_size: int, chunk_overlap: int, encoding, length_fn: Callable[[str], int]) -> List[str]:
    """
    Token-based split for a single long unit.
    """
    if encoding is None:
        # Approximate using words
        words = unit.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            start = end - chunk_overlap
        return chunks

    tokens = encoding.encode(unit)
    chunks: List[str] = []
    start = 0
    total = len(tokens)

    while start < total:
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end >= total:
            break
        start = end - chunk_overlap

    return chunks


def _tail_overlap(text: str, overlap_tokens: int, encoding, length_fn: Callable[[str], int]) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    if encoding is None:
        words = text.split()
        if len(words) <= overlap_tokens:
            return text
        return " ".join(words[-overlap_tokens:])

    tokens = encoding.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    return encoding.decode(tokens[-overlap_tokens:])


def split_text(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    encoding_name: Optional[str] = None,
) -> List[str]:
    """
    Token-aware, semantic-first chunking.
    - Prefer splitting by paragraphs, then sentences.
    - Enforce token-based chunk_size with token overlap.
    - Falls back to word-based approximation if tiktoken is unavailable.
    """
    if not text:
        return []

    encoding = _get_tokenizer(encoding_name or DEFAULT_ENCODING)
    length_fn = _token_len_fn(encoding)

    units = _semantic_units(text, chunk_size, length_fn)

    chunks: List[str] = []
    current_parts: List[str] = []

    def flush_current():
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append(chunk_text)

    current_tokens = 0

    for unit in units:
        unit_len = length_fn(unit)
        if unit_len == 0:
            continue

        # If this unit alone is too large, split it token-wise.
        if unit_len > chunk_size:
            flush_current()
            current_parts.clear()
            current_tokens = 0

            long_chunks = _split_long_unit(unit, chunk_size, chunk_overlap, encoding, length_fn)
            chunks.extend(long_chunks)

            # Prepare overlap seed for next chunk
            if chunks:
                seed = _tail_overlap(chunks[-1], chunk_overlap, encoding, length_fn)
                current_parts = [seed] if seed else []
                current_tokens = length_fn(seed) if seed else 0
            continue

        if current_tokens + unit_len <= chunk_size:
            current_parts.append(unit)
            current_tokens += unit_len
            continue

        # Finalize current chunk and start a new one with overlap
        flush_current()
        last_chunk = chunks[-1] if chunks else ""
        overlap_seed = _tail_overlap(last_chunk, chunk_overlap, encoding, length_fn)
        current_parts = [overlap_seed] if overlap_seed else []
        current_tokens = length_fn(overlap_seed) if overlap_seed else 0

        current_parts.append(unit)
        current_tokens += unit_len

    flush_current()
    return [c for c in chunks if c]
