# PyMuPDF / Unstructured loader
from pathlib import Path
from typing import List, Dict, Any, Optional


def _read_with_pdfplumber(path: Path) -> List[str]:
    import pdfplumber

    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def _read_with_pypdf(path: Path) -> List[str]:
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_pages(path: Path) -> List[str]:
    """
    Try pdfplumber first (better layout fidelity), fall back to PyPDF2.
    """
    try:
        return _read_with_pdfplumber(path)
    except ImportError:
        pass

    try:
        return _read_with_pypdf(path)
    except ImportError:
        raise ImportError(
            "Install pdfplumber or PyPDF2 to enable PDF ingestion."
        )


def load_pdf(
    path: Path,
    per_page: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load a PDF and return a list of text blobs + metadata.
    - per_page=True returns one entry per page.
    - per_page=False returns a single concatenated entry.
    """
    pages = _extract_pages(path)
    if not pages:
        return []

    base_meta: Dict[str, Any] = {
        "source": str(path),
        "filename": path.name,
        "doc_type": "pdf",
    }
    if extra_metadata:
        base_meta.update(extra_metadata)

    if per_page:
        results: List[Dict[str, Any]] = []
        for idx, text in enumerate(pages, start=1):
            if not text.strip():
                continue
            meta = {**base_meta, "page": idx}
            results.append({"text": text, "metadata": meta})
        return results

    # Concatenate all pages
    text = "\n\n".join(pages)
    if not text.strip():
        return []
    return [{"text": text, "metadata": base_meta}]
