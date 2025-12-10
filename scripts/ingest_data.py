# Script: ingest documents
import os
from pathlib import Path
from typing import List, Dict, Any

from backend.indexing.dense_index import add_texts


RAW_DIR = Path("data/raw")


def load_txt_files() -> List[Dict[str, Any]]:
    docs = []
    if not RAW_DIR.exists():
        print(f"[ingest] RAW_DIR {RAW_DIR} does not exist, creating it.")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        return docs

    for path in RAW_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        docs.append(
            {
                "text": text,
                "metadata": {
                    "source": str(path),
                    "filename": path.name,
                },
            }
        )
    return docs


def main():
    docs = load_txt_files()
    if not docs:
        print(
            "[ingest] No .txt files found in data/raw. "
            "Add some text files and rerun."
        )
        return

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    print(f"[ingest] Indexing {len(texts)} documents into Chroma...")
    add_texts(texts=texts, metadatas=metadatas)
    print("[ingest] Done.")


if __name__ == "__main__":
    main()
