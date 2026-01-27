import os
import sys
import asyncio
import time
import uuid
# from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.indexing.postgres_client import PostgresVectorDB
from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.pipeline.chunking import Chunker
from backend.ingestion.pipeline.contextual_enrichment import ContextualEnricher
from backend.models.embedding_client import embed_documents, embed_sparse

# Load Environment Variables
load_dotenv()

async def main():
    print("🚀 STARTING POSTGRES INGESTION PIPELINE")
    print("============================================================")
    
    # Force Postgres for this script
    os.environ["VECTOR_DB_PROVIDER"] = "postgres"

    # 1. SETUP
    data_dir = Path("data/pdfs")
    if not data_dir.exists():
        print(f"❌ Error: Directory '{data_dir}' not found.")
        return

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDFs found in {data_dir}.")
        return

    # Initialize Components
    # Use Docling strategy to match Qdrant script
    chunker = Chunker(strategy="docling", chunk_size=1024, chunk_overlap=200)
    
    # Initialize DB
    try:
        db = PostgresVectorDB()
        print(f"🔌 Connected to Postgres Table: '{db.table_name}'")
    except Exception as e:
        print(f"❌ Failed to connect to Postgres: {e}")
        return

    try:
        enricher = ContextualEnricher()
        print("✅ Teacher LLM Connected for Enrichment.")
    except Exception as e:
        enricher = None
        print(f"⚠️ Contextual Enricher skipped: {e}")

    # 2. PROCESS FILES
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        
        # A. CHUNK (Docling)
        chunks = []
        try:
            if chunker.strategy == "docling":
                print("   🧠 Using Docling Smart Chunking...")
                chunks = chunker.chunk(file_path=str(pdf_file), metadata={"source": pdf_file.name})
            else:
                print("   ⚠️ Only Docling strategy is fully supported in this script version.")
                continue
        except Exception as e:
            print(f"   ❌ Failed to chunk: {e}")
            continue

        print(f"   ✂️ Generated {len(chunks)} chunks.")
        enricher = None

        # B. ENRICH
        if enricher and chunks:
            print("   👨‍🏫 Enriching chunks...")
            tasks = []
            window_size = 3
            for i, chunk in enumerate(chunks):
                start_i = max(0, i - window_size)
                end_i = min(len(chunks), i + window_size + 1)
                neighbor_text = "\n---\n".join([c["text"] for c in chunks[start_i:end_i]])
                tasks.append(enricher.enrich_chunk(chunk["text"], neighbor_text))
            
            # Batch execution
# {
# "text":"3 La mesure est également ouverte aux personnes n'ayant pas fait de stage d'évaluation ou qui ne sont pas suivies par le service de réinsertion professionnelle. 4 Le projet de formation, élaboré dans le cadre du contrat d'aide sociale individuel, tient compte des aptitudes du bénéficiaire et des débouchés offerts par le marché de l'emploi. Il est examiné par une commission d'attribution désignée par l'Hospice général qui se prononce sur sa pertinence et son adéquation socioéconomique, ainsi que sur l'octroi et le montant de l'allocation. 5 La décision est notifiée par l'Hospice général qui est lié par l'avis de la commission d'attribution ainsi que par les montants déterminés par celle-ci."
# "metadata":{
# "source":"LIASI - Règlement d'application - 19-06-2007 - 31-12-2024.pdf"
# "hierarchy_path":"Chapitre V (7) Insertion professionnelle / Section 3 (7) Lien avec les mesures cantonales en matière de chômage / Art. 23F (18) Formation de base de courte durée"
# "chunk_index":86
# "total_chunks":126
# }
# "search_content":"3 La mesure est également ouverte aux personnes n'ayant pas fait de stage d'évaluation ou qui ne sont pas suivies par le service de réinsertion professionnelle. 4 Le projet de formation, élaboré dans le cadre du contrat d'aide sociale individuel, tient compte des aptitudes du bénéficiaire et des débouchés offerts par le marché de l'emploi. Il est examiné par une commission d'attribution désignée par l'Hospice général qui se prononce sur sa pertinence et son adéquation socioéconomique, ainsi que sur l'octroi et le montant de l'allocation. 5 La décision est notifiée par l'Hospice général qui est lié par l'avis de la commission d'attribution ainsi que par les montants déterminés par celle-ci."
# "display_content":"3 La mesure est également ouverte aux personnes n'ayant pas fait de stage d'évaluation ou qui ne sont pas suivies par le service de réinsertion professionnelle. 4 Le projet de formation, élaboré dans le cadre du contrat d'aide sociale individuel, tient compte des aptitudes du bénéficiaire et des débouchés offerts par le marché de l'emploi. Il est examiné par une commission d'attribution désignée par l'Hospice général qui se prononce sur sa pertinence et son adéquation socioéconomique, ainsi que sur l'octroi et le montant de l'allocation. 5 La décision est notifiée par l'Hospice général qui est lié par l'avis de la commission d'attribution ainsi que par les montants déterminés par celle-ci."

        # C. INDEX
        if chunks:
            print("   🧠 Generating Embeddings...")
            try:
                search_texts = [c.get("search_content", c["text"]) for c in chunks]
                dense_vectors = embed_documents(search_texts)
                sparse_vectors = embed_sparse(search_texts)

                points = []
                for i, chunk in enumerate(chunks):
                    source = chunk["metadata"].get("source", "unknown")
                    index = chunk["metadata"].get("chunk_index", i)
                    signature = f"{source}_{index}"
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, signature))
                    
                    sp_indices = list(int(k) for k in sparse_vectors[i].keys())
                    sp_values = list(float(v) for v in sparse_vectors[i].values())

                    points.append({
                        "payload": {**chunk, "id": point_id},
                        "vector": {
                            "dense": dense_vectors[i],
                            "sparse": {"indices": sp_indices, "values": sp_values}
                        }
                    })
                
                print(f"   📤 Upserting {len(points)} points to Postgres...")
                db.upsert(points)
                print(f"   ✅ File '{pdf_file.name}' Fully Indexed!")

            except Exception as e:
                print(f"   ❌ Indexing Failed: {e}")

    print("🎉 POSTGRES INGESTION COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())
