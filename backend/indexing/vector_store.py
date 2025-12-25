import os
from backend.indexing.qdrant_client import QdrantVectorDB
from backend.indexing.postgres_client import PostgresVectorDB

class VectorDBClient:
    def __init__(self):
        """
        Factory Class: Initializes either Qdrant or Postgres based on configuration.
        """
        # Check environment variable or settings for provider
        self.provider = os.getenv("VECTOR_DB_PROVIDER", "qdrant").lower()
        print(f"🚀 Initializing Vector DB Provider: {self.provider.upper()}")
        
        if self.provider == "postgres":
            self.client = PostgresVectorDB()
        else:
            self.client = QdrantVectorDB()

    def upsert(self, points: list):
        return self.client.upsert(points)

    def search(self, query_text: str, limit: int = 5):
        return self.client.search(query_text, limit)

if __name__ == "__main__":
    print("--- TEST: VectorDBClient Factory ---")
    try:
        # This will initialize whichever provider is set in .env (VECTOR_DB_PROVIDER)
        client = VectorDBClient()
        print(f"✅ Successfully initialized provider: {client.provider.upper()}")
        
        # Simple connectivity check (assuming data exists or just checking for no errors)
        results = client.search("test query", limit=1)
        print(f"✅ Search executed. Found {len(results)} results.")
    except Exception as e:
        print(f"❌ Factory Test failed: {e}")