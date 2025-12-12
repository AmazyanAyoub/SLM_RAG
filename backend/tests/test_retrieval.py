from backend.indexing.dense_index import DenseIndexer
from backend.indexing.vector_store import VectorDBClient

def test_query_manual(query: str):
    print(f"\n🔎 Testing Query: '{query}'")
    print("=" * 60)

    # 1. Initialize (Loads BGE-M3 model again - takes a few seconds)
    try:
        indexer = DenseIndexer()
        client = VectorDBClient()
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Embed the Query
    print("🧠 Embedding query...")
    # returns a list of vectors, we take the first one [0]
    query_vector = indexer.embed_texts([query])[0]

    # 3. Search Qdrant
    print("🔌 Searching Qdrant...")
    results = client.search(query_vector, limit=5)

    if not results:
        print("⚠️ No results found. Is the DB empty?")
        return

    # 4. Print Results with Scores
    print(f"\n✅ Top {len(results)} Matches:")
    for i, match in enumerate(results):
        score = match.score
        # Metadata payload
        text = match.payload.get("text", "No text found")
        source = match.payload.get("source", "Unknown")
        
        # Color code the score
        score_icon = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
        
        print(f"\n[{i+1}] Score: {score:.4f} {score_icon}")
        print(f"    📄 Source: {source}")
        print(f"    💬 Content: {text[:200]}...") # Show first 200 chars
        print("-" * 40)

if __name__ == "__main__":
    # You can change this question to something specific from your PDF
    TEST_QUESTION = "Quelles sont les conditions d'admission?" 
    
    # Or uncomment this to type it in the terminal:
    # TEST_QUESTION = input("Enter your query: ")
    
    test_query_manual(TEST_QUESTION)