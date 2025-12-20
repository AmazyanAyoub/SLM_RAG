from langchain_core.documents import Document
from backend.langgraph_flow.state import GraphState
from backend.indexing.vector_store import VectorDBClient

def retrieve(state: GraphState):
    """
    Retrieve documents based on the current question in the state.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updates 'documents' and 'steps'
    """
    print("---NODE: RETRIEVE---")
    question = state["question"]

    # 1. Initialize VectorDB
    try:
        vector_db = VectorDBClient()
        
        # 2. Execute Retrieval
        # We fetch a bit more than needed to allow the 'Grade' node to filter
        results = vector_db.search(question, limit=5)
        
        # Convert Qdrant ScoredPoints to LangChain Documents
        docs = []
        for hit in results:
            # Reconstruct content if 'search_content' is missing
            payload = hit.payload or {}
            content = payload.get("search_content")
            if not content:
                summary = payload.get("context_summary", "")
                raw_text = payload.get("text", "")
                content = f"{summary}\n{raw_text}" if summary else raw_text
            
            # Map Qdrant hit to Document
            doc = Document(
                page_content=content,
                metadata={
                    "score": hit.score,
                    "id": hit.id,
                    **payload
                }
            )
            docs.append(doc)
            
        print(f"✅ Retrieved {len(docs)} documents.")
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        docs = []

    # 3. Update State
    return {
        "documents": docs, 
        "steps": ["retrieve"],
        # Increment attempts if this was a retry (handled by logic outside, but good to know)
    }
