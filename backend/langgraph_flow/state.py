from typing import List, Dict, Any, TypedDict, Optional

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: The user's original query.
        generation: The LLM's final answer.
        documents: List of retrieved documents (LangChain Document objects or dicts).
        steps: A list of strings tracking which nodes were visited (e.g. ['retrieve', 'grade']).
        attempts: Number of times we've tried to retrieve/rewrite.
        classification: The result of the router (vector_store, graph_rag, generate).
        error: Any error message to display.
        grade: The result of the hallucination/relevance check (useful, not useful, hallucination).
    """
    question: str
    generation: Optional[str]
    documents: List[Any]
    steps: List[str]
    attempts: int
    classification: Optional[str]
    error: Optional[str]
    grade: Optional[str]
