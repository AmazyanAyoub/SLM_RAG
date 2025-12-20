from langgraph.graph import END, StateGraph
from backend.langgraph_flow.state import GraphState

from backend.langgraph_flow.nodes.node_query_classify import classify_query
from backend.langgraph_flow.nodes.rewrite_query import rewrite_query
from backend.langgraph_flow.nodes.node_retrieve import retrieve
from backend.langgraph_flow.nodes.grade_documents import grade_documents
from backend.langgraph_flow.nodes.node_generate import generate
from backend.langgraph_flow.nodes.hallucination_check import hallucination_check

def decide_route(state):
    """
    Route based on the classification results.
    """
    print("---DECIDE ROUTE---")
    classification = state["classification"]
    if classification == "vector_store":
        return "vector_store"
    else:
        return "generate"

def check_doc_relevance(state):
    """
    Check if we have relevant documents after grading.
    """
    print("---CHECK DOC RELEVANCE---")
    documents = state["documents"]
    if not documents:
        # All documents filtered out, rewrite query
        print("---DECISION: NO DOCUMENTS, REWRITE QUERY---")
        return "rewrite_query"
    else:
        # We have relevant documents, generate answer
        print("---DECISION: DOCUMENTS FOUND, GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether to end or retry based on hallucination check.
    """
    print("---GRADE GENERATION---")
    grade = state["grade"]
    
    if grade == "useful":
        print("---DECISION: USEFUL, END---")
        return "useful"
    elif grade == "not useful":
        print("---DECISION: NOT USEFUL, REWRITE QUERY---")
        return "not useful"
    else:
        # Hallucination
        print("---DECISION: HALLUCINATION, RE-GENERATE---")
        return "hallucination"

# Build the Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("classify_query", classify_query)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("hallucination_check", hallucination_check)

# Add Edges
workflow.set_entry_point("classify_query")

# Conditional Edge from Classifier
workflow.add_conditional_edges(
    "classify_query",
    decide_route,
    {
        "vector_store": "rewrite_query",
        "generate": "generate",
    }
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge from Grader (Empty docs check)
workflow.add_conditional_edges(
    "grade_documents",
    check_doc_relevance,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate",
    }
)

workflow.add_edge("generate", "hallucination_check")

# Conditional Edge from Hallucination Check
workflow.add_conditional_edges(
    "hallucination_check",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "rewrite_query",
        "hallucination": "generate",
    }
)

# Compile
app = workflow.compile()
