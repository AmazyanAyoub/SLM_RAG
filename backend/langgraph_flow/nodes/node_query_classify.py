from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from backend.models.llm_factory import LLMFactory
from backend.langgraph_flow.state import GraphState

# 1. Define the Output Schema for the Classifier
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vector_store", "generate"] = Field(
        ...,
        description="Given a user question choose to route it to 'vector_store' (specific facts/lookup) or 'generate' (simple chit-chat/no context needed)."
    )

def classify_query(state: GraphState):
    """
    Determines whether to use vector search or direct generation.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updates 'steps' key
        str: The next node to call
    """
    print("---NODE: CLASSIFY QUERY---")
    question = state["question"]
    
    # 2. Get the Student LLM
    llm = LLMFactory.get_student_llm()
    
    # 3. Create the Structured Output LLM
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # 4. Define the Prompt
    system_prompt = """You are an expert at routing a user question to a vectorstore or directly to generation.

    - Use 'vector_store' for questions about specific facts, numbers, clauses, or "lookup" style queries. (e.g. "What is the deductible?", "How do I reset my password?")
    - Use 'generate' for simple greetings, compliments, or questions that don't require external knowledge. (e.g. "Hello", "Thanks", "Who are you?")
    """
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # 5. Build the Chain
    router = route_prompt | structured_llm_router

    # 6. Execute
    try:
        source = router.invoke({"question": question})
        decision = source.datasource
    except Exception as e:
        print(f"⚠️ Classification failed: {e}. Defaulting to 'vector_store'.")
        decision = "vector_store"

    print(f"--> ROUTING TO: {decision}")
    
    # Update State
    return {"steps": ["classify_query"], "classification": decision}
