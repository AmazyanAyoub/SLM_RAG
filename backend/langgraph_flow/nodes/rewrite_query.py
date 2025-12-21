from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.models.llm_factory import LLMFactory
from backend.langgraph_flow.state import GraphState

def rewrite_query(state: GraphState):
    """
    Transform the query to produce a better question.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'question' and 'steps'
    """
    print("---NODE: REWRITE QUERY---")
    question = state["question"]
    
    # 1. Initialize LLM
    try:
        llm = LLMFactory.get_fast_llm()
        
        # 2. Define Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
            ]
        )
        
        question_rewriter = re_write_prompt | llm | StrOutputParser()
        
        # 3. Rewrite Question
        better_question = question_rewriter.invoke({"question": question})
        
        print(f"✨ Rewritten Query: '{better_question}'")
        
    except Exception as e:
        print(f"⚠️ Query rewriting failed: {e}")
        better_question = question

    return {
        "question": better_question,
        "steps": state.get("steps", []) + ["rewrite_query"]
    }

if __name__ == "__main__":
    print("--- TEST: Rewrite Query ---")
    state = {"question": "aid limits", "steps": []}
    print(rewrite_query(state))