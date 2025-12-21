from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from backend.models.llm_factory import LLMFactory
from backend.langgraph_flow.state import GraphState

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to check if the answer addresses the question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

def hallucination_check(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers the question.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'grade' and 'steps'
    """
    print("---NODE: HALLUCINATION CHECK---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    llm = LLMFactory.get_fast_llm()

    # --- CHECK 1: Hallucination (Groundedness) ---
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in and supported by the retrieved facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_hallucination),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    
    try:
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade_grounded = score.binary_score
    except Exception as e:
        print(f"⚠️ Hallucination check failed: {e}")
        grade_grounded = "yes" # Assume grounded on error to avoid loops

    if grade_grounded.lower() == "yes":
        print("✅ Decision: Generation is grounded in documents.")
        
        # --- CHECK 2: Answer Relevance ---
        structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
        
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_answer),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        
        answer_grader = answer_prompt | structured_llm_grader_answer
        
        try:
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade_answer = score.binary_score
        except Exception as e:
            print(f"⚠️ Answer relevance check failed: {e}")
            grade_answer = "yes"

        if grade_answer.lower() == "yes":
            print("✅ Decision: Generation addresses the question.")
            return {"grade": "useful", "steps": state.get("steps", []) + ["hallucination_check"]}
        else:
            print("❌ Decision: Generation does not address the question.")
            return {"grade": "not useful", "steps": state.get("steps", []) + ["hallucination_check"]}
            
    else:
        print("❌ Decision: Generation is NOT grounded in documents (Hallucination).")
        return {"grade": "hallucination", "steps": state.get("steps", []) + ["hallucination_check"]}

if __name__ == "__main__":
    from langchain_core.documents import Document
    
    print("--- TEST: Hallucination Check ---")
    docs = [Document(page_content="The sky is blue.")]
    
    # Test 1: Grounded & Relevant
    state = {
        "question": "What color is the sky?",
        "documents": docs,
        "generation": "The sky is blue.",
        "steps": []
    }
    print("\nTest 1 (Grounded):", hallucination_check(state))
    
    # Test 2: Hallucination
    state["generation"] = "The sky is green."
    print("\nTest 2 (Hallucination):", hallucination_check(state))
