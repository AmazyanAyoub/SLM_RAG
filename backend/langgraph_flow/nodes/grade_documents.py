from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from backend.models.llm_factory import LLMFactory
from backend.langgraph_flow.state import GraphState

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    Filters out irrelevant documents.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'documents' and 'steps'
    """
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    # 1. Initialize LLM & Grader
    try:
        llm = LLMFactory.get_fast_llm()
        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # 2. Define Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning useful for answering the question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        
        retrieval_grader = grade_prompt | structured_llm_grader

        # 3. Score Documents
        filtered_docs = []
        for d in documents:
            try:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
                print(f"grade: {grade}")
                
                if grade.lower() == "yes":
                    # print(f"✅ Document relevant: {d.metadata.get('id', 'unknown')}")
                    filtered_docs.append(d)
                else:
                    print(f"❌ Document not relevant: {d.metadata.get('id', 'unknown')}")
                    continue
            except Exception as e:
                print(f"⚠️ Grading error for doc {d.metadata.get('id', 'unknown')}: {e}")
                # We skip documents that fail grading to be safe
                continue
                
        print(f"🔍 Graded {len(documents)} documents. Retained {len(filtered_docs)} relevant.")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize grader or run grading loop: {e}")
        # Fallback: return original documents if grading completely fails
        filtered_docs = documents

    return {
        "documents": filtered_docs,
        "steps": state.get("steps", []) + ["grade_documents"]
    }

if __name__ == "__main__":
    from langchain_core.documents import Document
    
    print("--- TEST: Grade Documents ---")
    docs = [
        Document(page_content="The fortune limit for a couple is $100,000.", metadata={"id": "1"}),
        Document(page_content="The weather is nice today and the sun is shining.", metadata={"id": "2"})
    ]
    state = {
        "question": "What is the fortune limit?",
        "documents": docs,
        "steps": []
    }
    result = grade_documents(state)
    print(f"Original docs: {len(docs)}")
    print(f"Filtered docs: {len(result['documents'])}")