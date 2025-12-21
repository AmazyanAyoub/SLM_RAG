from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.models.llm_factory import LLMFactory
from backend.langgraph_flow.state import GraphState

def generate(state: GraphState):
    """
    Generate answer using the vectorstore documents.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updates 'generation' and 'steps'
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # 1. Initialize LLM (Smart Model for high quality answer)
    llm = LLMFactory.get_smart_llm()
    
    # 2. Define Prompt
    system = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question} \n\n Context: {context} \n\n Answer:"),
        ]
    )
    
    # 3. Build Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    # 4. Format Context
    context_text = "\n\n".join([d.page_content for d in documents])
    
    # 5. Generate
    try:
        generation = rag_chain.invoke({"context": context_text, "question": question})
        print(f"💡 Generated Answer: {generation}")
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        generation = "Sorry, I could not generate an answer due to an internal error."

    return {
        "generation": generation,
        "steps": state.get("steps", []) + ["generate"]
    }

if __name__ == "__main__":
    from langchain_core.documents import Document
    
    print("--- TEST: Generate ---")
    docs = [Document(page_content="The fortune limit for a couple is $100,000.")]
    state = {
        "question": "What is the fortune limit?",
        "documents": docs,
        "steps": []
    }
    print(generate(state))
