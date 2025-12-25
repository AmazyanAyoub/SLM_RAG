# PHASE 1.5: Teacher LLM summarizer for chunks
import os, re
# from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Define the Teacher Model (Groq)
# We use a cheaper/faster model for this bulk task to save costs/time
# TEACHER_MODEL_ID = "llama-3.3-70b-versatile"
# TEACHER_MODEL_ID = "llama-3.1-8b-instant"

# REMOTE_OLLAMA_URL = "http://18.132.143.112:14528"
# CONTEXT_MODEL = "qwen3:8b"  # Updated to match your local model
CONTEXT_MODEL = "qwen3:4b"  # Local Ollama model for context enrichment


class ContextualEnricher:
    def __init__(self):
        """
        Initialize the Contextual Enricher with the Teacher LLM.
        """
        # api_key = os.getenv("GROQ_API_KEY")
        # if not api_key:
        #     raise ValueError("❌ GROQ_API_KEY not found in environment variables.")

        # self.llm = ChatGroq(
        #     model=TEACHER_MODEL_ID,
        #     temperature=0.3, # Slight creativity allowed for summaries
        #     max_tokens=512,
        #     api_key=api_key
        # )

        self.llm = ChatOllama(
            model=CONTEXT_MODEL,
            base_url="http://localhost:11434",
            temperature=0.1, 
            num_ctx=4096,
            keep_alive="15m"
        )



        # The Anthropic/2025 Style Prompt
        self.prompt = ChatPromptTemplate.from_template(
            """
            <document>
            {document_content}
            </document>

            Here is a chunk we have extracted from the document above:
            <chunk>
            {chunk_content}
            </chunk>

            Please provide a short, succinct context to situate this chunk within the overall document. 
            Answer the question: "What is this document about and where does this chunk fit in?"
            
            Do not repeat the chunk. Just give the context in 1-2 sentences.
            Start the context with: "This chunk is from [Document Title/Subject]..."
            """
        )
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def clean_response(self, text: str) -> str:
        """Removes <think> tags if the model outputs them."""
        # Remove <think>...</think> content
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove any leading "Context:" labels if the model adds them redundantly
        cleaned = cleaned.replace("Context:", "").strip()
        return cleaned

    async def enrich_chunk(self, chunk_content: str, full_document_content: str) -> str:
        """
        Generates context for a single chunk.
        """
        try:
            # We truncate full_document_content to avoid blowing up the context window
            # Taking the first 2000 chars is usually enough for "Global Context"
            truncated_doc = full_document_content[:5000] 
            
            raw_context = await self.chain.ainvoke({
                "document_content": truncated_doc,
                "chunk_content": chunk_content
            })
            
            # Combine Context + Original Content
            clean_context = self.clean_response(raw_context)

            # This is what gets EMBEDDED, but the LLM sees only the original chunk usually
            enriched_text = f"Context: {clean_context}\n\nContent: {chunk_content}"
            return enriched_text

        except Exception as e:
            print(f"⚠️ Error enriching chunk: {e}")
            return chunk_content # Fallback to original text if Teacher fails

# Simple test block to verify it works
if __name__ == "__main__":
    import asyncio
    import dotenv
    dotenv.load_dotenv()

    async def test():
        enricher = ContextualEnricher()
        doc = "Standard Operating Procedure for Coffee Machine. Step 1: Turn on. Step 2: Add water."
        chunk = "Step 2: Add water."
        
        print("Enriching...")
        res = await enricher.enrich_chunk(chunk, doc)
        print(f"\nResult:\n{res}")

    asyncio.run(test())