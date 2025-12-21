import sys
import os

# Add project root to path to ensure imports work correctly
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.langgraph_flow.graph_builder import app

def main():
    print("==========================================")
    print("🤖 Agentic RAG CLI")
    print("Type 'exit' or 'q' to quit.")
    print("==========================================")

    while True:
        try:
            query = input("\nUser Query: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue

            # Initialize State
            initial_state = {
                "question": query,
                "attempts": 0,
                "steps": [],
                "documents": [],
                "generation": None,
                "classification": None,
                "error": None,
                "grade": None
            }

            print("\n🔄 Processing...")
            
            # Run the graph
            # We use invoke to run until the end state
            result = app.invoke(initial_state)

            print("\n" + "="*30)
            print("💡 FINAL ANSWER")
            print("="*30)
            print(result.get("generation"))
            
            print("\n" + "-"*30)
            print("🛠️  Steps Taken:")
            print(" -> ".join(result.get("steps", [])))
            print("-" * 30)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()