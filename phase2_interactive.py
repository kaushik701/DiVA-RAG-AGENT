from src.rag_agent import RAGAgent
import sys

def main():
    print("Initializing Phase 2: Full RAG Agent (Qwen 2.5)...")
    # This might take a moment to load the model
    agent = RAGAgent()
    print("\n--- RAG Agent Ready (Type 'exit' to quit) ---")

    while True:
        try:
            query = input("\nAsk Qwen a question about Diabetes SoC: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            response = agent.ask(query)
            
            print("\n>>> Answer:")
            print(response['result'])
            print("\n(Source Documents used:)")
            for doc in response['source_documents']:
                print(f"- {doc.metadata.get('title', 'Unknown')}: {doc.page_content[:50]}...")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()