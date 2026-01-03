from src.rag_agent import RAGAgent

def main():
    print("--- Phase 2: RAG Test (Non-Interactive) ---")
    print("Loading Agent (this may take time)...")
    agent = RAGAgent()
    
    # Test Query
    query = "How should I manage hyperglycemia in critically ill patients?"
    print(f"\nQuery: {query}")
    
    response = agent.ask(query)
    
    print("\n>>> Generated Answer:")
    print(response['result'])

if __name__ == "__main__":
    main()