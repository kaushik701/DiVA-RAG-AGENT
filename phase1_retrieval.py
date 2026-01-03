from src.retrieval_agent import RetrievalAgent

def main():
    print("--- Phase 1: Retrieval Test (Non-Interactive) ---")
    agent = RetrievalAgent()
    
    # Test Query
    query = "What is the A1C goal for surgery?"
    print(f"\nQuery: {query}")
    
    docs = agent.query(query)
    
    print(f"\nFound {len(docs)} relevant documents:")
    for i, doc in enumerate(docs):
        print(f"\n[Document {i+1}] Source: {doc.metadata.get('title', 'Unknown')}")
        print(f"Content: {doc.page_content[:1000]}...")

if __name__ == "__main__":
    main()