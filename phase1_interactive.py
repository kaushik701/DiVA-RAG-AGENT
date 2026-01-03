from src.retrieval_agent import RetrievalAgent
import sys

def main():
    print("Initializing Phase 1: Retrieval Only Agent...")
    agent = RetrievalAgent()
    print("\n--- Agent Ready (Type 'exit' to quit) ---")

    while True:
        try:
            query = input("\nEnter your query: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            docs = agent.query(query)
            
            print(f"\nFound {len(docs)} relevant documents:")
            for i, doc in enumerate(docs):
                print(f"\n--- Document {i+1} ---")
                print(doc.page_content)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()