from src.orchestrator import Orchestrator
import sys

def main():
    print("--- Phase 3: Orchestrator (Manager) Agent ---")
    
    # Initialize Orchestrator
    manager = Orchestrator()
    
    # 1. Hardcoded Tests (Proof of Concept)
    print("\n=== Running Hardcoded Tests ===")
    test_queries = [
        "What is the A1C goal for surgery?",
        "Tell me a joke about Python." # This tests how the agent handles irrelevant/unknown queries which is out of context in Diabetes Soc
    ]

    for q in test_queries:
        print(f"\nUser Input: '{q}'")
        answer = manager.run(q)
        print(f"Final Output: {answer}")
        print("-" * 50)

    # 2. Interactive Loop
    print("\n=== Interactive Mode (Type 'exit' to quit) ===")
    while True:
        try:
            user_input = input("\nManager > Enter query: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            print(f">>> {manager.run(user_input)}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()