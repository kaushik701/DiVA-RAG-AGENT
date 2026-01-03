import config
from src.data_loader import load_diabetes_data
import os

def main():
    print("--- Verifying Data ---")
    print(f"Looking for file at: {config.JSON_FILE_PATH}")
    
    if not os.path.exists(config.JSON_FILE_PATH):
        print("ERROR: File not found!")
        return

    docs = load_diabetes_data(config.JSON_FILE_PATH)
    print(f"\nSUCCESS: Loaded {len(docs)} document chunks.")
    if len(docs) > 0:
        print(f"Sample Chunk 1: {docs[0].page_content[:100]}...")

if __name__ == "__main__":
    main()