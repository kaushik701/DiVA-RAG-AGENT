import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.data_loader import load_diabetes_data
import config

def get_vector_store():
    """
    Returns a FAISS vector store. 
    If the index exists on disk, it loads it.
    If not, it creates it from the JSON data and saves it.
    """
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # Check if index exists
    if os.path.exists(config.FAISS_INDEX_DIR) and \
    os.path.exists(os.path.join(config.FAISS_INDEX_DIR, "index.faiss")):
        
        print("Loading existing Vector Store from disk...")
        try:
            vector_store = FAISS.load_local(
                config.FAISS_INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            print(f"Error loading index: {e}. Rebuilding...")

    # Rebuild index
    print("Creating new Vector Store...")
    docs = load_diabetes_data(config.JSON_FILE_PATH)
    
    if not docs:
        raise ValueError("No documents loaded. Check your JSON file path and content.")

    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save to disk
    vector_store.save_local(config.FAISS_INDEX_DIR)
    print(f"Vector Store saved to {config.FAISS_INDEX_DIR}")
    
    return vector_store