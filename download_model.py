import config
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings

def download_models():
    print("--- Downloading Models ---")
    
    # 1. Download Embedding Model
    print(f"Downloading Embedding Model: {config.EMBEDDING_MODEL_NAME}...")
    HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    print("Embedding Model Downloaded.")

    # 2. Download LLM
    print(f"Downloading LLM: {config.LLM_MODEL_ID}...")
    print("Downloading Tokenizer...")
    AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    print("Downloading Model Weights (this is large)...")
    AutoModelForCausalLM.from_pretrained(config.LLM_MODEL_ID, trust_remote_code=True)
    print("LLM Downloaded.")

if __name__ == "__main__":
    download_models()