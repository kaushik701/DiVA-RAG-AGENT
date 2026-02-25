import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# Data File
JSON_FILE_PATH = os.path.join(DATA_DIR, "pdfs-converted_28_all_chapters.json")

# Models
# Using a standard embedding model for retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# The Qwen model requested
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Retrieval & Chunking Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 3