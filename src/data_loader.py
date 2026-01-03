import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def load_diabetes_data(file_path):
    """
    Loads the Diabetes SoC JSON file and converts it into LangChain Documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []

    # The JSON is a list of objects
    for entry in data:
        source_url = entry.get("url", "unknown")
        title = entry.get("title", "unknown")
        
        # Extract from 'recommendations' (list of strings)
        if "recommendations" in entry:
            for rec in entry["recommendations"]:
                if rec and isinstance(rec, str) and len(rec.strip()) > 10:
                    doc = Document(
                        page_content=rec.strip(),
                        metadata={"source": source_url, "title": title, "type": "recommendation"}
                    )
                    documents.append(doc)

        # Extract from 'recommendations_structured' (list of dicts)
        if "recommendations_structured" in entry:
            for item in entry["recommendations_structured"]:
                text = item.get("text", "")
                rec_id = item.get("rec_id", "")
                if text and len(text.strip()) > 10:
                    full_content = f"Recommendation {rec_id}: {text.strip()}"
                    doc = Document(
                        page_content=full_content,
                        metadata={"source": source_url, "title": title, "type": "structured", "rec_id": rec_id}
                    )
                    documents.append(doc)
    
    # Apply RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} raw entries. Split into {len(split_docs)} chunks (Size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}).")
    return split_docs