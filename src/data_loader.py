import json
import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src import config

def load_documents():
    """
    Loads data from the specific JSON structure provided (pdfs-converted_28_all_chapters.json).
    Extracts both 'text_core' and 'recommendations_structured'.
    """
    # Update this path if the filename changes
    file_path = os.path.join("data", "pdfs-converted_28_all_chapters.json")
    
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return []

    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    for chapter in data:
        # Common metadata for the chapter
        title = chapter.get("title", "Unknown Chapter")
        source_url = chapter.get("url", "")
        
        # 1. Process Main Text (text_core)
        main_text = chapter.get("text_core", "")
        if main_text:
            chunks = text_splitter.split_text(main_text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source_url,
                        "title": title,
                        "type": "text_core"
                    }
                )
                documents.append(doc)

        # 2. Process Structured Recommendations
        recommendations = chapter.get("recommendations_structured", [])
        for rec in recommendations:
            rec_text = rec.get("text", "")
            rec_id = rec.get("rec_id", "")
            grade = rec.get("grade", "N/A")
            
            if rec_text:
                # Format content to include the ID for context
                content = f"Recommendation {rec_id} (Grade {grade}): {rec_text}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": source_url,
                        "title": title,
                        "type": "recommendation",
                        "rec_id": rec_id,
                        "grade": grade
                    }
                )
                documents.append(doc)

    print(f"Loaded {len(documents)} documents from {len(data)} chapters.")
    return documents