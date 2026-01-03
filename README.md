# DiVA RAG Agent

A Retrieval-Augmented Generation (RAG) agent designed to answer questions based on the Diabetes Standards of Care (SoC) 2026 guidelines.

## Features
- **RAG Architecture**: Combines vector retrieval (FAISS) with a generative LLM (Qwen 2.5).
- **Interactive Mode**: Chat interface to query the knowledge base.
- **Source Citation**: Provides references to the specific recommendations used in the answer.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Agent**:
   ```bash
   python phase2_interactive.py
   ```
   *Note: The first run will download the model and build the vector index.*

## Configuration
Settings for chunking, retrieval, and model paths can be found in `config.py`.