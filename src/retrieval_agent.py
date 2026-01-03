from src.vector_store import get_vector_store
import config

class RetrievalAgent:
    def __init__(self):
        self.vector_store = get_vector_store()
        # Retrieve top k most relevant chunks based on config
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_TOP_K})

    def query(self, question):
        """
        Retrieves relevant documents for a given question.
        """
        print(f"\nSearching for: {question}...")
        docs = self.retriever.invoke(question)
        return docs