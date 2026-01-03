from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import get_vector_store
from src.llm_manager import get_qwen_llm
import config

class RAGAgent:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_qwen_llm()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_TOP_K})
        
        # Define a prompt template that forces the model to use the context
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )

        # Create the chain manually using LCEL to avoid import errors
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            RunnableParallel({"context": self.retriever, "input": RunnablePassthrough()})
            .assign(answer=(
                RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
                | PROMPT
                | self.llm
                | StrOutputParser()
            ))
        )

    def ask(self, question):
        response = self.rag_chain.invoke(question)
        
        # Map response to match previous structure for compatibility
        return {
            "result": response["answer"],
            "source_documents": response["context"]
        }

if __name__ == "__main__":
    agent = RAGAgent()
    print(agent.ask("What is the A1C goal for surgery?"))