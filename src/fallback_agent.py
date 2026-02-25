class FallbackAgent:
    """
    A simple agent that handles out-of-domain queries to prevent hallucinations.
    It returns a standard refusal message without invoking an LLM.
    """
    def ask(self, query):
        return {
            "result": "I'm sorry, but I am designed to answer questions specifically about the Diabetes Standards of Care 2026. Your query appears to be outside of this scope.",
            "source_documents": []
        }