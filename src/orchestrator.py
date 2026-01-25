import json
import logging
from datetime import datetime
from src.rag_agent import RAGAgent

# Configure logging to output structured observability data
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    def __init__(self):
        print("Orchestrator: Initializing agent registry...")
        # 1. Registry of Agents: Future agents can be added here
        self.agents = {
            "SoC_RAG_Agent": RAGAgent()
        }
        print("Orchestrator: System Ready.")

    def _route_query(self, query: str):
        """
        Decides which agent to call.
        Phase 1 Logic: Keyword check. Future: LLM Router.
        """
        query_lower = query.lower()
        keywords = ["diabetes", "glucose", "insulin", "a1c", "patient", "hospital", "soc", "guideline", "surgery", "dka"]
        
        if any(k in query_lower for k in keywords):
            return "SoC_RAG_Agent"
        
        # Fallback: For testing purposes, we default to SoC, 
        # but in a real multi-agent system, we might return None here.
        return "SoC_RAG_Agent"

    def run(self, user_query: str):
        """
        Main Orchestration Flow: Route -> Execute -> Log -> Normalize
        """
        start_time = datetime.now()
        
        # 2. Routing
        agent_name = self._route_query(user_query)
        
        # Initialize Log Payload
        log_payload = {
            "timestamp": start_time.isoformat(),
            "query": user_query,
            "agent_selected": agent_name,
            "status": "pending",
            "execution_time_ms": 0,
            "metadata": {}
        }

        if not agent_name:
            log_payload["status"] = "skipped_no_agent"
            self._log_event(log_payload)
            return "I'm sorry, I don't have an agent capable of answering that question."

        # 3. Execution
        try:
            selected_agent = self.agents[agent_name]
            
            # Call the external agent
            response = selected_agent.ask(user_query)
            
            # Calculate timing
            duration = (datetime.now() - start_time).total_seconds() * 1000
            log_payload["execution_time_ms"] = round(duration, 2)

            # 4. Observability & Metadata Extraction
            result_text = response.get("result", "")
            source_docs = response.get("source_documents", [])
            
            # Extract specific Recommendation IDs (e.g., "16.4a") from metadata
            rec_ids = [doc.metadata.get("rec_id") for doc in source_docs if doc.metadata.get("rec_id")]
            
            # Determine success (Heuristic: did the model say "I don't know"?)
            # Check first 50 chars for refusal to avoid false negatives from hallucinations at the end
            is_success = "don't know" not in result_text.lower()[:50] and len(source_docs) > 0
            
            log_payload["status"] = "success" if is_success else "miss"
            log_payload["metadata"] = {
                "hit_count": len(source_docs),
                "recommendation_ids": list(set(rec_ids)), # Deduplicate
                "sources": [doc.metadata.get("title", "unknown") for doc in source_docs]
            }
            
            self._log_event(log_payload)

            # 5. Normalization / Final Answer
            return result_text

        except Exception as e:
            log_payload["status"] = "error"
            log_payload["error_details"] = str(e)
            self._log_event(log_payload)
            return "An error occurred while processing your request."

    def _log_event(self, payload):
        """Logs the structured event to console (or file in future)."""
        logger.info(f"\n[OBSERVABILITY LOG]\n{json.dumps(payload, indent=2)}\n")