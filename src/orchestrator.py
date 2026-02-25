import json
import logging
import uuid
from datetime import datetime
from src.rag_agent import RAGAgent
from src.fallback_agent import FallbackAgent

# Configure logging to output structured observability data
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    def __init__(self):
        print("Orchestrator: Initializing agent registry...")
        # 1. Agent Registry: Defines agents and their routing rules.
        # The orchestrator checks them in order. The first match is used.
        self.agent_registry = [
            {
                "name": "SoC_RAG_Agent",
                "agent": RAGAgent(),
                "keywords": ["diabetes", "glucose", "insulin", "a1c", "patient", "hospital", "soc", "guideline", "surgery", "dka"],
                "description": "Answers questions about the Diabetes Standards of Care."
            },
            # The Fallback Agent should always be last, as it catches any query that wasn't routed.
            {
                "name": "Fallback_Agent",
                "agent": FallbackAgent(),
                "keywords": [], # No keywords makes this the default/fallback agent
                "description": "Handles queries that are out of scope."
            }
        ]
        print("Orchestrator: System Ready.")

    def _route_query(self, query: str):
        """
        Decides which agent to call by iterating through the agent registry.
        The first agent whose keywords match the query is selected.
        A fallback agent should be placed last with no keywords.
        Returns: (agent_name, agent_instance, intent)
        """
        query_lower = query.lower()
        
        for entry in self.agent_registry:
            # An entry with no keywords is a default/fallback, so it matches if no other agent has.
            is_fallback = not entry["keywords"]
            
            # Find specific keyword match for intent detection
            matched_keyword = next((k for k in entry["keywords"] if k in query_lower), None)

            if matched_keyword:
                return entry["name"], entry["agent"], f"intent_{matched_keyword}"
            
            if is_fallback:
                return entry["name"], entry["agent"], "intent_out_of_scope"
        
        # This case should not be reached if a fallback agent is correctly configured.
        return None, None, "unknown"

    def process_query(self, user_query: str):
        """
        Processes a query and returns a structured JSON response suitable for an API.
        """
        req_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # 2. Routing
        agent_name, selected_agent, intent = self._route_query(user_query)
        
        # Initialize Response Structure
        response_payload = {
            "req_id": req_id,
            "query": user_query,
            "agent_id": agent_name,
            "status": "pending",
            "timestamp": start_time.isoformat(),
            "time_taken_ms": 0,
            "answer": "",
            "routing_structure": {
                "confidence_score": 0.0,
                "intent": intent,
                "context": "medical_guidelines" if agent_name == "SoC_RAG_Agent" else "general",
                "grades": []
            }
        }

        if not selected_agent:
            response_payload["status"] = "failed"
            response_payload["answer"] = "No suitable agent found."
            self._log_event(response_payload)
            return response_payload

        # 3. Execution
        try:
            # Call the selected agent
            response = selected_agent.ask(user_query)
            
            # Calculate timing
            duration = (datetime.now() - start_time).total_seconds() * 1000
            response_payload["time_taken_ms"] = round(duration, 2)

            # 4. Observability & Metadata Extraction
            result_text = response.get("result", "")
            source_docs = response.get("source_documents", [])
            
            # Extract Grades, Rec IDs, and Sources from metadata
            grades = list(set([doc.metadata.get("grade") for doc in source_docs if doc.metadata.get("grade")]))
            rec_ids = list(set([doc.metadata.get("rec_id") for doc in source_docs if doc.metadata.get("rec_id")]))
            sources = list(set([doc.metadata.get("title") for doc in source_docs if doc.metadata.get("title")]))
            
            # Determine success (Heuristic: did the model say "I don't know"?)
            # Check first 50 chars for refusal to avoid false negatives from hallucinations at the end
            is_success = "don't know" not in result_text.lower()[:50] and len(source_docs) > 0
            
            # Calculate a pseudo-confidence score based on retrieval success
            confidence = 0.95 if is_success else 0.1
            if not source_docs and agent_name == "SoC_RAG_Agent":
                confidence = 0.0

            response_payload["status"] = "success" if is_success else "miss"
            response_payload["answer"] = result_text
            
            # Update Routing Structure
            response_payload["routing_structure"].update({
                "confidence_score": confidence,
                "grades": grades,
                "rec_ids": rec_ids,
                "sources": sources,
                "source_count": len(source_docs)
            })
            
            self._log_event(response_payload)
            return response_payload

        except Exception as e:
            response_payload["status"] = "error"
            response_payload["answer"] = "An internal error occurred."
            response_payload["error_details"] = str(e)
            self._log_event(response_payload)
            return response_payload

    def run(self, user_query: str):
        """
        Legacy wrapper for CLI usage. Returns just the answer string.
        """
        result = self.process_query(user_query)
        return result["answer"]

    def _log_event(self, payload):
        """Logs the structured event to console (or file in future)."""
        logger.info(f"\n[OBSERVABILITY LOG]\n{json.dumps(payload, indent=2)}\n")