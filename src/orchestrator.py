import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

from src.models import OrchestratorRequest, AgentRequest, OrchestratorResponse, Citation, RoutingStructure
from src.rag_agent import RAGAgent
from src.fallback_agent import FallbackAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    def __init__(self):
        print("Orchestrator: Initializing agent registry...")
        self.agent_registry = [
            {
                "name": "SoC_RAG_Agent",
                "agent": RAGAgent(),
                "keywords": ["diabetes", "glucose", "insulin", "a1c", "patient", "hospital", "soc", "guideline", "surgery", "dka"],
                "required_inputs": [],
                "description": "Answers questions about the Diabetes Standards of Care."
            },
            # The Fallback Agent is always last.
            {
                "name": "Fallback_Agent",
                "agent": FallbackAgent(),
                "keywords": [],
                "required_inputs": [],
                "description": "Handles queries that are out of scope."
            }
        ]
        print("Orchestrator: System Ready.")

    def process_request(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Main orchestration flow following the 9-step process.
        """
        start_time = datetime.now()

        # Step 2: Basic validation is handled by Pydantic in OrchestratorRequest.

        # Step 3: Route the query to the appropriate agent.
        agent_def, intent = self._route_query(request.question)

        if not agent_def:
            return self._create_error_response(request, "No suitable agent found for the query.")

        # Step 4: Agent pre-check for required inputs.
        pre_check_status, follow_up_questions = self._pre_check_agent(agent_def, request)
        if pre_check_status == "needs_more_data":
            response = OrchestratorResponse(
                request_id=request.request_id,
                agent_used=agent_def['name'],
                status="needs_more_data",
                answer="I need more information to answer your question.",
                follow_ups=follow_up_questions
            )
            self._log_event(response.dict(), start_time)
            return response

        # Step 5: Call the selected agent with a standardized request.
        try:
            agent_request = AgentRequest(request_id=request.request_id, question=request.question, data=request.data)
            raw_agent_response = agent_def['agent'].ask(agent_request)
        except Exception as e:
            return self._create_error_response(request, f"Agent execution failed: {str(e)}", agent_name=agent_def['name'])

        # Step 6 & 7: Cleanup and normalize the raw agent response.
        sanitized_response = self._sanitize_response(raw_agent_response, agent_def['name'])

        # Step 8: Final packaging of the orchestrator's response.
        final_response = self._package_response(request, agent_def['name'], sanitized_response, intent)

        # Step 9: Log the final packaged response and return it.
        self._log_event(final_response.dict(), start_time)
        return final_response

    def _route_query(self, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Decides which agent to call based on keywords."""
        query_lower = query.lower()
        for entry in self.agent_registry:
            is_fallback = not entry["keywords"]
            matched_keyword = next((k for k in entry["keywords"] if k in query_lower), None)

            if matched_keyword:
                return entry, f"intent_{matched_keyword}"
            if is_fallback:
                return entry, "intent_out_of_scope"
        return None, None

    def _pre_check_agent(self, agent_def: Dict[str, Any], request: OrchestratorRequest) -> Tuple[str, List[str]]:
        """Checks if the agent has all the required inputs."""
        missing_inputs = [req for req in agent_def.get("required_inputs", []) if req not in request.data or not request.data[req]]
        if missing_inputs:
            follow_ups = [f"Could you please provide your {inp.replace('_', ' ')}?"]
            return "needs_more_data", follow_ups
        return "ok", []

    def _sanitize_response(self, raw_response: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Enforces schema, cleans up data, and performs safety checks."""
        sanitized = {}
        result_text = raw_response.get("result", "I'm sorry, I could not find an answer.")
        source_docs = raw_response.get("source_documents", [])

        # Safety post-check (f)
        if "you should take" in result_text.lower() or "i recommend you take" in result_text.lower():
            result_text = "I found some information that might be relevant, but I cannot provide medical advice or dosing instructions. Please consult with your healthcare provider."
            sanitized['status'] = "safety_override"
        else:
            is_success = "don't know" not in result_text.lower()[:50] and len(source_docs) > 0
            if agent_name == "Fallback_Agent": is_success = True
            sanitized['status'] = "success" if is_success else "miss"

        sanitized['answer'] = result_text

        # Filter citations (d) and remove internal fields (e)
        unique_citations = {}
        sanitized['results'] = []
        for doc in source_docs:
            source = doc.metadata.get("title", "Unknown Source")
            rec_id = doc.metadata.get("rec_id")
            grade = doc.metadata.get("grade")
            key = (source, rec_id)
            if key not in unique_citations:
                unique_citations[key] = Citation(source=source, rec_id=rec_id, grade=grade)
            
            sanitized['results'].append({"content": doc.page_content}) # Only include content

        sanitized['citations'] = list(unique_citations.values())
        sanitized['follow_ups'] = raw_response.get("follow_ups", []) # Fix missing pieces (b)

        return sanitized

    def _package_response(self, request: OrchestratorRequest, agent_name: str, sanitized: Dict[str, Any], intent: str) -> OrchestratorResponse:
        """Assembles the final OrchestratorResponse object."""
        # Calculate confidence score
        confidence = 0.0
        if sanitized['status'] == 'success': confidence = 0.95
        elif sanitized['status'] == 'miss': confidence = 0.4
        elif sanitized['status'] == 'safety_override': confidence = 0.8 # Confident it found something, but had to override

        # Extract metadata for routing structure
        citations = sanitized.get('citations', [])
        grades = list(set([c.grade for c in citations if c.grade]))
        rec_ids = list(set([c.rec_id for c in citations if c.rec_id]))
        sources = list(set([c.source for c in citations]))

        routing_structure = RoutingStructure(
            confidence_score=confidence,
            intent=intent,
            context="medical_guidelines" if agent_name == "SoC_RAG_Agent" else "general",
            grades=grades,
            rec_ids=rec_ids,
            sources=sources,
            source_count=len(sanitized.get('results', []))
        )

        return OrchestratorResponse(
            request_id=request.request_id,
            agent_used=agent_name,
            status=sanitized['status'],
            answer=sanitized['answer'],
            results=sanitized['results'],
            citations=sanitized['citations'],
            follow_ups=sanitized['follow_ups'],
            routing_structure=routing_structure
        )

    def _create_error_response(self, request: OrchestratorRequest, error_msg: str, agent_name: Optional[str] = None) -> OrchestratorResponse:
        """Creates a standardized error response."""
        response = OrchestratorResponse(
            request_id=request.request_id,
            agent_used=agent_name,
            status="error",
            answer="An internal error occurred while processing your request.",
            error_details=error_msg
        )
        self._log_event(response.dict(), datetime.now())
        return response

    def _log_event(self, payload: Dict[str, Any], start_time: datetime):
        """Logs the structured event, including timing."""
        duration = (datetime.now() - start_time).total_seconds() * 1000
        log_payload = payload.copy()
        log_payload['timestamp'] = start_time.isoformat()
        log_payload['time_taken_ms'] = round(duration, 2)
        logger.info(f"\n[OBSERVABILITY LOG]\n{json.dumps(log_payload, indent=2, default=str)}\n")

    def run(self, user_query: str) -> str:
        """Legacy wrapper for CLI usage. Returns just the answer string."""
        request = OrchestratorRequest(question=user_query)
        result = self.process_request(request)
        return result.answer