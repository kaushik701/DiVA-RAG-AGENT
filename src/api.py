from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.orchestrator import Orchestrator
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="DiVA RAG Agent API",
    description="API for the Diabetes Standards of Care RAG Agent with Orchestration",
    version="1.0.0"
)

# Initialize Orchestrator
orchestrator = Orchestrator()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/v1/query")
async def query_agent(request: QueryRequest):
    """
    Endpoint to submit a query to the Orchestrator.
    Returns a structured JSON response with routing details and answer.
    """
    response = orchestrator.process_query(request.query)
    return response

if __name__ == "__main__":
    print("Starting API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)