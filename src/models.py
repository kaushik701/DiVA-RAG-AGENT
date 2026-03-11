import uuid
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class TimeRange(BaseModel):
    start: datetime
    end: datetime

    @validator('end')
    def start_must_be_before_end(cls, v, values, **kwargs):
        if 'start' in values and v <= values['start']:
            raise ValueError('end time must be after start time')
        return v

class OrchestratorRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    data: Dict[str, Any] = Field(default_factory=dict)
    time_range: Optional[TimeRange] = None

    @validator('question')
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('question cannot be empty')
        return v

class AgentRequest(BaseModel):
    request_id: str
    question: str
    data: Dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    source: str
    rec_id: Optional[str] = None
    grade: Optional[str] = None

class RoutingStructure(BaseModel):
    confidence_score: float
    intent: str
    context: str
    grades: List[str]
    rec_ids: List[str]
    sources: List[str]
    source_count: int

class OrchestratorResponse(BaseModel):
    request_id: str
    agent_used: Optional[str] = None
    status: str
    answer: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list)
    routing_structure: Optional[RoutingStructure] = None
    error_details: Optional[str] = None