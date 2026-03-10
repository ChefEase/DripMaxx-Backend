from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class EventIn(BaseModel):
  user_id: Optional[str] = None
  name: str
  payload: Dict[str, Any] = Field(default_factory=dict)


class EventOut(BaseModel):
  status: str = "ok"
