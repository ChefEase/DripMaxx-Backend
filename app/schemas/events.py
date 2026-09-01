from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class EventIn(BaseModel):
  anonymous_id: Optional[str] = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
  name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
  payload: Dict[str, Any] = Field(default_factory=dict)


class EventOut(BaseModel):
  status: str = "ok"
