from typing import List, Optional
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
  color_match: float = Field(..., ge=0, le=10)
  fit_quality: float = Field(..., ge=0, le=10)
  body_compatibility: float = Field(..., ge=0, le=10)
  trend_score: float = Field(..., ge=0, le=10)
  style_match: float = Field(..., ge=0, le=10)


class SuggestionCard(BaseModel):
  title: str
  type: str
  description: str
  image_url: Optional[str] = None


class ScoreResponse(BaseModel):
  drip_score: float = Field(..., ge=0, le=10)
  breakdown: ScoreBreakdown
  suggestions: List[SuggestionCard] = Field(default_factory=list)
  warnings: List[str] = Field(default_factory=list)


class UserContext(BaseModel):
  style_preferences: List[str] = Field(default_factory=list)
  style_inspirations: List[str] = Field(default_factory=list)
  user_height: Optional[str] = None
  user_body_type: Optional[str] = None
  gender_style_preference: Optional[str] = None
  user_id: Optional[str] = None


class ScoreRequest(BaseModel):
  user_context: UserContext
