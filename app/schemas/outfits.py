from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
  color_match: float = Field(..., ge=0, le=10)
  fit_quality: float = Field(..., ge=0, le=10)
  body_compatibility: float = Field(..., ge=0, le=10)
  trend_score: float = Field(..., ge=0, le=10)
  style_match: float = Field(..., ge=0, le=10)


class SuggestionCard(BaseModel):
  id: Optional[str] = None
  title: str
  type: str
  description: str
  image_url: Optional[str] = None
  current_state: Optional[str] = None
  recommended_change: Optional[str] = None
  reason: Optional[str] = None
  importance: Literal["high", "medium", "low"] = "medium"
  target_state: Optional[str] = None
  impact: float = Field(default=0.3, ge=0, le=2)


class EvolutionRecommendationResult(BaseModel):
  id: str
  status: Literal["completed", "partial", "remaining", "regressed"]
  confidence: float = Field(..., ge=0, le=1)
  evidence: str


class EvolutionRevisionResult(BaseModel):
  revision_number: int
  previous_score: float = Field(..., ge=0, le=10)
  current_score: float = Field(..., ge=0, le=10)
  score_change: float
  completed_count: int
  total_recommendations: int
  recommendations: List[EvolutionRecommendationResult] = Field(default_factory=list)
  new_issues: List[str] = Field(default_factory=list)
  summary: str
  confidence: float = Field(..., ge=0, le=1)


class EvolutionSessionResponse(BaseModel):
  session_id: str
  original_outfit_id: str
  original_image_url: Optional[str] = None
  original_score: float = Field(..., ge=0, le=10)
  current_score: float = Field(..., ge=0, le=10)
  potential_score: float = Field(..., ge=0, le=10)
  target_image_url: Optional[str] = None
  target_generation_status: str = "pending"
  target_generation_error: Optional[str] = None
  recommendations: List[SuggestionCard] = Field(default_factory=list)
  revisions: List[EvolutionRevisionResult] = Field(default_factory=list)
  latest_revision: Optional[EvolutionRevisionResult] = None


class ScoreResponse(BaseModel):
  outfit_id: Optional[str] = None
  xp_awarded: int = 0
  drip_score: float = Field(..., ge=0, le=10)
  overall_score: float = Field(..., ge=0, le=10)
  quality_tier: str
  breakdown: ScoreBreakdown
  suggestions: List[SuggestionCard] = Field(default_factory=list)
  warnings: List[str] = Field(default_factory=list)
  unavailable_metrics: List[str] = Field(default_factory=list)
  visual_analysis: dict[str, Any] = Field(default_factory=dict, exclude=True)
  target_look: dict[str, Any] = Field(default_factory=dict, exclude=True)
  evolution: Optional[EvolutionSessionResponse] = None


class UserContext(BaseModel):
  style_preferences: List[str] = Field(default_factory=list)
  style_inspirations: List[str] = Field(default_factory=list)
  user_height: Optional[str] = None
  user_body_type: Optional[str] = None
  gender_style_preference: Optional[str] = None
  user_id: Optional[str] = None


class ScoreRequest(BaseModel):
  user_context: UserContext
