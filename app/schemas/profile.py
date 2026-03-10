
from pydantic import BaseModel, Field
from typing import List, Optional


class ProfileSyncRequest(BaseModel):
  user_id: Optional[str] = None
  username: Optional[str] = None
  email: Optional[str] = None
  display_name: Optional[str] = None
  avatar_url: Optional[str] = None
  style_preferences: Optional[List[str]] = None
  style_inspirations: Optional[List[str]] = None
  user_height: Optional[float] = None
  user_body_type: Optional[str] = None
  gender_style_preference: Optional[str] = None
  country: Optional[str] = None
  locale: Optional[str] = None
  profile_visibility: Optional[str] = None


class ProfileSyncResponse(BaseModel):
  user_id: str


class OutfitSummary(BaseModel):
  id: str
  image_url: Optional[str] = None
  drip_score: Optional[float] = None
  scanned_at: Optional[str] = None


class ScoreHistoryPoint(BaseModel):
  recorded_at: str
  drip_score: float


class ProfileHistoryResponse(BaseModel):
  recent_outfits: List[OutfitSummary] = Field(default_factory=list)
  history: List[ScoreHistoryPoint] = Field(default_factory=list)


class StyleDNAResponse(BaseModel):
  user_id: str
  label: str
  description: str
  tags: List[str] = Field(default_factory=list)
