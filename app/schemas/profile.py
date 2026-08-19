
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional

from app.core.profile_values import parse_body_type, parse_gender_style
from app.models.entities import BodyTypeEnum, GenderStyleEnum


class ProfileSyncRequest(BaseModel):
  user_id: Optional[str] = None
  username: Optional[str] = None
  email: Optional[str] = None
  display_name: Optional[str] = None
  avatar_url: Optional[str] = None
  style_preferences: Optional[List[str]] = None
  style_inspirations: Optional[List[str]] = None
  user_height: Optional[float] = None
  user_body_type: Optional[BodyTypeEnum] = None
  gender_style_preference: Optional[GenderStyleEnum] = None
  country: Optional[str] = None
  locale: Optional[str] = None
  profile_visibility: Optional[str] = None
  profile_visibility_choice: Optional[Literal["private", "public", "undecided"]] = None
  community_feed_enabled: Optional[bool | Literal["undecided"]] = None
  leaderboard_enabled: Optional[bool | Literal["undecided"]] = None
  onboarding_privacy_completed: Optional[bool] = None

  @field_validator("user_body_type", mode="before")
  @classmethod
  def normalize_body_type(cls, value):
    if value is None:
      return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"n/a", "na", "none", "null"}:
      return None
    parsed = parse_body_type(raw)
    if parsed is None:
      raise ValueError(f"Invalid body type: {value}")
    return parsed

  @field_validator("gender_style_preference", mode="before")
  @classmethod
  def normalize_gender_style(cls, value):
    if value is None:
      return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"n/a", "na", "none", "null"}:
      return None
    parsed = parse_gender_style(raw)
    if parsed is None:
      raise ValueError(f"Invalid gender style preference: {value}")
    return parsed


class ProfileSyncResponse(BaseModel):
  user_id: str


class PrivacySocialPreferencesResponse(BaseModel):
  profile_visibility: Literal["private", "public", "undecided"]
  community_feed_enabled: bool | Literal["undecided"]
  leaderboard_enabled: bool | Literal["undecided"]
  onboarding_completed: bool


class DeleteAccountRequest(BaseModel):
  user_id: str | None = None


class DeleteAccountResponse(BaseModel):
  ok: bool
  auth_deleted: bool


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
