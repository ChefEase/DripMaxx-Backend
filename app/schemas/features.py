from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class SubmitFeatureRequest(BaseModel):
  outfit_id: str
  feature_username: str | None = Field(default=None, max_length=50)
  instagram_url: str | None = Field(default=None, max_length=500)
  tiktok_url: str | None = Field(default=None, max_length=500)
  display_consent: bool

  @field_validator("feature_username", "instagram_url", "tiktok_url")
  @classmethod
  def trim_optional(cls, value: str | None):
    return value.strip() or None if value else None


class SubmitFeatureResponse(BaseModel):
  submission_id: str


class FeatureSubmissionResponse(BaseModel):
  id: str
  outfit_id: str
  user_id: str
  image_url: str
  drip_score: float
  account_email: str | None = None
  account_username: str | None = None
  feature_username: str | None = None
  instagram_url: str | None = None
  tiktok_url: str | None = None
  display_consent: bool
  status: str
  consented_at: datetime
  created_at: datetime


class FeatureSubmissionListResponse(BaseModel):
  submissions: list[FeatureSubmissionResponse]
