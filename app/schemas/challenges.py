from datetime import datetime
from pydantic import BaseModel, Field


class AnnouncementResponse(BaseModel):
  id: str
  title: str
  body: str | None = None
  cta_label: str | None = None
  cta_url: str | None = None
  starts_at: datetime | None = None
  ends_at: datetime | None = None


class ChallengeResponse(BaseModel):
  id: str
  title: str
  description: str | None = None
  reward_scans: int
  reward_xp: int
  participation_xp: int
  winner_xp: int
  starts_at: datetime
  ends_at: datetime
  winner_submission_id: str | None = None
  winner_selected_at: datetime | None = None


class ActiveChallengeResponse(BaseModel):
  announcement: AnnouncementResponse | None = None
  challenge: ChallengeResponse | None = None


class SubmitChallengeRequest(BaseModel):
  outfit_id: str
  display_consent: bool = Field(False)


class SubmitChallengeResponse(BaseModel):
  submission_id: str
  challenge_id: str
  awarded_xp: int


class VoteRequest(BaseModel):
  submission_id: str


class AdminRankRequest(BaseModel):
  first_submission_id: str
  second_submission_id: str
  third_submission_id: str


class WinnerRequest(BaseModel):
  submission_id: str


class ChallengeSubmissionResponse(BaseModel):
  id: str
  challenge_id: str
  user_id: str
  outfit_id: str
  image_url: str | None = None
  display_name: str | None = None
  admin_rank: int | None = None
  admin_points: float
  user_vote_points: float
  final_points: float
  created_at: datetime


class ChallengeResultsResponse(BaseModel):
  challenge_id: str
  winner_submission_id: str | None = None
  winner_selected_at: datetime | None = None
  viewer_vote_submission_id: str | None = None
  submissions: list[ChallengeSubmissionResponse]
