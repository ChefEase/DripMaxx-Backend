"""Schemas for rankings and leaderboards."""

from pydantic import BaseModel, Field
from typing import List, Optional


class LeaderboardEntry(BaseModel):
  rank: int
  user_id: str
  display_name: Optional[str] = None
  avg_drip_score: float
  rating_count: int


class LeaderboardResponse(BaseModel):
  scope: str
  entries: List[LeaderboardEntry] = Field(default_factory=list)
  total_eligible: int = 0


class UserRankingSummary(BaseModel):
  scope: str
  rank: Optional[int] = None
  total_eligible: int
  avg_drip_score: Optional[float] = None
  rating_count: int


class UserRankingsResponse(BaseModel):
  user_id: str
  ratings_count: int
  avg_drip_score: Optional[float] = None
  eligible_for_leaderboard: bool = False
  rankings: List[UserRankingSummary] = Field(default_factory=list)


class CreateGroupRequest(BaseModel):
  name: str
  user_id: str


class CreateGroupResponse(BaseModel):
  group_id: str
  code: str
  name: str


class JoinGroupRequest(BaseModel):
  code: str
  user_id: str


class JoinGroupResponse(BaseModel):
  group_id: str
  name: str
  joined: bool = True


class GroupSummary(BaseModel):
  id: str
  name: str
  code: str
  is_owner: bool = False


class GroupMemberStanding(BaseModel):
  rank: int
  user_id: str
  display_name: Optional[str] = None
  avg_drip_score: Optional[float] = None
  rating_count: int = 0


class GroupDetailsResponse(BaseModel):
  group_id: str
  name: str
  code: str
  members: List[GroupMemberStanding] = Field(default_factory=list)
