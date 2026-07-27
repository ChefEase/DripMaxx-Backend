from datetime import datetime
from typing import Any

from pydantic import BaseModel


class CommunityNewsResponse(BaseModel):
  id: str
  kind: str
  scope: str
  category: str
  eyebrow: str
  title: str
  caption: str
  image_url: str | None = None
  content: dict[str, Any]
  published_at: datetime
  liked: bool = False
  like_count: int = 0


class CommunityNewsFeedResponse(BaseModel):
  items: list[CommunityNewsResponse]


class CommunityNewsLikeResponse(BaseModel):
  liked: bool
  like_count: int
