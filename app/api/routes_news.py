from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.db.session import get_db
from app.models import CommunityNews, CommunityNewsDismissal, CommunityNewsLike, UserProfile
from app.schemas.news import CommunityNewsFeedResponse, CommunityNewsLikeResponse, CommunityNewsResponse
from app.services.storage import create_signed_image_url

router = APIRouter(prefix="/v1/news", tags=["community-news"])


def _signed_news_content(content: dict) -> dict:
  """Replace persisted private object references only at an authenticated boundary."""
  result = dict(content or {})
  for key in ("before_image_url", "after_image_url"):
    if result.get(key):
      result[key] = create_signed_image_url(result[key])
  if isinstance(result.get("placements"), list):
    result["placements"] = [
      {**item, "image_url": create_signed_image_url(item.get("image_url"))}
      if isinstance(item, dict) and item.get("image_url") else item
      for item in result["placements"]
    ]
  return result


@router.get("/feed", response_model=CommunityNewsFeedResponse)
async def get_news_feed(
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  country = (
    await db.execute(select(UserProfile.country).where(UserProfile.user_id == auth.app_user_id))
  ).scalar_one_or_none()
  dismissed = select(CommunityNewsDismissal.news_id).where(
    CommunityNewsDismissal.user_id == auth.app_user_id
  )
  now = datetime.now(timezone.utc)
  query = (
    select(CommunityNews)
    .where(
      CommunityNews.id.not_in(dismissed),
      CommunityNews.published_at <= now,
      or_(CommunityNews.expires_at.is_(None), CommunityNews.expires_at > now),
      or_(CommunityNews.audience_country.is_(None), CommunityNews.audience_country == country),
    )
    .order_by(CommunityNews.published_at.asc())
    .limit(20)
  )
  rows = (await db.execute(query)).scalars().all()
  news_ids = [row.id for row in rows]
  liked_ids: set[str] = set()
  like_counts: dict[str, int] = {}
  if news_ids:
    liked_ids = set(
      (
        await db.execute(
          select(CommunityNewsLike.news_id).where(
            CommunityNewsLike.user_id == auth.app_user_id,
            CommunityNewsLike.news_id.in_(news_ids),
          )
        )
      ).scalars().all()
    )
    count_rows = await db.execute(
      select(CommunityNewsLike.news_id, func.count(CommunityNewsLike.id))
      .where(CommunityNewsLike.news_id.in_(news_ids))
      .group_by(CommunityNewsLike.news_id)
    )
    like_counts = {news_id: int(count) for news_id, count in count_rows.fetchall()}
  return CommunityNewsFeedResponse(
    items=[
      CommunityNewsResponse(
        id=row.id,
        kind=row.kind,
        scope=row.scope,
        category=row.category,
        eyebrow=row.eyebrow,
        title=row.title,
        caption=row.caption,
        image_url=create_signed_image_url(row.image_url),
        content=_signed_news_content(row.content or {}),
        published_at=row.published_at,
        liked=row.id in liked_ids,
        like_count=like_counts.get(row.id, 0),
      )
      for row in rows
    ]
  )


@router.post("/{news_id}/dismiss", status_code=204)
async def dismiss_news(
  news_id: str,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  exists = (await db.execute(select(CommunityNews.id).where(CommunityNews.id == news_id))).scalar_one_or_none()
  if not exists:
    raise HTTPException(status_code=404, detail="News item not found")
  db.add(CommunityNewsDismissal(news_id=news_id, user_id=auth.app_user_id))
  try:
    await db.commit()
  except IntegrityError:
    await db.rollback()


@router.post("/{news_id}/like", response_model=CommunityNewsLikeResponse)
async def toggle_news_like(
  news_id: str,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  exists = (await db.execute(select(CommunityNews.id).where(CommunityNews.id == news_id))).scalar_one_or_none()
  if not exists:
    raise HTTPException(status_code=404, detail="News item not found")
  existing = (
    await db.execute(
      select(CommunityNewsLike).where(
        CommunityNewsLike.news_id == news_id,
        CommunityNewsLike.user_id == auth.app_user_id,
      )
    )
  ).scalar_one_or_none()
  if existing:
    await db.execute(delete(CommunityNewsLike).where(CommunityNewsLike.id == existing.id))
    liked = False
  else:
    db.add(CommunityNewsLike(news_id=news_id, user_id=auth.app_user_id))
    liked = True
  await db.commit()
  count = (
    await db.execute(
      select(func.count(CommunityNewsLike.id)).where(CommunityNewsLike.news_id == news_id)
    )
  ).scalar() or 0
  return CommunityNewsLikeResponse(liked=liked, like_count=int(count))
