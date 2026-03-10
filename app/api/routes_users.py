"""Public user profile for leaderboard click-through."""

import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, desc, cast
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased
from sqlalchemy.dialects.postgresql import JSONB

from app.db.session import get_db
from app.models import User, UserProfile, Outfit, OutfitScore, RankingGroupMember

router = APIRouter(prefix="/v1/users", tags=["users"])
logger = logging.getLogger(__name__)
MIN_RATINGS_FOR_LEADERBOARD = 10
STYLE_SCOPES = ["Streetwear", "Minimal", "Vintage", "Luxury", "Y2K", "Casual"]


def _scope_start(scope: str):
  now = datetime.now(timezone.utc)
  if scope == "daily":
    return now.replace(hour=0, minute=0, second=0, microsecond=0)
  if scope == "weekly":
    start = now - timedelta(days=now.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0)
  if scope == "monthly":
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
  if scope == "yearly":
    return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
  return None


@router.get("/{user_id}/public-profile")
async def get_public_profile(
  user_id: str,
  viewer_user_id: str | None = None,
  db: AsyncSession = Depends(get_db),
):
  """Get a user's public profile for leaderboard view. Returns rank, avg score, top 5 outfits (if visibility allows)."""
  logger.info("get_public_profile user_id=%s", user_id)
  try:
    user_stmt = select(User.id, User.username, User.display_name, User.avatar_url).where(User.id == user_id)
    user_res = await db.execute(user_stmt)
    user_row = user_res.fetchone()
  except SQLAlchemyError:
    # Fallback when username column is not migrated yet.
    user_stmt = select(User.id, User.display_name, User.avatar_url).where(User.id == user_id)
    user_res = await db.execute(user_stmt)
    user_row = user_res.fetchone()
  if not user_row:
    raise HTTPException(status_code=404, detail="User not found")

  try:
    profile_stmt = select(
      UserProfile.profile_visibility,
      UserProfile.profile_visibility_mode,
      UserProfile.country,
    ).where(UserProfile.user_id == user_id)
    profile_res = await db.execute(profile_stmt)
    profile_row = profile_res.fetchone()
    if profile_row and profile_row.profile_visibility_mode in ("public", "friends_only", "private"):
      visibility = profile_row.profile_visibility_mode
    else:
      vis_bool = profile_row.profile_visibility if profile_row else True
      visibility = "public" if vis_bool else "private"
  except SQLAlchemyError:
    profile_stmt = select(UserProfile.profile_visibility, UserProfile.country).where(UserProfile.user_id == user_id)
    profile_res = await db.execute(profile_stmt)
    profile_row = profile_res.fetchone()
    vis_bool = profile_row.profile_visibility if profile_row else True
    visibility = "public" if vis_bool else "private"

  # Avg score and count
  stats_stmt = (
    select(
      func.count(OutfitScore.id).label("cnt"),
      func.avg(OutfitScore.drip_score).label("avg"),
    )
    .join(Outfit, Outfit.id == OutfitScore.outfit_id)
    .where(Outfit.user_id == user_id)
  )
  stats_res = await db.execute(stats_stmt)
  stats = stats_res.fetchone()
  rating_count = stats.cnt or 0
  avg_drip_score = round(float(stats.avg), 2) if stats and stats.avg is not None else None

  result = {
    "user_id": user_id,
    "display_name": getattr(user_row, "username", None) or user_row.display_name or "User",
    "avatar_url": user_row.avatar_url,
    "profile_visibility": visibility,
    "avg_drip_score": avg_drip_score,
    "rating_count": rating_count,
    "top_outfits": [],
    "rankings": [],
  }

  can_view_details = True
  if visibility == "private":
    can_view_details = False
  elif visibility == "friends_only":
    if viewer_user_id is None or viewer_user_id == user_id:
      can_view_details = viewer_user_id == user_id
    else:
      m1 = aliased(RankingGroupMember)
      m2 = aliased(RankingGroupMember)
      shared_group_stmt = (
        select(func.count())
        .select_from(m1)
        .join(m2, m1.group_id == m2.group_id)
        .where(m1.user_id == user_id, m2.user_id == viewer_user_id)
      )
      shared_res = await db.execute(shared_group_stmt)
      can_view_details = (shared_res.scalar() or 0) > 0

  if not can_view_details:
    result["profile_visibility"] = visibility
    return result

  # Top 5 rated outfits.
  top_stmt = (
    select(Outfit.id, Outfit.image_url, Outfit.scanned_at, OutfitScore.drip_score)
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id == user_id)
    .order_by(desc(OutfitScore.drip_score))
    .limit(5)
  )
  top_res = await db.execute(top_stmt)
  result["top_outfits"] = [
    {
      "id": str(r.id),
      "image_url": r.image_url,
      "scanned_at": r.scanned_at.isoformat() if r.scanned_at else None,
      "drip_score": round(float(r.drip_score), 2) if r.drip_score else None,
    }
    for r in top_res.fetchall()
  ]

  # Rankings across standard scopes (only shown if user has enough ratings in that scope)
  scopes = ["global", "yearly", "monthly", "weekly", "daily"]
  if profile_row and profile_row.country:
    scopes.append("country")
  scopes.extend([f"style:{s}" for s in STYLE_SCOPES])
  for scope in scopes:
    q = (
      select(
        Outfit.user_id,
        func.avg(OutfitScore.drip_score).label("avg_drip"),
        func.count(OutfitScore.id).label("cnt"),
      )
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .where(Outfit.user_id.isnot(None))
      .group_by(Outfit.user_id)
      .having(func.count(OutfitScore.id) >= MIN_RATINGS_FOR_LEADERBOARD)
      .order_by(desc(func.avg(OutfitScore.drip_score)))
    )
    start = _scope_start(scope)
    if start is not None:
      q = q.where(OutfitScore.created_at >= start)
    if scope == "country" and profile_row and profile_row.country:
      q = q.join(UserProfile, UserProfile.user_id == Outfit.user_id).where(UserProfile.country == profile_row.country)
    if scope.startswith("style:"):
      style_name = scope.replace("style:", "")
      q = q.where(cast(Outfit.style_tags, JSONB).contains([style_name]))
    try:
      res = await db.execute(q)
      rows = res.fetchall()
    except SQLAlchemyError:
      rows = []
    rank = None
    for idx, row in enumerate(rows, 1):
      if row.user_id == user_id:
        rank = idx
        break
    result["rankings"].append(
      {
        "scope": scope,
        "rank": rank,
        "total_eligible": len(rows),
      }
    )
  return result
