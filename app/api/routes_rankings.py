"""Rankings and leaderboard API. Users appear after 10 outfit ratings."""

import logging
import secrets
import string
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, cast, desc, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, optional_auth, require_auth
from app.db.session import get_db
from app.models import (
  Outfit,
  OutfitScore,
  RankingGroup,
  RankingGroupMember,
  User,
  UserProfile,
)
from app.schemas.rankings import (
  CreateGroupRequest,
  CreateGroupResponse,
  GroupDetailsResponse,
  GroupMemberStanding,
  GroupSummary,
  JoinGroupRequest,
  JoinGroupResponse,
  LeaderboardEntry,
  LeaderboardResponse,
  UserRankingSummary,
  UserRankingsResponse,
)


router = APIRouter(prefix="/v1/rankings", tags=["rankings"])
logger = logging.getLogger(__name__)
MIN_RATINGS_FOR_LEADERBOARD = 10
STYLE_SCOPES = ["Streetwear", "Minimal", "Vintage", "Luxury", "Y2K", "Casual"]


def _require_actor_user_id(auth: AuthContext, claimed_user_id: str | None) -> str:
  if claimed_user_id and claimed_user_id != auth.app_user_id:
    raise HTTPException(status_code=403, detail="user_id does not match authenticated user")
  return auth.app_user_id


def _gen_group_code() -> str:
  alphabet = string.ascii_uppercase + string.digits
  return "".join(secrets.choice(alphabet) for _ in range(6))


def _time_bounds(scope: str):
  now = datetime.now(timezone.utc)
  if scope == "daily":
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
  elif scope == "weekly":
    start = now - timedelta(days=now.weekday())
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
  elif scope == "monthly":
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
  elif scope == "yearly":
    start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
  else:
    start = None
  return start, now


def _eligible_users_subquery():
  return (
    select(Outfit.user_id.label("user_id"))
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id.isnot(None))
    .group_by(Outfit.user_id)
    .having(func.count(OutfitScore.id) >= MIN_RATINGS_FOR_LEADERBOARD)
    .subquery()
  )


async def _get_user_ratings_count(
  db: AsyncSession,
  user_id: str,
  scope: str,
  country: str | None = None,
  group_id: str | None = None,
  style: str | None = None,
) -> tuple[int, float | None]:
  q = (
    select(func.count(OutfitScore.id).label("cnt"), func.avg(OutfitScore.drip_score).label("avg"))
    .join(Outfit, Outfit.id == OutfitScore.outfit_id)
    .where(Outfit.user_id == user_id)
  )
  if scope != "global":
    start, _ = _time_bounds(scope)
    if start:
      q = q.where(OutfitScore.created_at >= start)
  if country:
    q = q.join(UserProfile, UserProfile.user_id == Outfit.user_id).where(UserProfile.country == country)
  if group_id:
    q = q.join(RankingGroupMember, RankingGroupMember.user_id == Outfit.user_id).where(RankingGroupMember.group_id == group_id)
  if style:
    q = q.where(cast(Outfit.style_tags, JSONB).contains([style]))
  res = await db.execute(q)
  row = res.fetchone()
  cnt = row.cnt or 0
  avg_val = float(row.avg) if row and row.avg is not None else None
  return cnt, avg_val


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
  scope: str = Query("global", description="global|yearly|monthly|weekly|daily|country|group|style"),
  country: str | None = Query(None),
  group_id: str | None = Query(None),
  style: str | None = Query(None),
  limit: int = Query(30, ge=1, le=100),
  auth: AuthContext | None = Depends(optional_auth),
  db: AsyncSession = Depends(get_db),
):
  if scope == "country" and not country:
    raise HTTPException(status_code=400, detail="country required for country scope")
  if scope == "group" and not group_id:
    raise HTTPException(status_code=400, detail="group_id required for group scope")
  if scope == "style" and not style:
    raise HTTPException(status_code=400, detail="style required for style scope")
  if scope == "group":
    if auth is None:
      raise HTTPException(status_code=401, detail="Missing token")
    member_res = await db.execute(
      select(RankingGroupMember.id).where(
        RankingGroupMember.group_id == group_id,
        RankingGroupMember.user_id == auth.app_user_id,
      )
    )
    if member_res.scalar_one_or_none() is None:
      raise HTTPException(status_code=403, detail="You are not a member of this group")

  try:
    eligible_users = _eligible_users_subquery()
    base = (
      select(
        Outfit.user_id,
        func.avg(OutfitScore.drip_score).label("avg_drip"),
        func.count(OutfitScore.id).label("cnt"),
      )
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .join(eligible_users, eligible_users.c.user_id == Outfit.user_id)
      .where(Outfit.user_id.isnot(None))
      .group_by(Outfit.user_id)
    )
    if scope not in ("global", "country", "group", "style"):
      start, _ = _time_bounds(scope)
      if start:
        base = base.where(OutfitScore.created_at >= start)
    if scope == "country" and country:
      base = base.join(UserProfile, UserProfile.user_id == Outfit.user_id).where(UserProfile.country == country)
    if scope == "group" and group_id:
      base = base.join(RankingGroupMember, RankingGroupMember.user_id == Outfit.user_id).where(RankingGroupMember.group_id == group_id)
    if scope == "style" and style:
      base = base.where(cast(Outfit.style_tags, JSONB).contains([style]))

    subq = base.subquery()
    stmt = select(subq.c.user_id, subq.c.avg_drip, subq.c.cnt).order_by(subq.c.avg_drip.desc()).limit(limit)
    res = await db.execute(stmt)
    rows = res.fetchall()
  except SQLAlchemyError:
    return LeaderboardResponse(scope=scope, entries=[], total_eligible=0)

  user_ids = [r.user_id for r in rows]
  users_map: dict[str, str] = {}
  if user_ids:
    try:
      ur = await db.execute(select(User.id, User.username, User.display_name).where(User.id.in_(user_ids)))
      for u in ur.fetchall():
        users_map[u.id] = u.username or u.display_name or "User"
    except SQLAlchemyError:
      ur = await db.execute(select(User.id, User.display_name).where(User.id.in_(user_ids)))
      for u in ur.fetchall():
        users_map[u.id] = u.display_name or "User"

  entries = [
    LeaderboardEntry(
      rank=idx + 1,
      user_id=r.user_id,
      display_name=users_map.get(r.user_id),
      avg_drip_score=round(float(r.avg_drip), 2),
      rating_count=int(r.cnt),
    )
    for idx, r in enumerate(rows)
  ]
  return LeaderboardResponse(scope=scope, entries=entries, total_eligible=len(rows))


@router.get("/me", response_model=UserRankingsResponse)
async def get_my_rankings(
  user_id: str | None = Query(None),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  user_id = _require_actor_user_id(auth, user_id)

  count_stmt = (
    select(func.count(OutfitScore.id))
    .join(Outfit, Outfit.id == OutfitScore.outfit_id)
    .where(Outfit.user_id == user_id)
  )
  count_res = await db.execute(count_stmt)
  total_count = count_res.scalar() or 0

  avg_stmt = (
    select(func.avg(OutfitScore.drip_score))
    .join(Outfit, Outfit.id == OutfitScore.outfit_id)
    .where(Outfit.user_id == user_id)
  )
  avg_res = await db.execute(avg_stmt)
  avg_scalar = avg_res.scalar()
  avg_drip = float(avg_scalar) if avg_scalar is not None else None

  if total_count < MIN_RATINGS_FOR_LEADERBOARD:
    return UserRankingsResponse(
      user_id=user_id,
      ratings_count=total_count,
      avg_drip_score=round(avg_drip, 2) if avg_drip else None,
      eligible_for_leaderboard=False,
      rankings=[],
    )

  profile_stmt = select(UserProfile.country).where(UserProfile.user_id == user_id)
  profile_res = await db.execute(profile_stmt)
  user_country = profile_res.scalar_one_or_none()

  scopes_to_check = [("global", None, None), ("yearly", None, None), ("monthly", None, None), ("weekly", None, None), ("daily", None, None)]
  if user_country:
    scopes_to_check.append(("country", user_country, None))
  for style_name in STYLE_SCOPES:
    scopes_to_check.append((f"style:{style_name}", None, None))

  rankings = []
  for scope, country, group_id in scopes_to_check:
    style_name = scope.split(":", 1)[1] if scope.startswith("style:") else None
    count_scope = "style" if style_name else scope
    cnt, avg = await _get_user_ratings_count(db, user_id, count_scope, country, group_id, style_name)
    if cnt < 1:
      rankings.append(UserRankingSummary(scope=scope, rank=None, total_eligible=0, avg_drip_score=round(avg, 2) if avg else None, rating_count=cnt))
      continue

    eligible_users = _eligible_users_subquery()
    base = (
      select(Outfit.user_id, func.avg(OutfitScore.drip_score).label("avg_drip"), func.count(OutfitScore.id).label("cnt"))
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .join(eligible_users, eligible_users.c.user_id == Outfit.user_id)
      .where(Outfit.user_id.isnot(None))
      .group_by(Outfit.user_id)
    )
    if scope not in ("global", "country", "style") and not style_name:
      start, _ = _time_bounds(scope)
      if start:
        base = base.where(OutfitScore.created_at >= start)
    if country:
      base = base.join(UserProfile, UserProfile.user_id == Outfit.user_id).where(UserProfile.country == country)
    if style_name:
      base = base.where(cast(Outfit.style_tags, JSONB).contains([style_name]))

    rank_subq = base.subquery()
    rres = await db.execute(select(rank_subq).order_by(desc(rank_subq.c.avg_drip)))
    rows = rres.fetchall()
    rank_val = None
    for i, r in enumerate(rows, 1):
      if r.user_id == user_id:
        rank_val = i
        break
    rankings.append(
      UserRankingSummary(
        scope=scope,
        rank=rank_val,
        total_eligible=len(rows),
        avg_drip_score=round(avg, 2) if avg else None,
        rating_count=cnt,
      )
    )

  return UserRankingsResponse(
    user_id=user_id,
    ratings_count=total_count,
    avg_drip_score=round(avg_drip, 2) if avg_drip else None,
    eligible_for_leaderboard=True,
    rankings=rankings,
  )


@router.post("/groups", response_model=CreateGroupResponse)
async def create_group(
  payload: CreateGroupRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, payload.user_id)
  code = _gen_group_code()
  group = RankingGroup(name=payload.name, code=code, created_by_user_id=actor_user_id)
  db.add(group)
  await db.flush()
  db.add(RankingGroupMember(group_id=group.id, user_id=actor_user_id))
  await db.commit()
  return CreateGroupResponse(group_id=group.id, code=code, name=group.name)


@router.post("/groups/join", response_model=JoinGroupResponse)
async def join_group(
  payload: JoinGroupRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, payload.user_id)
  res = await db.execute(select(RankingGroup).where(RankingGroup.code == payload.code.upper()))
  group = res.scalar_one_or_none()
  if not group:
    raise HTTPException(status_code=404, detail="Group not found")
  if group.created_by_user_id == actor_user_id:
    raise HTTPException(status_code=400, detail="You cannot join your own group with the invite code")

  existing = await db.execute(
    select(RankingGroupMember).where(
      and_(RankingGroupMember.group_id == group.id, RankingGroupMember.user_id == actor_user_id)
    )
  )
  if existing.scalar_one_or_none():
    await db.commit()
    return JoinGroupResponse(group_id=group.id, name=group.name, joined=False)

  db.add(RankingGroupMember(group_id=group.id, user_id=actor_user_id))
  await db.commit()
  return JoinGroupResponse(group_id=group.id, name=group.name, joined=True)


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(
  group_id: str,
  user_id: str | None = Query(None),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, user_id)
  res = await db.execute(select(RankingGroup).where(RankingGroup.id == group_id))
  group = res.scalar_one_or_none()
  if not group:
    raise HTTPException(status_code=404, detail="Group not found")
  if group.created_by_user_id != actor_user_id:
    raise HTTPException(status_code=403, detail="Only the creator can delete this group")

  await db.execute(RankingGroupMember.__table__.delete().where(RankingGroupMember.group_id == group_id))
  await db.execute(RankingGroup.__table__.delete().where(RankingGroup.id == group_id))
  await db.commit()


@router.get("/groups", response_model=list[GroupSummary])
async def list_groups(
  user_id: str | None = Query(None),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, user_id)
  stmt = (
    select(RankingGroup.id, RankingGroup.name, RankingGroup.code, RankingGroup.created_by_user_id)
    .join(RankingGroupMember, RankingGroupMember.group_id == RankingGroup.id)
    .where(RankingGroupMember.user_id == actor_user_id)
  )
  res = await db.execute(stmt)
  rows = res.fetchall()
  return [
    GroupSummary(id=r.id, name=r.name, code=r.code, is_owner=(r.created_by_user_id == actor_user_id))
    for r in rows
  ]


@router.get("/groups/{group_id}/leaderboard", response_model=LeaderboardResponse)
async def get_group_leaderboard(
  group_id: str,
  limit: int = Query(30, ge=1, le=100),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  return await get_leaderboard(scope="group", group_id=group_id, limit=limit, auth=auth, db=db)


@router.get("/groups/{group_id}/details", response_model=GroupDetailsResponse)
async def get_group_details(
  group_id: str,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  group_res = await db.execute(select(RankingGroup).where(RankingGroup.id == group_id))
  group = group_res.scalar_one_or_none()
  if not group:
    raise HTTPException(status_code=404, detail="Group not found")

  member_res = await db.execute(
    select(RankingGroupMember.id).where(
      RankingGroupMember.group_id == group_id,
      RankingGroupMember.user_id == auth.app_user_id,
    )
  )
  if member_res.scalar_one_or_none() is None:
    raise HTTPException(status_code=403, detail="You are not a member of this group")

  members_stmt = (
    select(
      User.id,
      User.username,
      User.display_name,
      func.count(OutfitScore.id).label("cnt"),
      func.avg(OutfitScore.drip_score).label("avg_drip"),
    )
    .join(RankingGroupMember, RankingGroupMember.user_id == User.id)
    .outerjoin(Outfit, Outfit.user_id == User.id)
    .outerjoin(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(RankingGroupMember.group_id == group_id)
    .group_by(User.id, User.username, User.display_name)
    .order_by(func.avg(OutfitScore.drip_score).desc().nullslast(), func.count(OutfitScore.id).desc())
  )
  try:
    members_res = await db.execute(members_stmt)
    members_rows = members_res.fetchall()
  except SQLAlchemyError:
    fallback_stmt = (
      select(
        User.id,
        User.display_name,
        func.count(OutfitScore.id).label("cnt"),
        func.avg(OutfitScore.drip_score).label("avg_drip"),
      )
      .join(RankingGroupMember, RankingGroupMember.user_id == User.id)
      .outerjoin(Outfit, Outfit.user_id == User.id)
      .outerjoin(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .where(RankingGroupMember.group_id == group_id)
      .group_by(User.id, User.display_name)
      .order_by(func.avg(OutfitScore.drip_score).desc().nullslast(), func.count(OutfitScore.id).desc())
    )
    members_res = await db.execute(fallback_stmt)
    members_rows = members_res.fetchall()

  members = [
    GroupMemberStanding(
      rank=idx,
      user_id=row.id,
      display_name=getattr(row, "username", None) or row.display_name or "User",
      avg_drip_score=round(float(row.avg_drip), 2) if row.avg_drip is not None else None,
      rating_count=int(row.cnt or 0),
    )
    for idx, row in enumerate(members_rows, start=1)
  ]

  return GroupDetailsResponse(group_id=group.id, name=group.name, code=group.code, members=members)
