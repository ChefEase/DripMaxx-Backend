from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import cast, delete, desc, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
  LeaderboardAward,
  Outfit,
  OutfitScore,
  UserBadge,
  UserProfile,
)
from app.services.rewards import add_scan_credits, award_xp_once

STYLE_SCOPES = ["Streetwear", "Minimal", "Vintage", "Luxury", "Y2K", "Casual"]
REWARDS = {
  "daily": ((75, 0), (50, 0), (25, 0)),
  "weekly": ((200, 3), (125, 2), (75, 1)),
  "monthly": ((500, 10), (300, 6), (200, 3)),
  "monthly_category": ((250, 5), (150, 3), (100, 2)),
}
MEDALS = {1: ("gold", "Gold"), 2: ("silver", "Silver"), 3: ("bronze", "Bronze")}


def _previous_period(scope: str, now: datetime) -> tuple[datetime, datetime]:
  if scope == "daily":
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return end - timedelta(days=1), end
  if scope == "weekly":
    end = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return end - timedelta(days=7), end
  end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
  previous_month_day = end - timedelta(days=1)
  start = previous_month_day.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
  return start, end


async def _standings(
  db: AsyncSession,
  start: datetime,
  end: datetime,
  country: str | None = None,
  style: str | None = None,
):
  query = (
    select(
      Outfit.user_id,
      func.avg(OutfitScore.drip_score).label("average"),
      func.count(OutfitScore.id).label("count"),
    )
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(
      Outfit.user_id.isnot(None),
      OutfitScore.created_at >= start,
      OutfitScore.created_at < end,
    )
    .group_by(Outfit.user_id)
    .order_by(desc("average"), desc("count"), Outfit.user_id.asc())
    .limit(3)
  )
  if country:
    query = query.join(UserProfile, UserProfile.user_id == Outfit.user_id).where(UserProfile.country == country)
  if style:
    query = query.where(cast(Outfit.style_tags, JSONB).contains([style]))
  return (await db.execute(query)).fetchall()


async def _award_standings(
  db: AsyncSession,
  scope: str,
  category: str,
  start: datetime,
  end: datetime,
  standings,
  reward_key: str,
) -> int:
  created = 0
  for rank, row in enumerate(standings, start=1):
    xp, scans = REWARDS[reward_key][rank - 1]
    try:
      async with db.begin_nested():
        award = LeaderboardAward(
          user_id=row.user_id,
          scope=scope,
          category=category,
          period_start=start,
          period_end=end,
          rank=rank,
          xp_awarded=xp,
          scan_credits_awarded=scans,
        )
        db.add(award)
        await db.flush()
    except IntegrityError:
      continue

    medal_key, medal_label = MEDALS[rank]
    category_label = f" · {category}" if category else ""
    period_label = start.strftime("%b %Y") if scope.startswith("monthly") else start.strftime("%Y-%m-%d")
    db.add(
      UserBadge(
        user_id=row.user_id,
        award_id=award.id,
        badge_key=f"{scope}_{medal_key}",
        label=f"{medal_label} · {scope.replace('_', ' ').title()}{category_label} · {period_label}",
        rank=rank,
        scope=scope,
        category=category,
      )
    )
    source_id = award.id
    await award_xp_once(db, row.user_id, xp, "leaderboard_award", source_id, f"#{rank} {scope} leaderboard")
    if scans:
      await add_scan_credits(db, row.user_id, scans)
    created += 1
  return created


async def settle_recent_leaderboard_periods(db: AsyncSession) -> int:
  """Settle the most recently completed day, week and month exactly once."""
  now = datetime.now(timezone.utc)
  created = 0
  for scope in ("daily", "weekly", "monthly"):
    start, end = _previous_period(scope, now)
    standings = await _standings(db, start, end)
    created += await _award_standings(db, scope, "", start, end, standings, scope)

  month_start, month_end = _previous_period("monthly", now)
  countries = (
    await db.execute(
      select(UserProfile.country)
      .join(Outfit, Outfit.user_id == UserProfile.user_id)
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .where(
        UserProfile.country.isnot(None),
        OutfitScore.created_at >= month_start,
        OutfitScore.created_at < month_end,
      )
      .distinct()
    )
  ).scalars().all()
  for country in countries:
    standings = await _standings(db, month_start, month_end, country=country)
    created += await _award_standings(
      db, "monthly_country", str(country), month_start, month_end, standings, "monthly_category"
    )
  for style in STYLE_SCOPES:
    standings = await _standings(db, month_start, month_end, style=style)
    created += await _award_standings(
      db, "monthly_style", style, month_start, month_end, standings, "monthly_category"
    )
  await db.commit()
  return created


async def sync_all_time_badges(db: AsyncSession) -> None:
  """Replace the live all-time podium badges with the current eligible top three."""
  eligible = (
    select(Outfit.user_id.label("user_id"))
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id.isnot(None))
    .group_by(Outfit.user_id)
    .having(func.count(OutfitScore.id) >= 10)
    .subquery()
  )
  standings = (
    await db.execute(
      select(
        Outfit.user_id,
        func.avg(OutfitScore.drip_score).label("average"),
        func.count(OutfitScore.id).label("count"),
      )
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .join(eligible, eligible.c.user_id == Outfit.user_id)
      .group_by(Outfit.user_id)
      .order_by(desc("average"), desc("count"), Outfit.user_id.asc())
      .limit(3)
    )
  ).fetchall()

  await db.execute(delete(UserBadge).where(UserBadge.scope == "global", UserBadge.is_current.is_(True)))
  for rank, row in enumerate(standings, start=1):
    medal_key, medal_label = MEDALS[rank]
    db.add(
      UserBadge(
        user_id=row.user_id,
        award_id=None,
        badge_key=f"global_current_{medal_key}",
        label=f"{medal_label} · All-Time Current #{rank}",
        rank=rank,
        scope="global",
        category="",
        is_current=True,
      )
    )
  await db.commit()
