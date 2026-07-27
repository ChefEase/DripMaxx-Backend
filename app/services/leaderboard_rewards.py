from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import cast, delete, desc, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
  LeaderboardAward,
  CommunityNews,
  Outfit,
  OutfitScore,
  User,
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
  if scope == "yearly":
    end = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return end.replace(year=end.year - 1), end
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
  minimum_scans: int = 1,
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
    .having(func.count(OutfitScore.id) >= minimum_scans)
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
  await _create_podium_news(db, scope, category, start, end, standings)
  return created


async def _create_podium_news(
  db: AsyncSession,
  scope: str,
  category: str,
  start: datetime,
  end: datetime,
  standings,
) -> None:
  if not standings:
    return
  news_key = f"podium:{scope}:{category}:{start.isoformat()}"
  if (
    await db.execute(select(CommunityNews.id).where(CommunityNews.news_key == news_key))
  ).scalar_one_or_none():
    return

  user_ids = [row.user_id for row in standings]
  user_rows = await db.execute(
    select(User.id, User.username, User.display_name).where(User.id.in_(user_ids))
  )
  names = {row.id: row.username or row.display_name or "Community member" for row in user_rows.fetchall()}
  podium = [
    {
      "rank": rank,
      "username": names.get(row.user_id, "Community member"),
      "score": round(float(row.average) * 10),
    }
    for rank, row in enumerate(standings, start=1)
  ]

  winner_id = standings[0].user_id
  winner_outfit = (
    await db.execute(
      select(Outfit.image_url, OutfitScore)
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .where(
        Outfit.user_id == winner_id,
        OutfitScore.created_at >= start,
        OutfitScore.created_at < end,
      )
      .order_by(OutfitScore.drip_score.desc())
      .limit(1)
    )
  ).first()
  image_url = winner_outfit[0] if winner_outfit else None
  strengths: list[str] = []
  if winner_outfit:
    score = winner_outfit[1]
    metrics = [
      ("Color coordination", score.color_match),
      ("Strong fit", score.fit_quality),
      ("Trend awareness", score.trend_score),
      ("Body compatibility", score.body_compatibility),
      ("Style match", score.style_match),
    ]
    strengths = [
      label for label, _ in sorted(metrics, key=lambda item: float(item[1] or 0), reverse=True)[:3]
    ]

  readable_scope = scope.replace("_", " ").title()
  category_suffix = f" · {category}" if category else ""
  title = f"{readable_scope}{category_suffix} Top 3"
  winner = podium[0]
  caption = (
    f"🔥 COMMUNITY FIT\n\n👤 @{winner['username']}\n"
    f"Score: {winner['score']}/100\n\n"
    f"Strengths:\n" + "\n".join(f"✓ {item}" for item in strengths) +
    f"\n\n{title}\n#DripMaxx"
  )
  lifetime = {
    "daily": timedelta(days=2),
    "weekly": timedelta(days=8),
    "monthly": timedelta(days=35),
    "monthly_country": timedelta(days=35),
    "monthly_style": timedelta(days=35),
    "yearly": timedelta(days=370),
  }.get(scope, timedelta(days=35))
  db.add(
    CommunityNews(
      news_key=news_key,
      kind="podium",
      scope=scope,
      category=category,
      audience_country=category if scope == "monthly_country" else None,
      eyebrow="🔥 COMMUNITY FIT",
      title=title,
      caption=caption,
      image_url=image_url,
      content={"podium": podium, "strengths": strengths},
      published_at=end,
      expires_at=end + lifetime,
    )
  )


async def _create_glow_up_news(
  db: AsyncSession,
  scope: str,
  start: datetime,
  end: datetime,
) -> None:
  news_key = f"glow-up:{scope}:{start.isoformat()}"
  if (
    await db.execute(select(CommunityNews.id).where(CommunityNews.news_key == news_key))
  ).scalar_one_or_none():
    return
  rows = (
    await db.execute(
      select(Outfit.user_id, Outfit.image_url, OutfitScore.drip_score, OutfitScore.created_at)
      .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
      .where(
        Outfit.user_id.isnot(None),
        OutfitScore.created_at >= start,
        OutfitScore.created_at < end,
      )
      .order_by(Outfit.user_id, OutfitScore.created_at.asc())
    )
  ).fetchall()
  by_user: dict[str, list] = {}
  for row in rows:
    by_user.setdefault(row.user_id, []).append(row)
  candidates = [
    (float(scans[-1].drip_score) - float(scans[0].drip_score), user_id, scans[0], scans[-1])
    for user_id, scans in by_user.items()
    if len(scans) >= 2 and float(scans[-1].drip_score) > float(scans[0].drip_score)
  ]
  if not candidates:
    return
  improvement, user_id, before, after = max(candidates, key=lambda item: item[0])
  user = (
    await db.execute(select(User.username, User.display_name).where(User.id == user_id))
  ).first()
  username = (user[0] or user[1] or "Community member") if user else "Community member"
  before_score = round(float(before.drip_score) * 10)
  after_score = round(float(after.drip_score) * 10)
  readable_scope = scope.title()
  lifetime = {"weekly": timedelta(days=8), "monthly": timedelta(days=35), "yearly": timedelta(days=370)}[scope]
  db.add(
    CommunityNews(
      news_key=news_key,
      kind="glow_up",
      scope=scope,
      category="Biggest Glow-Up",
      eyebrow="✨ BIGGEST GLOW-UP",
      title=f"{readable_scope} Biggest Glow-Up",
      caption=(
        f"✨ BIGGEST GLOW-UP\n\n👤 @{username}\n\n"
        f"Before: {before_score}/100\nAfter: {after_score}/100\n"
        f"Improvement: +{round(improvement * 10)} points\n\n"
        "Consistency changed the score. Keep scanning, learning, and leveling up.\n\n#DripMaxx"
      ),
      image_url=after.image_url,
      content={
        "username": username,
        "before_image_url": before.image_url,
        "after_image_url": after.image_url,
        "before_score": before_score,
        "after_score": after_score,
        "improvement": round(improvement * 10),
      },
      published_at=end,
      expires_at=end + lifetime,
    )
  )


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
  year_start, year_end = _previous_period("yearly", now)
  year_standings = await _standings(db, year_start, year_end, minimum_scans=10)
  await _create_podium_news(db, "yearly", "", year_start, year_end, year_standings)
  for glow_scope in ("weekly", "monthly", "yearly"):
    glow_start, glow_end = _previous_period(glow_scope, now)
    await _create_glow_up_news(db, glow_scope, glow_start, glow_end)
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
