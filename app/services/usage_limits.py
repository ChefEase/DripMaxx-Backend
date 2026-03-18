from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Outfit, OutfitScore, UserSubscription

FREE_FIRST_SCANS = 3
FREE_DAILY_LIMIT = 1
PAID_MONTHLY_LIMIT = 999999


def _month_start_utc(now: datetime) -> datetime:
  return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def get_scan_quota(db: AsyncSession, user_id: str) -> dict:
  now = datetime.now(timezone.utc)
  sub_stmt = select(UserSubscription).where(UserSubscription.user_id == user_id)
  try:
    sub_res = await db.execute(sub_stmt)
    sub = sub_res.scalar_one_or_none()
  except SQLAlchemyError:
    # Billing table may not exist in environments where paywall is disabled.
    await db.rollback()
    sub = None

  is_paid = bool(
    sub
    and sub.plan == "monthly"
    and sub.status in ("active", "trialing")
    and (sub.current_period_end is None or sub.current_period_end >= now)
  )

  if is_paid:
    start = _month_start_utc(now)
    limit = PAID_MONTHLY_LIMIT
    limit_type = "unlimited"
  else:
    rolling_start = now.replace(microsecond=0) - timedelta(hours=24)
    total_count_stmt = (
      select(func.count(OutfitScore.id))
      .join(Outfit, Outfit.id == OutfitScore.outfit_id)
      .where(Outfit.user_id == user_id)
    )
    total_used = 0
    try:
      total_res = await db.execute(total_count_stmt)
      total_used = int(total_res.scalar() or 0)
    except SQLAlchemyError:
      await db.rollback()
    # Every user gets their first 3 scans total. After that, 1 scan per rolling 24h.
    if total_used < FREE_FIRST_SCANS:
      start = datetime(1970, 1, 1, tzinfo=timezone.utc)
      limit = FREE_FIRST_SCANS
      limit_type = "first_scans"
    else:
      start = rolling_start
      limit = FREE_DAILY_LIMIT
      limit_type = "daily"

  count_stmt = (
    select(func.count(OutfitScore.id))
    .join(Outfit, Outfit.id == OutfitScore.outfit_id)
    .where(Outfit.user_id == user_id, OutfitScore.created_at >= start)
  )
  try:
    count_res = await db.execute(count_stmt)
    used = int(count_res.scalar() or 0)
  except SQLAlchemyError:
    # Keep scoring available during billing table migrations/issues.
    await db.rollback()
    used = 0
  remaining = max(limit - used, 0)

  return {
    "plan": "monthly" if is_paid else "free",
    "subscription_status": sub.status if sub else "inactive",
    "limit_type": limit_type,
    "limit": limit,
    "used": used,
    "remaining": remaining,
    "allowed": remaining > 0,
  }
