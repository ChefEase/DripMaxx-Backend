from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Outfit, OutfitScore, UserSubscription

FREE_DAILY_LIMIT = 20
PAID_MONTHLY_LIMIT = 190


def _day_start_utc(now: datetime) -> datetime:
  return now.replace(hour=0, minute=0, second=0, microsecond=0)


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
    limit_type = "monthly"
  else:
    start = _day_start_utc(now)
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
