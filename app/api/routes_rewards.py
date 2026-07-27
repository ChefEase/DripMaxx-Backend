from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.db.session import get_db
from app.models import UserBadge, XpLedger
from app.services.leaderboard_rewards import settle_recent_leaderboard_periods, sync_all_time_badges
from app.services.rewards import XP_PER_SCAN_REWARD, get_or_create_balance

router = APIRouter(prefix="/v1/rewards", tags=["rewards"])


@router.get("/me")
async def get_my_rewards(
  user_id: str | None = Query(None),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  if user_id and user_id != auth.app_user_id:
    raise HTTPException(status_code=403, detail="user_id does not match authenticated user")

  await settle_recent_leaderboard_periods(db)
  await sync_all_time_badges(db)
  balance = await get_or_create_balance(db, auth.app_user_id)
  ledger_res = await db.execute(
    select(XpLedger)
    .where(XpLedger.user_id == auth.app_user_id)
    .order_by(desc(XpLedger.created_at))
    .limit(20)
  )
  history = [
    {
      "id": row.id,
      "points": int(row.points or 0),
      "source_type": row.source_type,
      "note": row.note,
      "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    for row in ledger_res.scalars().all()
  ]

  xp = int(balance.xp or 0)
  badge_res = await db.execute(
    select(UserBadge)
    .where(UserBadge.user_id == auth.app_user_id)
    .order_by(desc(UserBadge.earned_at))
    .limit(50)
  )
  badges = [
    {
      "id": badge.id,
      "badge_key": badge.badge_key,
      "label": badge.label,
      "rank": badge.rank,
      "scope": badge.scope,
      "category": badge.category,
      "earned_at": badge.earned_at.isoformat() if badge.earned_at else None,
    }
    for badge in badge_res.scalars().all()
  ]
  return {
    "user_id": auth.app_user_id,
    "xp": xp,
    "scan_credits": int(balance.scan_credits or 0),
    "xp_per_scan_reward": XP_PER_SCAN_REWARD,
    "xp_until_next_reward": max(XP_PER_SCAN_REWARD - xp, 0),
    "history": history,
    "badges": badges,
  }
