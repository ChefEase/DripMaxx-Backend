from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserRewardBalance, XpLedger

XP_PER_SCAN_REWARD = 500
SCAN_REWARD_AMOUNT = 10
SCAN_XP = 10
DAILY_CHALLENGE_PARTICIPATION_XP = 25
CHALLENGE_WIN_XP = 100
THREE_DAY_STREAK_XP = 20
INVITE_SIGNUP_XP = 100
OUTFIT_VOTE_XP = 5


async def get_or_create_balance(db: AsyncSession, user_id: str) -> UserRewardBalance:
  res = await db.execute(select(UserRewardBalance).where(UserRewardBalance.user_id == user_id))
  balance = res.scalar_one_or_none()
  if balance:
    return balance
  balance = UserRewardBalance(user_id=user_id, xp=0, scan_credits=0)
  db.add(balance)
  await db.flush()
  return balance


async def award_xp_once(
  db: AsyncSession,
  user_id: str,
  points: int,
  source_type: str,
  source_id: str | None = None,
  note: str | None = None,
) -> bool:
  if points == 0:
    return False
  try:
    async with db.begin_nested():
      ledger = XpLedger(
        user_id=user_id,
        points=points,
        source_type=source_type,
        source_id=source_id,
        note=note,
      )
      db.add(ledger)
      await db.flush()
  except IntegrityError:
    return False

  balance = await get_or_create_balance(db, user_id)
  balance.xp = int(balance.xp or 0) + points
  while balance.xp >= XP_PER_SCAN_REWARD:
    balance.xp -= XP_PER_SCAN_REWARD
    balance.scan_credits = int(balance.scan_credits or 0) + SCAN_REWARD_AMOUNT
  return True


async def add_scan_credits(db: AsyncSession, user_id: str, amount: int) -> None:
  balance = await get_or_create_balance(db, user_id)
  balance.scan_credits = int(balance.scan_credits or 0) + amount


async def consume_scan_credit_if_needed(db: AsyncSession, user_id: str, quota: dict) -> bool:
  if quota.get("remaining", 0) > 0 and quota.get("limit_type") != "bonus_scan_credits":
    return False
  balance = await get_or_create_balance(db, user_id)
  if int(balance.scan_credits or 0) < 1:
    return False
  balance.scan_credits = int(balance.scan_credits or 0) - 1
  return True
