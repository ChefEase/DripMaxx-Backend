from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.models import UserSubscription
from app.schemas.billing import BillingStatusResponse, VerifyPurchaseRequest, VerifyPurchaseResponse
from app.services.usage_limits import get_scan_quota

router = APIRouter(prefix="/v1/billing", tags=["billing"])
settings = get_settings()


@router.get("/status", response_model=BillingStatusResponse)
async def billing_status(user_id: str = Query(...), db: AsyncSession = Depends(get_db)):
  quota = await get_scan_quota(db, user_id)
  return BillingStatusResponse(
    user_id=user_id,
    plan=quota["plan"],
    subscription_status=quota["subscription_status"],
    limit_type=quota["limit_type"],
    limit=quota["limit"],
    used=quota["used"],
    remaining=quota["remaining"],
  )


@router.post("/verify-purchase", response_model=VerifyPurchaseResponse)
async def verify_purchase(payload: VerifyPurchaseRequest, db: AsyncSession = Depends(get_db)):
  platform = payload.platform.strip().lower()
  expected_product_id = (
    settings.premium_monthly_product_id_android
    if platform == "android"
    else settings.premium_monthly_product_id_ios
  )
  if payload.product_id != expected_product_id:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Unknown product ID.",
    )

  if not settings.billing_dev_mode:
    raise HTTPException(
      status_code=status.HTTP_501_NOT_IMPLEMENTED,
      detail="Store purchase verification is not wired for production yet.",
    )

  now = datetime.now(timezone.utc)
  period_end = now + timedelta(days=30)
  sub_stmt = select(UserSubscription).where(UserSubscription.user_id == payload.user_id)
  sub_res = await db.execute(sub_stmt)
  sub = sub_res.scalar_one_or_none()
  if sub is None:
    sub = UserSubscription(
      user_id=payload.user_id,
      plan="monthly",
      status="active",
      current_period_start=now,
      current_period_end=period_end,
    )
    db.add(sub)
  else:
    sub.plan = "monthly"
    sub.status = "active"
    sub.current_period_start = now
    sub.current_period_end = period_end
    sub.updated_at = now

  await db.commit()

  return VerifyPurchaseResponse(
    ok=True,
    plan="monthly",
    subscription_status="active",
    current_period_end=period_end.isoformat(),
    mode="dev",
  )
