from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.models import UserSubscription
from app.schemas.billing import (
  BillingStatusResponse,
  CheckoutSessionRequest,
  CheckoutSessionResponse,
)
from app.services.usage_limits import get_scan_quota

router = APIRouter(prefix="/v1/billing", tags=["billing"])

try:
  import stripe
except Exception:  # pragma: no cover
  stripe = None


def _require_stripe():
  settings = get_settings()
  if stripe is None:
    raise HTTPException(status_code=500, detail="Stripe SDK is not installed on the server")
  if not settings.stripe_secret_key:
    raise HTTPException(status_code=500, detail="Stripe is not configured")
  stripe.api_key = settings.stripe_secret_key
  return settings


async def _get_or_create_subscription(db: AsyncSession, user_id: str) -> UserSubscription:
  stmt = select(UserSubscription).where(UserSubscription.user_id == user_id)
  res = await db.execute(stmt)
  sub = res.scalar_one_or_none()
  if sub:
    return sub
  sub = UserSubscription(user_id=user_id, plan="free", status="inactive")
  db.add(sub)
  await db.flush()
  return sub


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


@router.post("/checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(payload: CheckoutSessionRequest, db: AsyncSession = Depends(get_db)):
  settings = _require_stripe()
  if not settings.stripe_monthly_price_id:
    raise HTTPException(status_code=500, detail="Stripe monthly price ID is not configured")

  sub = await _get_or_create_subscription(db, payload.user_id)
  customer_id = sub.stripe_customer_id
  if not customer_id:
    customer = stripe.Customer.create(
      email=payload.email,
      metadata={"user_id": payload.user_id},
    )
    customer_id = customer["id"]
    sub.stripe_customer_id = customer_id
    await db.commit()

  session = stripe.checkout.Session.create(
    mode="subscription",
    customer=customer_id,
    client_reference_id=payload.user_id,
    line_items=[{"price": settings.stripe_monthly_price_id, "quantity": 1}],
    success_url=settings.stripe_success_url,
    cancel_url=settings.stripe_cancel_url,
  )

  return CheckoutSessionResponse(checkout_url=session["url"], session_id=session["id"])


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
  settings = _require_stripe()
  payload = await request.body()
  sig_header = request.headers.get("stripe-signature")
  if not settings.stripe_webhook_secret:
    raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured")
  try:
    event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
  except Exception:
    raise HTTPException(status_code=400, detail="Invalid webhook signature")

  event_type = event["type"]
  data = event["data"]["object"]

  if event_type == "checkout.session.completed":
    user_id = data.get("client_reference_id")
    if user_id:
      sub = await _get_or_create_subscription(db, user_id)
      sub.plan = "monthly"
      sub.status = "active"
      sub.stripe_customer_id = data.get("customer")
      if data.get("subscription"):
        sub.stripe_subscription_id = data.get("subscription")
      await db.commit()

  if event_type in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
    stripe_subscription_id = data.get("id")
    stripe_customer_id = data.get("customer")
    stmt = select(UserSubscription).where(
      (UserSubscription.stripe_subscription_id == stripe_subscription_id)
      | (UserSubscription.stripe_customer_id == stripe_customer_id)
    )
    res = await db.execute(stmt)
    sub = res.scalar_one_or_none()
    if sub:
      status = data.get("status", "inactive")
      sub.status = status
      sub.plan = "monthly" if status in ("active", "trialing", "past_due", "incomplete") else "free"
      sub.stripe_customer_id = stripe_customer_id
      sub.stripe_subscription_id = stripe_subscription_id
      period_start = data.get("current_period_start")
      period_end = data.get("current_period_end")
      if period_start:
        sub.current_period_start = datetime.fromtimestamp(period_start, tz=timezone.utc)
      if period_end:
        sub.current_period_end = datetime.fromtimestamp(period_end, tz=timezone.utc)
      await db.commit()

  return {"received": True}
