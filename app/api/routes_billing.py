from datetime import datetime, timezone
from urllib.parse import quote

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.core.config import get_settings
from app.db.session import get_db
from app.models import UserSubscription
from app.schemas.billing import (
  BillingStatusResponse,
  RevenueCatSyncRequest,
  RevenueCatSyncResponse,
)
from app.services.usage_limits import get_scan_quota


router = APIRouter(prefix="/v1/billing", tags=["billing"])
settings = get_settings()


def _require_actor_user_id(auth: AuthContext, claimed_user_id: str | None) -> str:
  if claimed_user_id and claimed_user_id != auth.app_user_id:
    raise HTTPException(status_code=403, detail="user_id does not match authenticated user")
  return auth.app_user_id


def _parse_datetime(value: str | None) -> datetime | None:
  if not value:
    return None
  try:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
  except ValueError:
    return None
  return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


async def _get_revenuecat_customer(app_user_id: str) -> dict:
  if not settings.revenuecat_secret_api_key:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="REVENUECAT_SECRET_API_KEY is not configured on the backend.",
    )

  def _request():
    return requests.get(
      f"https://api.revenuecat.com/v1/subscribers/{quote(app_user_id, safe='')}",
      headers={
        "Authorization": f"Bearer {settings.revenuecat_secret_api_key}",
        "Accept": "application/json",
      },
      timeout=15,
    )

  try:
    response = await run_in_threadpool(_request)
  except requests.RequestException as exc:
    raise HTTPException(status_code=502, detail="RevenueCat could not be reached.") from exc
  if response.status_code not in {200, 201}:
    logger.warning("RevenueCat customer lookup failed status={}", response.status_code)
    raise HTTPException(status_code=502, detail="RevenueCat customer verification failed.")
  return response.json()


async def _sync_revenuecat_subscription(db: AsyncSession, user_id: str) -> tuple[bool, datetime | None]:
  customer = await _get_revenuecat_customer(user_id)
  entitlement = (
    customer.get("subscriber", {})
    .get("entitlements", {})
    .get(settings.revenuecat_entitlement_id)
  )
  expires_at = _parse_datetime(entitlement.get("expires_date")) if entitlement else None
  grace_expires_at = _parse_datetime(entitlement.get("grace_period_expires_date")) if entitlement else None
  effective_end = grace_expires_at or expires_at
  now = datetime.now(timezone.utc)
  active = bool(entitlement) and (effective_end is None or effective_end > now)

  result = await db.execute(select(UserSubscription).where(UserSubscription.user_id == user_id))
  subscription = result.scalar_one_or_none()
  if subscription is None:
    subscription = UserSubscription(user_id=user_id)
    db.add(subscription)
  subscription.plan = "monthly" if active else "free"
  subscription.status = "active" if active else "inactive"
  subscription.current_period_end = effective_end
  subscription.updated_at = now
  await db.commit()
  return active, effective_end


@router.get("/status", response_model=BillingStatusResponse)
async def billing_status(
  user_id: str | None = Query(None),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, user_id)
  # Keep renewals, cancellations, refunds, and cross-platform restores current.
  if settings.revenuecat_secret_api_key:
    try:
      await _sync_revenuecat_subscription(db, actor_user_id)
    except HTTPException as exc:
      logger.warning("RevenueCat status refresh failed user={} status={}", actor_user_id, exc.status_code)
  quota = await get_scan_quota(db, actor_user_id)
  return BillingStatusResponse(
    user_id=actor_user_id,
    plan=quota["plan"],
    subscription_status=quota["subscription_status"],
    limit_type=quota["limit_type"],
    limit=quota["limit"],
    used=quota["used"],
    remaining=quota["remaining"],
  )


@router.post("/sync-revenuecat", response_model=RevenueCatSyncResponse)
async def sync_revenuecat(
  payload: RevenueCatSyncRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, payload.user_id)
  if payload.platform.strip().lower() not in {"ios", "android"}:
    raise HTTPException(status_code=400, detail="Unsupported RevenueCat platform.")
  active, period_end = await _sync_revenuecat_subscription(db, actor_user_id)
  if not active:
    raise HTTPException(status_code=402, detail="RevenueCat entitlement is not active.")
  logger.info("user={} endpoint=/v1/billing/sync-revenuecat status=active", actor_user_id)
  return RevenueCatSyncResponse(
    ok=True,
    plan="monthly",
    subscription_status="active",
    current_period_end=period_end.isoformat() if period_end else None,
    mode="revenuecat",
  )
