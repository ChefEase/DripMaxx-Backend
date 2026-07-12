from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from appstoreserverlibrary.models.Environment import Environment
from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier, VerificationException
from loguru import logger
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.core.config import get_settings
from app.db.session import get_db
from app.models import BillingReceipt, UserSubscription
from app.schemas.billing import BillingStatusResponse, VerifyPurchaseRequest, VerifyPurchaseResponse
from app.services.usage_limits import get_scan_quota


router = APIRouter(prefix="/v1/billing", tags=["billing"])
settings = get_settings()


def _require_actor_user_id(auth: AuthContext, claimed_user_id: str | None) -> str:
  if claimed_user_id and claimed_user_id != auth.app_user_id:
    raise HTTPException(status_code=403, detail="user_id does not match authenticated user")
  return auth.app_user_id


def _parse_google_datetime(value: str | None) -> datetime | None:
  if not value:
    return None
  normalized = value.replace("Z", "+00:00")
  try:
    parsed = datetime.fromisoformat(normalized)
  except ValueError:
    return None
  if parsed.tzinfo is None:
    return parsed.replace(tzinfo=timezone.utc)
  return parsed


def _resolve_service_account_file() -> Path:
  if not settings.google_play_service_account_file:
    raise HTTPException(status_code=500, detail="GOOGLE_PLAY_SERVICE_ACCOUNT_FILE is not configured.")
  raw_path = Path(settings.google_play_service_account_file.strip()).expanduser()
  if raw_path.is_absolute():
    return raw_path.resolve()
  if raw_path.exists():
    return raw_path.resolve()
  api_root = Path(__file__).resolve().parents[2]
  return (api_root / raw_path).resolve()


def verify_with_google_play(purchase_token: str) -> dict:
  if not settings.google_play_package_name:
    raise HTTPException(status_code=500, detail="GOOGLE_PLAY_PACKAGE_NAME is not configured.")

  service_account_file = _resolve_service_account_file()
  if not service_account_file.exists():
    raise HTTPException(status_code=500, detail="Google Play service account file was not found.")

  credentials = service_account.Credentials.from_service_account_file(
    str(service_account_file),
    scopes=["https://www.googleapis.com/auth/androidpublisher"],
  )
  service = build("androidpublisher", "v3", credentials=credentials, cache_discovery=False)

  try:
    return (
      service.purchases()
      .subscriptionsv2()
      .get(packageName=settings.google_play_package_name, token=purchase_token)
      .execute()
    )
  except HttpError as exc:
    detail = "Google Play verification failed."
    status_code = exc.resp.status if exc.resp is not None else 502
    if status_code == 404:
      detail = "Purchase token was not found in Google Play."
      status_code = 400
    raise HTTPException(status_code=status_code, detail=detail) from exc


def verify_with_app_store(signed_transaction: str):
  if not settings.apple_root_certificates:
    raise HTTPException(status_code=500, detail="APPLE_ROOT_CERTIFICATES is not configured.")

  certificate_paths = [Path(value.strip()).expanduser() for value in settings.apple_root_certificates.split(",") if value.strip()]
  try:
    root_certificates = [path.read_bytes() for path in certificate_paths]
  except OSError as exc:
    raise HTTPException(status_code=500, detail="An Apple root certificate could not be read.") from exc

  # StoreKit test purchases are sandbox-signed; TestFlight and production use the
  # production environment. Validate against both, never by decoding an untrusted JWS.
  for environment in (Environment.PRODUCTION, Environment.SANDBOX):
    if environment == Environment.PRODUCTION and settings.apple_app_id is None:
      continue
    verifier = SignedDataVerifier(
      root_certificates,
      True,
      environment,
      settings.apple_bundle_id,
      settings.apple_app_id if environment == Environment.PRODUCTION else None,
    )
    try:
      return verifier.verify_and_decode_signed_transaction(signed_transaction)
    except VerificationException:
      continue
  raise HTTPException(status_code=400, detail="App Store transaction verification failed.")


@router.get("/status", response_model=BillingStatusResponse)
async def billing_status(
  user_id: str | None = Query(None),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, user_id)
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


@router.post("/verify-purchase", response_model=VerifyPurchaseResponse)
async def verify_purchase(
  payload: VerifyPurchaseRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  actor_user_id = _require_actor_user_id(auth, payload.user_id)
  platform = payload.platform.strip().lower()
  expected_product_id = (
    settings.premium_monthly_product_id_android
    if platform == "android"
    else settings.premium_monthly_product_id_ios
  )
  if payload.product_id != expected_product_id:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown product ID.")

  provided_token = payload.token or payload.purchase_token
  if not provided_token:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="token is required.",
    )

  if platform not in {"android", "ios"}:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Unsupported purchase platform.",
    )

  now = datetime.now(timezone.utc)
  if platform == "android":
    verified = verify_with_google_play(provided_token)
    line_items = verified.get("lineItems") or []
    if not line_items:
      raise HTTPException(status_code=400, detail="Google Play response did not include a subscription line item.")
    product_ids = {item.get("productId") for item in line_items if item.get("productId")}
    if payload.product_id not in product_ids:
      raise HTTPException(status_code=400, detail="Verified purchase does not match requested product ID.")
    if verified.get("subscriptionState") not in {"SUBSCRIPTION_STATE_ACTIVE", "SUBSCRIPTION_STATE_IN_GRACE_PERIOD"}:
      raise HTTPException(status_code=400, detail="Subscription is not active.")
    verified_purchase_token = provided_token
    verified_transaction_id = verified.get("latestOrderId")
    period_end = _parse_google_datetime(line_items[0].get("expiryTime")) or (now + timedelta(days=30))
    verification_mode = "google-play"
  else:
    verified = verify_with_app_store(provided_token)
    if verified.productId != payload.product_id:
      raise HTTPException(status_code=400, detail="Verified purchase does not match requested product ID.")
    if verified.revocationDate is not None:
      raise HTTPException(status_code=400, detail="App Store transaction was revoked.")
    if verified.expiresDate is None:
      raise HTTPException(status_code=400, detail="App Store subscription did not include an expiration date.")
    period_end = datetime.fromtimestamp(verified.expiresDate / 1000, tz=timezone.utc)
    if period_end <= now:
      raise HTTPException(status_code=400, detail="App Store subscription is expired.")
    verified_purchase_token = provided_token
    verified_transaction_id = verified.transactionId
    verification_mode = "app-store"

  if not verified_transaction_id:
    raise HTTPException(status_code=400, detail="Verified purchase did not include a transaction ID.")

  receipt_stmt = select(BillingReceipt).where(
    BillingReceipt.platform == platform,
    or_(
      BillingReceipt.purchase_token == verified_purchase_token,
      BillingReceipt.transaction_id == verified_transaction_id,
    ),
  )
  receipt_res = await db.execute(receipt_stmt)
  existing_receipt = receipt_res.scalar_one_or_none()

  if existing_receipt and existing_receipt.user_id != actor_user_id:
    raise HTTPException(
      status_code=status.HTTP_409_CONFLICT,
      detail="Already used",
    )

  sub_stmt = select(UserSubscription).where(UserSubscription.user_id == actor_user_id)
  sub_res = await db.execute(sub_stmt)
  sub = sub_res.scalar_one_or_none()
  if sub is None:
    sub = UserSubscription(
      user_id=actor_user_id,
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

  receipt_payload = {
    "platform": platform,
    "product_id": payload.product_id,
    "token": provided_token,
    "purchase_token": verified_purchase_token,
    "transaction_id": verified_transaction_id,
  }

  if existing_receipt is None:
    receipt = BillingReceipt(
      user_id=actor_user_id,
      platform=platform,
      product_id=payload.product_id,
      purchase_token=verified_purchase_token,
      transaction_id=verified_transaction_id,
      raw_receipt=receipt_payload,
      verified_at=now,
      expires_at=period_end,
    )
    db.add(receipt)
  else:
    existing_receipt.product_id = payload.product_id
    existing_receipt.purchase_token = verified_purchase_token
    existing_receipt.transaction_id = verified_transaction_id
    existing_receipt.raw_receipt = receipt_payload
    existing_receipt.verified_at = now
    existing_receipt.expires_at = period_end

  await db.commit()
  logger.info(f"user={actor_user_id} endpoint=/v1/billing/verify-purchase platform={platform} status=success")

  return VerifyPurchaseResponse(
    ok=True,
    plan="monthly",
    subscription_status="active",
    current_period_end=period_end.isoformat(),
    mode=verification_mode,
  )
