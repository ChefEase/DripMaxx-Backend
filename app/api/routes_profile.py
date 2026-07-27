from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, desc, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from supabase import create_client as create_supabase_client

from app.core.config import get_settings
from app.core.auth import AuthContext, require_auth
from app.db.session import get_db
from app.models.entities import User, UserProfile, Outfit, OutfitScore, DripScoreHistory, StyleDNA
from app.schemas.profile import (
  DeleteAccountRequest,
  DeleteAccountResponse,
  ProfileSyncRequest,
  ProfileSyncResponse,
  StyleDNAResponse,
)

router = APIRouter(prefix="/v1/profile", tags=["profile"])
settings = get_settings()


def _average(values: list[float]) -> float:
  return sum(values) / len(values) if values else 0.0


def _improvement(values: list[float]) -> float:
  """Compare the newest and oldest up-to-three scans to reduce one-scan noise."""
  if len(values) < 2:
    return 0.0
  window = min(3, max(1, len(values) // 2))
  return _average(values[-window:]) - _average(values[:window])


def _current_streak(scanned_at: list[datetime]) -> int:
  days = sorted({value.date() for value in scanned_at if value}, reverse=True)
  if not days:
    return 0
  today = datetime.now(timezone.utc).date()
  if days[0] < today - timedelta(days=1):
    return 0
  streak = 1
  for previous, current in zip(days, days[1:]):
    if previous - current != timedelta(days=1):
      break
    streak += 1
  return streak


def _visibility_flag(value: str | None) -> bool:
  """Map visibility string to boolean column (True=public, False=private/friends)."""
  if value is None:
    return True
  return value == "public"


def _visibility_mode(value: str | None) -> str:
  if value in ("public", "friends_only", "private"):
    return value
  return "public"


def _storage_path_from_url(image_url: str | None, bucket: str) -> str | None:
  if not image_url or image_url.startswith("uploaded://"):
    return None
  try:
    parsed = urlparse(image_url)
    marker = f"/storage/v1/object/public/{bucket}/"
    if marker in parsed.path:
      return parsed.path.split(marker, 1)[1]
  except Exception:
    return None
  return None


def _delete_storage_objects(paths: list[str]) -> None:
  if not paths or not settings.supabase_url or not settings.supabase_service_key:
    return
  client = create_supabase_client(settings.supabase_url, settings.supabase_service_key)
  bucket = settings.supabase_bucket or "outfits"
  unique_paths = list(dict.fromkeys([p for p in paths if p]))
  for idx in range(0, len(unique_paths), 100):
    batch = unique_paths[idx : idx + 100]
    try:
      client.storage.from_(bucket).remove(batch)
    except Exception as exc:
      logger.warning(f"Supabase storage delete failed for batch size={len(batch)}: {exc}")


async def _get_or_create_user(
  db: AsyncSession,
  user_id: str,
  auth_id: str | None = None,
  email: str | None = None,
  display_name: str | None = None,
  username: str | None = None,
):
  stmt = select(User).where(User.id == user_id)
  res = await db.execute(stmt)
  user = res.scalar_one_or_none()
  if user:
    if auth_id and not user.auth_id:
      user.auth_id = auth_id
    return user
  user = User(id=user_id, auth_id=auth_id, email=email, display_name=display_name, username=username)
  db.add(user)
  try:
    await db.flush()
  except IntegrityError:
    await db.rollback()
    res = await db.execute(stmt)
    existing = res.scalar_one_or_none()
    if not existing:
      raise
    if auth_id and not existing.auth_id:
      existing.auth_id = auth_id
    return existing
  return user


def _require_actor_user_id(auth: AuthContext, claimed_user_id: str | None) -> str:
  if claimed_user_id and claimed_user_id.strip() and claimed_user_id != auth.app_user_id:
    raise HTTPException(status_code=403, detail="user_id does not match authenticated user")
  return auth.app_user_id


@router.post("/sync", response_model=ProfileSyncResponse)
async def sync_profile(
  payload: ProfileSyncRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  user_id = _require_actor_user_id(auth, payload.user_id)

  # Upsert user
  user = await _get_or_create_user(db, user_id, auth.auth_user_id, payload.email, payload.display_name, payload.username)
  if payload.username:
    normalized_username = payload.username.strip().lower()
    user.username = normalized_username
    if not payload.display_name:
      user.display_name = normalized_username
  if payload.email:
    user.email = payload.email
  if payload.display_name:
    user.display_name = payload.display_name
  if payload.avatar_url:
    user.avatar_url = payload.avatar_url

  # Upsert profile
  stmt = select(UserProfile).where(UserProfile.user_id == user_id)
  res = await db.execute(stmt)
  profile = res.scalar_one_or_none()
  if not profile:
    profile = UserProfile(
      user_id=user_id,
      style_preference=",".join(payload.style_preferences) if payload.style_preferences else "",
      height_cm=float(payload.user_height) if payload.user_height else None,
      body_type=payload.user_body_type,
      gender_style_preference=payload.gender_style_preference,
      country=payload.country,
      locale=payload.locale,
      profile_visibility=_visibility_flag(payload.profile_visibility),
      profile_visibility_mode=_visibility_mode(payload.profile_visibility),
    )
    db.add(profile)
  else:
    if payload.style_preferences is not None:
      profile.style_preference = ",".join(payload.style_preferences)
    if payload.user_height is not None:
      profile.height_cm = float(payload.user_height)
    if payload.user_body_type is not None:
      profile.body_type = payload.user_body_type
    if payload.gender_style_preference is not None:
      profile.gender_style_preference = payload.gender_style_preference
    if payload.country is not None:
      profile.country = payload.country
    if payload.locale is not None:
      profile.locale = payload.locale
    if payload.profile_visibility is not None:
      profile.profile_visibility = _visibility_flag(payload.profile_visibility)
      profile.profile_visibility_mode = _visibility_mode(payload.profile_visibility)

  try:
    await db.commit()
  except IntegrityError:
    await db.rollback()
    raise HTTPException(status_code=409, detail="Username already taken")
  logger.info(
    "profile_sync ok user_id={} body_type={} gender_style={}",
    user_id,
    payload.user_body_type.value if payload.user_body_type else None,
    payload.gender_style_preference.value if payload.gender_style_preference else None,
  )
  return ProfileSyncResponse(user_id=user_id)


@router.post("/delete-account", response_model=DeleteAccountResponse)
async def delete_account(
  payload: DeleteAccountRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  user_id = _require_actor_user_id(auth, payload.user_id)

  stmt = select(User).where(User.id == user_id)
  res = await db.execute(stmt)
  user = res.scalar_one_or_none()
  if not user:
    raise HTTPException(status_code=404, detail="User not found")

  bucket = settings.supabase_bucket or "outfits"
  outfit_stmt = select(Outfit.image_url).where(Outfit.user_id == user_id)
  outfit_res = await db.execute(outfit_stmt)
  storage_paths = [
    path
    for path in (
      _storage_path_from_url(str(row[0]) if row[0] is not None else None, bucket)
      for row in outfit_res.fetchall()
    )
    if path
  ]

  _delete_storage_objects(storage_paths)

  auth_deleted = False
  if settings.supabase_url and settings.supabase_service_key:
    try:
      admin_client = create_supabase_client(settings.supabase_url, settings.supabase_service_key)
      admin_client.auth.admin.delete_user(auth.auth_user_id)
      auth_deleted = True
    except Exception as exc:
      logger.warning(f"Supabase auth delete failed for user_id={user_id}: {exc}")

  await db.delete(user)
  await db.commit()
  return DeleteAccountResponse(ok=True, auth_deleted=auth_deleted)


@router.get("/history", response_model=dict)
async def profile_history(
  user_id: str | None = None,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  """Return recent outfits and drip score history for a user."""
  user_id = _require_actor_user_id(auth, user_id)

  rec_stmt = (
    select(Outfit.id, Outfit.image_url, Outfit.scanned_at, OutfitScore.drip_score)
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id, isouter=True)
    .where(Outfit.user_id == user_id)
    .order_by(desc(Outfit.scanned_at))
    .limit(10)
  )
  rec_res = await db.execute(rec_stmt)
  recent = [
    {
      "id": str(r.id),
      "image_url": r.image_url,
      "scanned_at": r.scanned_at.isoformat() if r.scanned_at else None,
      "drip_score": float(r.drip_score) if r.drip_score is not None else None,
    }
    for r in rec_res.fetchall()
  ]

  best_stmt = (
    select(Outfit.id, Outfit.image_url, Outfit.scanned_at, OutfitScore.drip_score)
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id == user_id)
    .order_by(desc(OutfitScore.drip_score), desc(Outfit.scanned_at))
    .limit(1)
  )
  best_res = await db.execute(best_stmt)
  best_row = best_res.fetchone()
  best_outfit = (
    {
      "id": str(best_row.id),
      "image_url": best_row.image_url,
      "scanned_at": best_row.scanned_at.isoformat() if best_row.scanned_at else None,
      "drip_score": float(best_row.drip_score) if best_row.drip_score is not None else None,
    }
    if best_row
    else None
  )

  hist_stmt = (
    select(DripScoreHistory.recorded_at, DripScoreHistory.drip_score)
    .where(DripScoreHistory.user_id == user_id)
    .order_by(desc(DripScoreHistory.recorded_at))
    .limit(30)
  )
  hist_res = await db.execute(hist_stmt)
  history = [
    {
      "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
      "drip_score": float(r.drip_score) if r.drip_score is not None else None,
    }
    for r in hist_res.fetchall()
  ]

  score_cards_stmt = (
    select(
      Outfit.id,
      Outfit.image_url,
      Outfit.scanned_at,
      OutfitScore.drip_score,
      OutfitScore.color_match,
      OutfitScore.fit_quality,
      OutfitScore.body_compatibility,
      OutfitScore.trend_score,
      OutfitScore.style_match,
    )
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id == user_id)
    .order_by(desc(Outfit.scanned_at))
    .limit(30)
  )
  score_cards_res = await db.execute(score_cards_stmt)
  score_cards = [
    {
      "outfit_id": str(r.id),
      "image_url": r.image_url,
      "scanned_at": r.scanned_at.isoformat() if r.scanned_at else None,
      "drip_score": float(r.drip_score) if r.drip_score is not None else None,
      "breakdown": {
        "color_match": float(r.color_match) if r.color_match is not None else None,
        "fit_quality": float(r.fit_quality) if r.fit_quality is not None else None,
        "body_compatibility": float(r.body_compatibility) if r.body_compatibility is not None else None,
        "trend_score": float(r.trend_score) if r.trend_score is not None else None,
        "style_match": float(r.style_match) if r.style_match is not None else None,
      },
    }
    for r in score_cards_res.fetchall()
  ]

  profile_stmt = select(UserProfile.profile_visibility, UserProfile.profile_visibility_mode).where(UserProfile.user_id == user_id)
  profile_res = await db.execute(profile_stmt)
  profile_row = profile_res.fetchone()
  if profile_row and profile_row.profile_visibility_mode in ("public", "friends_only", "private"):
    profile_visibility = profile_row.profile_visibility_mode
  else:
    profile_visibility = "public" if (profile_row and profile_row.profile_visibility) else "private"

  return {
    "recent_outfits": recent,
    "best_outfit": best_outfit,
    "history": list(reversed(history)),
    "score_cards": score_cards,
    "profile_visibility": profile_visibility,
  }


@router.get("/progress-insights", response_model=dict)
async def progress_insights(
  user_id: str | None = None,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  """Return motivational scan and style progress derived from persisted scores."""
  user_id = _require_actor_user_id(auth, user_id)
  rows_res = await db.execute(
    select(
      Outfit.scanned_at,
      Outfit.style_tags,
      OutfitScore.drip_score,
      OutfitScore.style_match,
    )
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id == user_id)
    .order_by(Outfit.scanned_at)
  )
  rows = rows_res.fetchall()
  scores = [float(row.drip_score) for row in rows if row.drip_score is not None]

  style_series: dict[str, list[float]] = {}
  for row in rows:
    tags = [str(tag).strip() for tag in (row.style_tags or []) if str(tag).strip()]
    if tags:
      if row.style_match is None:
        continue
      for tag in dict.fromkeys(tags):
        style_series.setdefault(tag, []).append(float(row.style_match))
    elif row.drip_score is not None:
      # With no target style, the fair comparison is the overall outfit score.
      style_series.setdefault("No style selected", []).append(float(row.drip_score))

  user_averages_res = await db.execute(
    select(Outfit.user_id, func.avg(OutfitScore.drip_score).label("average_score"))
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.user_id.is_not(None), OutfitScore.drip_score.is_not(None))
    .group_by(Outfit.user_id)
  )
  user_averages = [
    float(row.average_score)
    for row in user_averages_res.fetchall()
    if row.average_score is not None and str(row.user_id) != str(user_id)
  ]
  average_score = _average(scores)
  percentile = (
    round(100 * sum(value < average_score for value in user_averages) / len(user_averages))
    if user_averages
    else 0
  )

  return {
    "outfits_scanned": len(scores),
    "current_streak_days": _current_streak([row.scanned_at for row in rows]),
    "average_score": round(average_score, 1),
    "improvement_points": round(_improvement(scores), 1),
    "better_than_percent": max(0, min(99, percentile)),
    "style_progress": [
      {
        "style": style,
        "scans": len(values),
        "average_score": round(_average(values), 1),
        "improvement_points": round(_improvement(values), 1),
      }
      for style, values in style_series.items()
    ],
  }


@router.get("/style_dna", response_model=StyleDNAResponse)
async def style_dna(
  user_id: str | None = None,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  user_id = _require_actor_user_id(auth, user_id)
  # Try to load existing
  existing_stmt = select(StyleDNA).where(StyleDNA.user_id == user_id)
  res = await db.execute(existing_stmt)
  dna = res.scalar_one_or_none()

  # Quick aggregate heuristics
  score_stmt = (
    select(
      OutfitScore.drip_score,
      OutfitScore.color_match,
      OutfitScore.fit_quality,
      OutfitScore.body_compatibility,
      OutfitScore.trend_score,
      OutfitScore.style_match,
    )
    .join(Outfit, Outfit.id == OutfitScore.outfit_id)
    .where(Outfit.user_id == user_id)
    .order_by(desc(OutfitScore.created_at))
    .limit(20)
  )
  score_res = await db.execute(score_stmt)
  rows = score_res.fetchall()
  if rows:
    avg_drip = float(sum(r.drip_score or 0 for r in rows) / len(rows))
    avg_fit = float(sum(r.fit_quality or 0 for r in rows) / len(rows))
    avg_color = float(sum(r.color_match or 0 for r in rows) / len(rows))
    avg_trend = float(sum(r.trend_score or 0 for r in rows) / len(rows))
    tags = []
    if avg_fit >= 7: tags.append("fit-driven")
    if avg_color >= 7: tags.append("color-forward")
    if avg_trend >= 7: tags.append("on-trend")
    if avg_drip >= 8: tags.append("high-drip")
    label = "Refined street luxe" if "fit-driven" in tags else "Polished casual"
    description = f"Prefers tailored, body-aware looks with solid color coordination. Avg drip {avg_drip:.1f}."
  else:
    label = "Getting started"
    description = "Scan more outfits to build your Style DNA."
    tags = []

  if dna:
    dna.label = label
    dna.description = description
    dna.tags = tags
  else:
    dna = StyleDNA(user_id=user_id, label=label, description=description, tags=tags)
    db.add(dna)

  await db.commit()
  return StyleDNAResponse(user_id=user_id, label=dna.label or label, description=dna.description or description, tags=dna.tags or tags)
