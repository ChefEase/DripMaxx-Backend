
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.entities import User, UserProfile, Outfit, OutfitScore, DripScoreHistory, StyleDNA
from app.schemas.profile import ProfileSyncRequest, ProfileSyncResponse, StyleDNAResponse

router = APIRouter(prefix="/v1/profile", tags=["profile"])


def _visibility_flag(value: str | None) -> bool:
  """Map visibility string to boolean column (True=public, False=private/friends)."""
  if value is None:
    return True
  return value == "public"


def _visibility_mode(value: str | None) -> str:
  if value in ("public", "friends_only", "private"):
    return value
  return "public"


async def _get_or_create_user(
  db: AsyncSession,
  user_id: str,
  email: str | None = None,
  display_name: str | None = None,
  username: str | None = None,
):
  stmt = select(User).where(User.id == user_id)
  res = await db.execute(stmt)
  user = res.scalar_one_or_none()
  if user:
    return user
  user = User(id=user_id, email=email, display_name=display_name, username=username)
  db.add(user)
  await db.flush()
  return user


@router.post("/sync", response_model=ProfileSyncResponse)
async def sync_profile(payload: ProfileSyncRequest, db: AsyncSession = Depends(get_db)):
  user_id = payload.user_id

  if not user_id:
    raise HTTPException(status_code=400, detail="user_id is required")

  # Upsert user
  user = await _get_or_create_user(db, user_id, payload.email, payload.display_name, payload.username)
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
  return ProfileSyncResponse(user_id=user_id)


@router.get("/history", response_model=dict)
async def profile_history(user_id: str, db: AsyncSession = Depends(get_db)):
  """Return recent outfits and drip score history for a user."""
  if not user_id:
    raise HTTPException(status_code=400, detail="user_id is required")
  await _get_or_create_user(db, user_id)

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

  profile_stmt = select(UserProfile.profile_visibility, UserProfile.profile_visibility_mode).where(UserProfile.user_id == user_id)
  profile_res = await db.execute(profile_stmt)
  profile_row = profile_res.fetchone()
  if profile_row and profile_row.profile_visibility_mode in ("public", "friends_only", "private"):
    profile_visibility = profile_row.profile_visibility_mode
  else:
    profile_visibility = "public" if (profile_row and profile_row.profile_visibility) else "private"

  return {
    "recent_outfits": recent,
    "history": list(reversed(history)),
    "profile_visibility": profile_visibility,
  }


@router.get("/style_dna", response_model=StyleDNAResponse)
async def style_dna(user_id: str, db: AsyncSession = Depends(get_db)):
  if not user_id:
    raise HTTPException(status_code=400, detail="user_id is required")

  await _get_or_create_user(db, user_id)
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
