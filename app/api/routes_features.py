from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.db.session import get_db
from app.models import FeatureSubmission, Outfit, OutfitScore, User
from app.schemas.features import (
  FeatureSubmissionListResponse,
  FeatureSubmissionResponse,
  SubmitFeatureRequest,
  SubmitFeatureResponse,
)

router = APIRouter(prefix="/v1/features", tags=["features"])
DEVELOPER_EMAIL = "onyiakamsy74@gmail.com"
MINIMUM_FEATURE_SCORE = 7.5


async def _require_developer(auth: AuthContext, db: AsyncSession) -> None:
  if auth.email != DEVELOPER_EMAIL:
    raise HTTPException(status_code=403, detail="Developer access required")


@router.post("/submissions", response_model=SubmitFeatureResponse)
async def submit_outfit_for_feature(
  payload: SubmitFeatureRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  if not payload.display_consent:
    raise HTTPException(status_code=400, detail="Permission to feature the outfit is required")

  result = await db.execute(
    select(Outfit, OutfitScore)
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.id == payload.outfit_id, Outfit.user_id == auth.app_user_id)
  )
  row = result.first()
  if not row:
    raise HTTPException(status_code=404, detail="Outfit not found for authenticated user")
  if float(row[1].drip_score or 0) < MINIMUM_FEATURE_SCORE:
    raise HTTPException(status_code=400, detail="Only outfits scoring 7.5 or higher can be submitted")

  submission = FeatureSubmission(
    outfit_id=row[0].id,
    user_id=auth.app_user_id,
    feature_username=payload.feature_username,
    instagram_url=payload.instagram_url,
    tiktok_url=payload.tiktok_url,
    display_consent=True,
  )
  db.add(submission)
  try:
    await db.commit()
  except IntegrityError as exc:
    await db.rollback()
    raise HTTPException(status_code=409, detail="This outfit has already been submitted") from exc
  return SubmitFeatureResponse(submission_id=submission.id)


@router.get("/submissions", response_model=FeatureSubmissionListResponse)
async def list_feature_submissions(
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  await _require_developer(auth, db)
  result = await db.execute(
    select(FeatureSubmission, Outfit.image_url, OutfitScore.drip_score, User.email, User.username)
    .join(Outfit, Outfit.id == FeatureSubmission.outfit_id)
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .join(User, User.id == FeatureSubmission.user_id)
    .order_by(FeatureSubmission.created_at.desc())
  )
  return FeatureSubmissionListResponse(
    submissions=[
      FeatureSubmissionResponse(
        id=row[0].id,
        outfit_id=row[0].outfit_id,
        user_id=row[0].user_id,
        image_url=row[1],
        drip_score=float(row[2]),
        account_email=row[3],
        account_username=row[4],
        feature_username=row[0].feature_username,
        instagram_url=row[0].instagram_url,
        tiktok_url=row[0].tiktok_url,
        display_consent=bool(row[0].display_consent),
        status=row[0].status,
        consented_at=row[0].consented_at,
        created_at=row[0].created_at,
      )
      for row in result.fetchall()
    ]
  )
