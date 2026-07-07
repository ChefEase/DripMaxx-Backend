import json
import logging

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.schemas.outfits import ScoreResponse, UserContext
from app.db.session import get_db
from app.services.ai_scoring import score_with_ai
from app.services.storage import upload_outfit_image
from app.models import Outfit, OutfitScore, OutfitSuggestion, SuggestionTypeEnum, DripScoreHistory
from app.services.usage_limits import get_scan_quota
from app.services.rewards import SCAN_XP, award_xp_once, consume_scan_credit_if_needed

router = APIRouter(prefix="/v1/outfits", tags=["outfits"])
logger = logging.getLogger(__name__)


@router.post(
  "/score",
  response_model=ScoreResponse,
  summary="Score an outfit image (stubbed)",
)
async def score_outfit(
  image: UploadFile = File(...),
  user_context: str = Form(..., description="JSON of user context"),
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  if not image.content_type or not image.content_type.startswith("image/"):
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Upload must be an image (jpg/png).",
    )

  try:
    ctx_raw = json.loads(user_context)
    # Allow either {"user_context": {...}} or direct {...}
    if "user_context" in ctx_raw and isinstance(ctx_raw["user_context"], dict):
      ctx_raw = ctx_raw["user_context"]
    user_ctx = UserContext.model_validate(ctx_raw)
  except Exception as exc:
    raise HTTPException(
      status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
      detail=f"Invalid user_context JSON: {exc}",
    ) from exc

  if user_ctx.user_id and user_ctx.user_id != auth.app_user_id:
    raise HTTPException(status_code=403, detail="user_id does not match authenticated user")
  user_id = auth.app_user_id

  quota = await get_scan_quota(db, user_id)
  if not quota["allowed"]:
    raise HTTPException(
      status_code=402,
      detail={
        "code": "scan_limit_reached",
        "message": f"Scan limit reached for your {quota['plan']} plan.",
        "plan": quota["plan"],
        "limit_type": quota["limit_type"],
        "limit": quota["limit"],
        "used": quota["used"],
      },
    )

  image_bytes = await image.read()
  if not image_bytes:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image upload is empty.")

  # Persist outfit first to get outfit_id, then upload image
  style_tags = list(user_ctx.style_preferences) if user_ctx.style_preferences else []
  outfit = Outfit(
    user_id=user_id,
    style_tags=style_tags,
    source="upload",
    image_url="uploaded://not-stored",
    notes=None,
    is_example=False,
  )
  db.add(outfit)
  await db.flush()

  # Upload image to Supabase Storage (before AI so AI can use URL)
  content_type = image.content_type or "image/jpeg"
  image_url = upload_outfit_image(image_bytes, outfit.id, user_id, content_type)
  if image_url:
    outfit.image_url = image_url
  else:
    logger.warning("outfit image upload failed; keeping placeholder URL for outfit_id=%s", outfit.id)
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Image upload failed; check Supabase storage configuration.",
    )

  score = await score_with_ai(image_bytes, user_ctx, outfit.image_url)

  db.add(
    OutfitScore(
      outfit_id=outfit.id,
      color_match=score.breakdown.color_match,
      fit_quality=score.breakdown.fit_quality,
      body_compatibility=score.breakdown.body_compatibility,
      trend_score=score.breakdown.trend_score,
      style_match=score.breakdown.style_match,
      drip_score=score.drip_score,
      model_version="clip+llama",
    )
  )

  db.add(
    DripScoreHistory(
      user_id=user_id,
      outfit_id=outfit.id,
      drip_score=score.drip_score,
    )
  )

  for idx, suggestion in enumerate(score.suggestions, start=1):
    try:
      sug_type = SuggestionTypeEnum(suggestion.type.lower())
    except Exception:
      sug_type = SuggestionTypeEnum.other
    db.add(
      OutfitSuggestion(
        outfit_id=outfit.id,
        type=sug_type,
        title=suggestion.title,
        description=suggestion.description,
        rank=idx,
      )
    )

  await consume_scan_credit_if_needed(db, user_id, quota)
  xp_awarded = SCAN_XP if await award_xp_once(db, user_id, SCAN_XP, "outfit_scan", outfit.id, "Every outfit scan") else 0

  await db.commit()
  await db.refresh(outfit)

  return score.model_copy(update={"outfit_id": outfit.id, "xp_awarded": xp_awarded})
