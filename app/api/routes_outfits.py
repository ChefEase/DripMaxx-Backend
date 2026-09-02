import json
import logging

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, Depends, HTTPException, status, Form
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, require_auth
from app.schemas.outfits import ScoreResponse, UserContext
from app.db.session import get_db
from app.services.ai_scoring import score_with_ai
from app.services.storage import upload_outfit_image
from app.models import (
  Outfit, OutfitScore, OutfitSuggestion, SuggestionTypeEnum, DripScoreHistory,
  OutfitEvolutionSession, OutfitEvolutionRecommendation, OutfitEvolutionRevision,
)
from app.services.outfit_evolution import (
  calculate_potential_score, calculate_revision_score, compare_revision,
  generate_target_image, load_evolution, serialize_evolution,
)
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
  background_tasks: BackgroundTasks,
  image: UploadFile = File(...),
  user_context: str = Form(..., description="JSON of user context"),
  evolution_session_id: str | None = Form(default=None),
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

  evolution_context = None
  if evolution_session_id:
    evolution_context = await load_evolution(db, evolution_session_id, user_id)

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

  logger.info("score_outfit stage=ai_start outfit_id=%s user_id=%s", outfit.id, user_id)
  try:
    score = await score_with_ai(
      image_bytes, user_ctx, outfit.image_url,
      generate_improvements=evolution_context is None,
    )
  except HTTPException as exc:
    logger.warning(
      "score_outfit stage=ai_rejected outfit_id=%s status=%s detail=%s",
      outfit.id,
      exc.status_code,
      exc.detail,
    )
    raise
  except Exception:
    logger.exception("score_outfit stage=ai_failed outfit_id=%s", outfit.id)
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail="AI scoring failed before results were generated.",
    )
  logger.info("score_outfit stage=ai_done outfit_id=%s drip_score=%s", outfit.id, score.drip_score)

  if evolution_context:
    session, original_outfit, evolution_recommendations, prior_revisions = evolution_context
    comparison = await compare_revision(
      original_outfit.image_url, outfit.image_url, float(session.original_score),
      float(session.potential_score), session.original_analysis or {},
      evolution_recommendations, prior_revisions,
    )
    new_issues_raw = comparison.get("new_issues") or []
    new_issues = [
      str(item.get("description") or "").strip() if isinstance(item, dict) else str(item).strip()
      for item in new_issues_raw
      if (str(item.get("description") or "").strip() if isinstance(item, dict) else str(item).strip())
    ]
    issue_severity = sum(
      max(0.0, min(0.6, float(item.get("severity", 0))))
      for item in new_issues_raw if isinstance(item, dict)
    )
    deviation = comparison.get("overall_deviation") or {}
    if isinstance(deviation, dict):
      issue_severity += max(0.0, min(1.0, float(deviation.get("severity", 0))))
      deviation_evidence = str(deviation.get("evidence") or "").strip()
      if deviation_evidence and deviation.get("level") in {"moderate", "major"}:
        new_issues.append(deviation_evidence)
    issue_severity = min(1.5, issue_severity)
    previous_score = float(session.current_score)
    revised_score = calculate_revision_score(
      float(session.original_score), previous_score, float(session.potential_score),
      float(score.drip_score), evolution_recommendations, comparison["recommendations"],
      issue_severity, comparison["confidence"],
    )
    revision = OutfitEvolutionRevision(
      session_id=session.id, outfit_id=outfit.id,
      revision_number=len(prior_revisions) + 1, previous_score=previous_score,
      current_score=revised_score, score_change=round(revised_score - previous_score, 1),
      completed_recommendation_ids=[
        item["id"] for item in comparison["recommendations"] if item["status"] == "completed"
      ],
      recommendation_results=comparison["recommendations"], new_issues=new_issues,
      summary=str(comparison.get("summary") or "Your changes were compared with the original outfit."),
      confidence=comparison["confidence"],
    )
    db.add(revision)
    session.current_score = revised_score
    if revised_score >= float(session.potential_score) - 0.1 and all(
      item["status"] == "completed" for item in comparison["recommendations"]
    ):
      session.status = "evolved"
    prior_revisions = [*prior_revisions, revision]
    score = score.model_copy(update={
      "drip_score": revised_score, "overall_score": revised_score,
      "quality_tier": "Elite" if revised_score >= 8.5 else "Strong" if revised_score >= 7 else "Solid" if revised_score >= 5.5 else "Needs work",
    })

  db.add(
    OutfitScore(
      outfit_id=outfit.id,
      color_match=score.breakdown.color_match,
      fit_quality=score.breakdown.fit_quality,
      body_compatibility=(
        None
        if "body_compatibility" in score.unavailable_metrics
        else score.breakdown.body_compatibility
      ),
      trend_score=score.breakdown.trend_score,
      style_match=(
        None
        if "style_match" in score.unavailable_metrics
        else score.breakdown.style_match
      ),
      drip_score=score.drip_score,
      model_version="clip+llama",
      raw_features=score.visual_analysis or None,
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

  if not evolution_context:
    session = OutfitEvolutionSession(
      user_id=user_id, original_outfit_id=outfit.id,
      original_score=score.drip_score, current_score=score.drip_score,
      potential_score=calculate_potential_score(float(score.drip_score), score.suggestions),
      original_analysis=score.visual_analysis or {},
      target_look=score.target_look or {},
      target_generation_status="queued",
    )
    db.add(session)
    await db.flush()
    evolution_recommendations = []
    for idx, suggestion in enumerate(score.suggestions[:5], start=1):
      recommendation = OutfitEvolutionRecommendation(
        session_id=session.id, position=idx, category=suggestion.type, title=suggestion.title,
        current_state=suggestion.current_state,
        recommended_change=suggestion.recommended_change or suggestion.description,
        reason=suggestion.reason, importance=suggestion.importance,
        target_state=suggestion.target_state, impact=suggestion.impact,
      )
      db.add(recommendation)
      evolution_recommendations.append(recommendation)
    await db.flush()
    original_outfit = outfit
    prior_revisions = []
    should_generate_target = True
  else:
    should_generate_target = False

  await consume_scan_credit_if_needed(db, user_id, quota)
  xp_awarded = SCAN_XP if await award_xp_once(db, user_id, SCAN_XP, "outfit_scan", outfit.id, "Every outfit scan") else 0

  await db.commit()
  await db.refresh(outfit)
  if should_generate_target:
    background_tasks.add_task(generate_target_image, str(session.id))

  return score.model_copy(update={
    "outfit_id": outfit.id,
    "xp_awarded": xp_awarded,
    "evolution": serialize_evolution(session, original_outfit, evolution_recommendations, prior_revisions),
  })


@router.get("/evolution/{session_id}")
async def get_outfit_evolution(
  session_id: str,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  return serialize_evolution(*(await load_evolution(db, session_id, auth.app_user_id)))


@router.get("/evolutions")
async def list_outfit_evolutions(
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  session_ids = list((await db.execute(
    select(OutfitEvolutionSession.id)
    .where(OutfitEvolutionSession.user_id == auth.app_user_id)
    .order_by(desc(OutfitEvolutionSession.updated_at))
    .limit(12)
  )).scalars().all())
  sessions = [
    serialize_evolution(*(await load_evolution(db, str(session_id), auth.app_user_id)))
    for session_id in session_ids
  ]
  return {"sessions": sessions}


@router.post("/evolution/{session_id}/target")
async def retry_target_generation(
  session_id: str,
  background_tasks: BackgroundTasks,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  session, original, recommendations, revisions = await load_evolution(db, session_id, auth.app_user_id)
  if not session.target_image_url and session.target_generation_status not in {"queued", "generating"}:
    session.target_generation_status = "queued"
    session.target_generation_error = None
    await db.commit()
    background_tasks.add_task(generate_target_image, str(session.id))
  return serialize_evolution(session, original, recommendations, revisions)
