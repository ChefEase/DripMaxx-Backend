import json
import time
from typing import Any, Sequence

import replicate
import requests
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models import (
  Outfit,
  OutfitEvolutionRecommendation,
  OutfitEvolutionRevision,
  OutfitEvolutionSession,
)
from app.schemas.outfits import (
  EvolutionRecommendationResult,
  EvolutionRevisionResult,
  EvolutionSessionResponse,
  ScoreResponse,
  SuggestionCard,
)
from app.services.storage import upload_target_image

settings = get_settings()
VALID_STATUSES = {"completed", "partial", "remaining", "regressed"}


async def generate_target_image(session_id: str) -> None:
  """Generate once after the score response has returned; failures never affect scoring."""
  async with AsyncSessionLocal() as db:
    session = (await db.execute(select(OutfitEvolutionSession).where(
      OutfitEvolutionSession.id == session_id
    ))).scalar_one_or_none()
    if not session or session.target_image_url or session.target_generation_status == "complete":
      return
    original = (await db.execute(select(Outfit).where(Outfit.id == session.original_outfit_id))).scalar_one_or_none()
    if not original or not original.image_url:
      session.target_generation_status = "failed"
      session.target_generation_error = "Original image is unavailable."
      await db.commit()
      return
    session.target_generation_status = "generating"
    session.target_generation_error = None
    await db.commit()

    prompt = (
      "Edit the supplied original outfit photo into a photorealistic preview of this same person's outfit after "
      "applying only the supplied DripMaxx recommendations. Preserve identity, face, skin tone, hairstyle, body "
      "shape, pose, camera angle, environment, and every garment or accessory not explicitly changed. Preserve "
      "the original aesthetic. Do not add random accessories, substitute unrelated wardrobe items, beautify the "
      "person, or change their body. Maintain realistic fabric, fit, and clothing proportions. This is a visual "
      "reference, not a new fashion-model outfit. Target-look specification:\n"
      f"{json.dumps(session.target_look or {}, ensure_ascii=True)}"
    )

    def _generate() -> bytes:
      client = replicate.Client(api_token=settings.replicate_api_token, timeout=180)
      output = client.run(settings.replicate_image_model, input={
        "prompt": prompt,
        "input_images": [original.image_url],
        "aspect_ratio": "2:3",
        "quality": "medium",
        "number_of_images": 1,
        "output_format": "webp",
        "user_id": str(session.user_id),
      })
      item = output[0] if isinstance(output, (list, tuple)) else output
      if hasattr(item, "read"):
        return item.read()
      url = getattr(item, "url", None) or str(item)
      response = requests.get(url, timeout=60)
      response.raise_for_status()
      return response.content

    try:
      image_bytes = await run_in_threadpool(_generate)
      image_url = await run_in_threadpool(upload_target_image, image_bytes, str(session.id), str(session.user_id))
      if not image_url:
        raise RuntimeError("Generated image could not be persisted")
      session.target_image_url = image_url
      session.target_generation_status = "complete"
    except Exception as exc:
      logger.exception("Target-look generation failed session_id={}", session_id)
      session.target_generation_status = "failed"
      session.target_generation_error = "Target image is temporarily unavailable."
    await db.commit()


def calculate_potential_score(original_score: float, suggestions: Sequence[SuggestionCard]) -> float:
  """Estimate attainable potential from grounded, weighted improvement opportunities."""
  opportunity = sum(float(item.impact) for item in suggestions)
  return round(min(9.8, max(original_score, original_score + opportunity)), 1)


def calculate_revision_score(
  original_score: float,
  previous_score: float,
  potential_score: float,
  objective_score: float,
  recommendations: Sequence[OutfitEvolutionRecommendation],
  results: Sequence[dict[str, Any]],
  new_issue_severity: float,
  confidence: float,
) -> float:
  """Blend grounded upgrade evidence with an independent rescore.

  Recommendation evidence drives progress. The independent score acts as a
  credibility check without allowing photographic variation alone to erase a
  clearly verified improvement.
  """
  by_id = {str(item.get("id")): item for item in results}
  earned = 0.0
  regressed = 0.0
  for rec in recommendations:
    result = by_id.get(str(rec.id), {})
    rec_confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
    impact = float(rec.impact)
    state = result.get("status")
    if state == "completed":
      earned += impact * rec_confidence
    elif state == "partial":
      earned += impact * 0.45 * rec_confidence
    elif state == "regressed":
      regressed += impact * 0.55 * rec_confidence

  evidence_score = original_score + earned - regressed - max(0, new_issue_severity)
  confidence = max(0.0, min(1.0, confidence))
  objective_weight = 0.2 + 0.2 * confidence
  score = evidence_score * (1 - objective_weight) + objective_score * objective_weight

  has_verified_improvement = any(
    item.get("status") in {"completed", "partial"} and float(item.get("confidence", 0)) >= 0.65
    for item in results
  )
  has_regression = any(item.get("status") == "regressed" for item in results) or new_issue_severity > 0.15
  if has_verified_improvement and not has_regression:
    score = max(score, original_score + min(earned * 0.7, potential_score - original_score))
  if not has_verified_improvement and not has_regression:
    # An unchanged outfit should remain stable despite pose/lighting noise.
    score = min(max(score, previous_score - 0.2), previous_score + 0.2)
  return round(max(0.0, min(10.0, score)), 1)


def _json_from_output(raw: str) -> dict[str, Any]:
  text = raw.strip()
  if "{" in text and "}" in text:
    text = text[text.index("{") : text.rindex("}") + 1]
  try:
    data = json.loads(text.replace("\\_", "_"))
  except Exception as exc:
    raise ValueError("Revision comparison did not return valid JSON") from exc
  if not isinstance(data, dict) or not isinstance(data.get("recommendations"), list):
    raise ValueError("Revision comparison is missing recommendations")
  return data


async def compare_revision(
  original_image_url: str,
  current_image_url: str,
  original_score: float,
  potential_score: float,
  original_analysis: dict[str, Any],
  recommendations: Sequence[OutfitEvolutionRecommendation],
  revision_history: Sequence[OutfitEvolutionRevision],
) -> dict[str, Any]:
  if not settings.replicate_api_token:
    raise HTTPException(status_code=503, detail="AI revision comparison is temporarily unavailable.")
  rec_payload = [
    {
      "id": str(rec.id),
      "category": rec.category,
      "current_state": rec.current_state,
      "recommended_change": rec.recommended_change,
      "reason": rec.reason,
      "importance": rec.importance,
      "target_state": rec.target_state,
    }
    for rec in recommendations
  ]
  history_payload = [
    {
      "revision_number": row.revision_number,
      "score": float(row.current_score),
      "recommendation_results": row.recommendation_results,
    }
    for row in revision_history
  ]
  prompt = (
    "This is a semantic comparison of two photos in one Outfit Evolution session. "
    "Image 1 is ORIGINAL and image 2 is CURRENT. Do not evaluate the current photo as an unrelated outfit. "
    "Ignore differences caused only by pose, lighting, distance, camera angle, background, shadows, wrinkles, "
    "minor image quality, or partial occlusion. Compare clothing type, color, fit, silhouette, footwear, "
    "accessories, layering, proportions, and styling. Do not claim completion without visible evidence. "
    "A recommendation can be completed, partial, remaining, or regressed. If uncertain, use remaining or partial "
    "with lower confidence. Identify genuinely new styling issues, not photography issues. Output JSON only:\n"
    '{"recommendations":[{"id":"exact supplied id","status":"completed|partial|remaining|regressed",'
    '"confidence":0.0,"evidence":"specific visible evidence"}],'
    '"new_issues":[{"description":"issue","severity":0.0}],'
    '"summary":"encouraging factual comparison","confidence":0.0}\n'
    f"Original score: {original_score}; potential score: {potential_score}.\n"
    f"Original visual analysis: {json.dumps(original_analysis, ensure_ascii=True)}\n"
    f"Recommendations: {json.dumps(rec_payload, ensure_ascii=True)}\n"
    f"Previous revisions: {json.dumps(history_payload, ensure_ascii=True)}"
  )

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=75)
    tries = 0
    while True:
      tries += 1
      try:
        result = client.run(
          settings.replicate_vlm_model,
          input={
            "system_prompt": "You are a precise fashion revision comparator. Return only valid JSON.",
            "prompt": prompt,
            "image_input": [original_image_url, current_image_url],
            "reasoning_effort": "medium",
            "max_completion_tokens": 1200,
          },
        )
        if isinstance(result, (list, tuple)):
          return "".join(str(value) for value in result)
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
          return "".join(str(value) for value in result)
        return str(result)
      except replicate.exceptions.ReplicateError as exc:
        if exc.status == 429 and tries == 1:
          time.sleep(4)
          continue
        raise

  try:
    data = _json_from_output(await run_in_threadpool(_call))
  except Exception as exc:
    logger.exception("Outfit evolution comparison failed")
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI revision comparison is unavailable; please retry shortly.") from exc

  known_ids = {str(rec.id) for rec in recommendations}
  normalized = []
  seen = set()
  for item in data.get("recommendations", []):
    rec_id = str(item.get("id", ""))
    if rec_id not in known_ids or rec_id in seen:
      continue
    item_status = str(item.get("status", "remaining")).lower()
    normalized.append({
      "id": rec_id,
      "status": item_status if item_status in VALID_STATUSES else "remaining",
      "confidence": max(0.0, min(1.0, float(item.get("confidence", 0)))),
      "evidence": str(item.get("evidence") or "Not clearly verifiable from the current photo."),
    })
    seen.add(rec_id)
  for rec_id in known_ids - seen:
    normalized.append({"id": rec_id, "status": "remaining", "confidence": 0.0, "evidence": "Not clearly verifiable from the current photo."})
  data["recommendations"] = normalized
  data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0))))
  return data


async def load_evolution(db: AsyncSession, session_id: str, user_id: str) -> tuple[OutfitEvolutionSession, Outfit, list[OutfitEvolutionRecommendation], list[OutfitEvolutionRevision]]:
  session = (await db.execute(select(OutfitEvolutionSession).where(
    OutfitEvolutionSession.id == session_id,
    OutfitEvolutionSession.user_id == user_id,
  ))).scalar_one_or_none()
  if not session:
    raise HTTPException(status_code=404, detail="Outfit evolution session not found.")
  original = (await db.execute(select(Outfit).where(Outfit.id == session.original_outfit_id))).scalar_one()
  recommendations = list((await db.execute(select(OutfitEvolutionRecommendation).where(
    OutfitEvolutionRecommendation.session_id == session.id
  ).order_by(OutfitEvolutionRecommendation.position))).scalars().all())
  revisions = list((await db.execute(select(OutfitEvolutionRevision).where(
    OutfitEvolutionRevision.session_id == session.id
  ).order_by(OutfitEvolutionRevision.revision_number))).scalars().all())
  return session, original, recommendations, revisions


def serialize_evolution(
  session: OutfitEvolutionSession,
  original: Outfit,
  recommendations: Sequence[OutfitEvolutionRecommendation],
  revisions: Sequence[OutfitEvolutionRevision],
) -> EvolutionSessionResponse:
  revision_models = []
  for row in revisions:
    results = [EvolutionRecommendationResult(**item) for item in (row.recommendation_results or [])]
    revision_models.append(EvolutionRevisionResult(
      revision_number=row.revision_number,
      previous_score=float(row.previous_score),
      current_score=float(row.current_score),
      score_change=float(row.score_change),
      completed_count=sum(item.status == "completed" for item in results),
      total_recommendations=len(recommendations),
      recommendations=results,
      new_issues=list(row.new_issues or []),
      summary=row.summary,
      confidence=float(row.confidence),
    ))
  cards = [SuggestionCard(
    id=str(rec.id), title=rec.title, type=rec.category, description=rec.recommended_change,
    current_state=rec.current_state, recommended_change=rec.recommended_change,
    reason=rec.reason, importance=rec.importance, target_state=rec.target_state, impact=float(rec.impact),
  ) for rec in recommendations]
  return EvolutionSessionResponse(
    session_id=str(session.id), original_outfit_id=str(session.original_outfit_id),
    original_image_url=original.image_url, original_score=float(session.original_score),
    current_score=float(session.current_score), potential_score=float(session.potential_score),
    target_image_url=session.target_image_url,
    target_generation_status=session.target_generation_status or "pending",
    target_generation_error=session.target_generation_error,
    recommendations=cards,
    revisions=revision_models, latest_revision=revision_models[-1] if revision_models else None,
  )
