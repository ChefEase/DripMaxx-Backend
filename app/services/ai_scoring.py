import io
import random
import statistics
import json
from typing import List, Sequence, Tuple

import replicate
import numpy as np
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image, ImageStat
import requests
try:
  from mediapipe import solutions as mp_solutions
except Exception:  # pragma: no cover
  mp_solutions = None

from app.core.config import get_settings
from app.schemas.outfits import ScoreBreakdown, ScoreResponse, SuggestionCard, UserContext
from app.services.ai_suggestions import generate_suggestions

settings = get_settings()
DEFAULT_MODEL_REF = (
  "krthr/clip-embeddings:1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4"
)
SAM_MODEL_REF = "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
VLM_MODEL_REF_DEFAULT = "chefease/fashionvlmodel:7f22da0c6e72f395e33b12c0467c9d9d158cbed71ca391e84c8704ef656c2609"
STYLE_PROMPTS = [
  "streetwear outfit photo",
  "minimalist outfit photo",
  "luxury fashion outfit",
  "vintage outfit photo",
  "modern trendy outfit",
  "outdated outfit",
]


def _clamp(score: float) -> float:
  return round(max(0.0, min(10.0, score)), 1)


def _normalize_score(raw_score: float) -> float:
  # Fast fix: stretch distribution to reduce inflated mid/high scores.
  adjusted = (raw_score - 7.0) * 1.8 + 5.0
  return _clamp(adjusted)


def _apply_noise(score: float) -> float:
  return _clamp(score + random.uniform(-0.4, 0.4))


def _label_to_score(label: str) -> float:
  mapping = {
    "bad": 2.0,
    "poor": 3.5,
    "average": 5.8,
    "good": 7.8,
    "excellent": 9.2,
  }
  return mapping.get(label.strip().lower(), 5.0)


def _apply_penalties(scores: dict, penalties: dict) -> dict:
  # penalties is expected to be booleans, e.g. {"excessive_monochrome": true, ...}
  trend_penalty = 0.0
  color_penalty = 0.0
  cohesion_penalty = 0.0
  if penalties.get("excessive_monochrome") and penalties.get("neon_colors"):
    trend_penalty -= 2.0
  if penalties.get("clashing_patterns"):
    color_penalty -= 1.5
    cohesion_penalty -= 1.0
  if penalties.get("costume_like"):
    trend_penalty -= 1.5
    cohesion_penalty -= 1.5
  if penalties.get("poor_layering"):
    cohesion_penalty -= 1.0
  if penalties.get("too_many_colors"):
    color_penalty -= 1.0
  scores["trend_score"] = _clamp(scores["trend_score"] + trend_penalty)
  scores["color_match"] = _clamp(scores["color_match"] + color_penalty)
  scores["style_match"] = _clamp(scores["style_match"] + cohesion_penalty)
  return scores


def _quality_tier(overall: float) -> str:
  if overall >= 9.0:
    return "Top_Notch"
  if overall >= 7.0:
    return "Good"
  if overall >= 5.0:
    return "Mid"
  return "Bad"


def _compute_color_metrics(image_bytes: bytes) -> dict:
  try:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  except Exception as exc:  # pragma: no cover - defensive guard for bad uploads
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Could not read image for scoring.",
    ) from exc

  stat = ImageStat.Stat(image)
  brightness = sum(stat.mean) / (3 * 255)
  contrast = sum(stat.stddev) / (3 * 128)
  return {"brightness": brightness, "contrast": contrast}


def _replicate_image_input(image_bytes: bytes, image_url: str | None):
  if image_url:
    return image_url
  file_obj = io.BytesIO(image_bytes)
  file_obj.name = "upload.jpg"
  return file_obj


async def _remote_mask_coverage(image_bytes: bytes, image_url: str | None = None) -> float | None:
  """Fallback using Replicate SAM-2; returns mask coverage ratio."""
  if not settings.replicate_api_token:
    return None

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token)
    image_input = _replicate_image_input(image_bytes, image_url)
    result = client.run(
      SAM_MODEL_REF,
      input={
        "image": image_input,
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "use_m2m": True,
      },
    )
    url = None
    if isinstance(result, dict) and "combined_mask" in result:
      url = result["combined_mask"]
    elif isinstance(result, str):
      url = result
    if not url:
      return None
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    mask_img = Image.open(io.BytesIO(resp.content)).convert("L")
    arr = np.array(mask_img)
    return float(np.mean(arr > 64))

  try:
    return await run_in_threadpool(_call)
  except Exception as exc:
    logger.warning(f"SAM-2 mask fetch failed: {exc}")
    return None


async def _remote_mask(image_bytes: bytes, image_url: str | None = None) -> np.ndarray | None:
  """Get binary mask from SAM-2."""
  if not settings.replicate_api_token:
    return None

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token)
    image_input = _replicate_image_input(image_bytes, image_url)
    result = client.run(
      SAM_MODEL_REF,
      input={
        "image": image_input,
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "use_m2m": True,
      },
    )
    # Prefer combined mask
    urls = []
    if isinstance(result, dict):
      if "combined_mask" in result and result["combined_mask"]:
        urls.append(result["combined_mask"])
      elif "masks" in result and isinstance(result["masks"], list):
        urls.extend([m for m in result["masks"] if m])
    elif isinstance(result, str):
      urls.append(result)

    if not urls:
      return None

    merged = None
    for url in urls:
      try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        mask_img = Image.open(io.BytesIO(resp.content)).convert("L")
        arr = (np.array(mask_img) > 64).astype(np.uint8)
        if merged is None:
          merged = arr
        else:
          if merged.shape != arr.shape:
            continue
          merged = np.maximum(merged, arr)
      except Exception as exc:
        logger.warning(f"mask fetch/merge failed for {url}: {exc}")
        continue

    return merged

  try:
    return await run_in_threadpool(_call)
  except Exception as exc:
    logger.warning(f"SAM-2 mask fetch failed: {exc}")
    return None


def _component_stats(mask: np.ndarray, min_ratio: float = 0.02):
  h, w = mask.shape
  visited = np.zeros_like(mask, dtype=bool)
  areas = []
  bboxes = []

  def neighbors(r, c):
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
      nr, nc = r+dr, c+dc
      if 0 <= nr < h and 0 <= nc < w:
        yield nr, nc

  for r in range(h):
    for c in range(w):
      if mask[r, c] == 0 or visited[r, c]:
        continue
      # flood fill
      stack = [(r, c)]
      visited[r, c] = True
      coords = []
      while stack:
        cr, cc = stack.pop()
        coords.append((cr, cc))
        for nr, nc in neighbors(cr, cc):
          if mask[nr, nc] and not visited[nr, nc]:
            visited[nr, nc] = True
            stack.append((nr, nc))
      area = len(coords)
      if area / (h * w) >= min_ratio:
        rs = [p[0] for p in coords]; cs = [p[1] for p in coords]
        bboxes.append((min(rs), min(cs), max(rs), max(cs)))
        areas.append(area)

  # sort descending by area to make largest first
  sorted_idx = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
  areas = [areas[i] for i in sorted_idx]
  bboxes = [bboxes[i] for i in sorted_idx]

  return areas, bboxes, h, w


def _pose_metrics(image: Image.Image):
  if mp_solutions is None:
    raise RuntimeError("mediapipe solutions module unavailable")

  np_img = np.array(image)
  pose = mp_solutions.pose.Pose(static_image_mode=True)
  res = pose.process(np_img)
  if not res.pose_landmarks:
    return 0, 0.0, False

  landmarks = res.pose_landmarks.landmark
  required = [
    mp_solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp_solutions.pose.PoseLandmark.LEFT_HIP,
    mp_solutions.pose.PoseLandmark.RIGHT_HIP,
    mp_solutions.pose.PoseLandmark.LEFT_KNEE,
    mp_solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp_solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp_solutions.pose.PoseLandmark.RIGHT_ANKLE,
  ]
  visibilities = [landmarks[i].visibility for i in range(len(landmarks))]
  avg_vis = float(sum(visibilities) / len(visibilities))

  def visible(idx):
    return landmarks[idx].visibility >= 0.5

  knees_ankles_ok = all(visible(i) for i in required[-4:])
  all_required = all(visible(i) for i in required)

  return len(landmarks), avg_vis, (knees_ankles_ok and all_required)


def _derive_breakdown(
  embedding: Sequence[float],
  user_ctx: UserContext,
  color_metrics: dict,
  style_sim: float,
) -> ScoreBreakdown:
  emb_mean = statistics.fmean(embedding)
  emb_std = statistics.pstdev(embedding)
  emb_abs = statistics.fmean(abs(v) for v in embedding)

  color_match = _clamp(4.5 + color_metrics["contrast"] * 3 + color_metrics["brightness"] * 3)
  fit_quality = _clamp(5.0 + emb_std * 2.2)
  body_bonus = 0.5 if user_ctx.user_body_type else 0.0
  body_compatibility = _clamp(4.2 + (color_metrics["brightness"] - 0.5) * 3 + body_bonus)
  trend_score = _clamp(4.0 + emb_mean * 5 + len(user_ctx.style_inspirations) * 0.3)
  style_boost = min(1.5, 0.3 * len(user_ctx.style_preferences))
  style_match = _clamp(4.5 + emb_abs * 2 + style_boost + style_sim * 4)

  return ScoreBreakdown(
    color_match=color_match,
    fit_quality=fit_quality,
    body_compatibility=body_compatibility,
    trend_score=trend_score,
    style_match=style_match,
  )


async def _run_replicate(image_bytes: bytes, image_url: str | None = None) -> Sequence[float]:
  if not settings.replicate_api_token:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Replicate API token is missing; set REPLICATE_API_TOKEN.",
    )

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=30)
    model_ref = settings.replicate_model or DEFAULT_MODEL_REF
    if ":" not in model_ref:
      model_ref = f"{model_ref}:{DEFAULT_MODEL_REF.split(':', 1)[1]}"
    image_input = _replicate_image_input(image_bytes, image_url)
    tries = 0
    while True:
      tries += 1
      try:
        result = client.run(model_ref, input={"image": image_input})
        break
      except replicate.exceptions.ReplicateError as exc:
        if exc.status == 429 and tries == 1:
          import time
          time.sleep(3)
          continue
        raise
    # Model returns a dict with "embedding" key
    return result["embedding"] if isinstance(result, dict) and "embedding" in result else result

  return await run_in_threadpool(_call)


async def _text_embeddings(prompt: str) -> Sequence[float]:
  if not settings.replicate_api_token:
    raise HTTPException(
      status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
      detail="Replicate API token is missing; set REPLICATE_API_TOKEN.",
    )

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=30)
    model_ref = settings.replicate_model or DEFAULT_MODEL_REF
    if ":" not in model_ref:
      model_ref = f"{model_ref}:{DEFAULT_MODEL_REF.split(':', 1)[1]}"
    tries = 0
    while True:
      tries += 1
      try:
        res = client.run(model_ref, input={"text": prompt})
        break
      except replicate.exceptions.ReplicateError as exc:
        if exc.status == 429 and tries == 1:
          import time
          time.sleep(3)
          continue
        raise
    if isinstance(res, dict) and "embedding" in res:
      return res["embedding"]
    return res

  return await run_in_threadpool(_call)


async def _vlm_breakdown(
  image_bytes: bytes, user_ctx: UserContext, image_url: str | None = None
) -> tuple[ScoreBreakdown, str, dict] | None:
  """Ask the VLM for qualitative labels + penalties; convert to scores."""
  if not settings.replicate_api_token:
    return None
  model_ref = settings.replicate_vlm_model or VLM_MODEL_REF_DEFAULT

  sys_prompt = (
    "You are a fashion rater. Look at the image and output ONLY JSON with qualitative labels and penalties:\n"
    "{"
    "\"color_balance\": \"bad|poor|average|good|excellent\","
    "\"silhouette_fit\": \"bad|poor|average|good|excellent\","
    "\"outfit_cohesion\": \"bad|poor|average|good|excellent\","
    "\"trend_alignment\": \"bad|poor|average|good|excellent\","
    "\"originality\": \"bad|poor|average|good|excellent\","
    "\"penalties\": {"
    "\"excessive_monochrome\": true|false,"
    "\"neon_colors\": true|false,"
    "\"clashing_patterns\": true|false,"
    "\"costume_like\": true|false,"
    "\"poor_layering\": true|false,"
    "\"too_many_colors\": true|false,"
    "\"simple_clean\": true|false"
    "}"
    "}.\n\n"
    "Rules:\n"
    "- Base ratings on how well the outfit fits the user's personal style and body, not just general trends.\n"
    "- Give higher weight to fit and cohesion (athletic build focus).\n"
    "- Color_balance reflects harmony, contrast, and luxury feel.\n"
    "- Trend_alignment counts, but do not harshly penalize stylish non-hype looks.\n"
    "- Originality reflects uniqueness without being costume-like.\n"
    "Output only JSON. No explanations, no extra text, no markdown."
  )
  user_prompt = (
    f"User style prefs: {', '.join(user_ctx.style_preferences) or 'unspecified'}; "
    f"inspirations: {', '.join(user_ctx.style_inspirations) or 'unspecified'}; "
    f"height: {user_ctx.user_height or 'n/a'}; body_type: {user_ctx.user_body_type or 'n/a'}; "
    f"gender_style: {user_ctx.gender_style_preference or 'n/a'}."
  )

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=60)
    image_input = _replicate_image_input(image_bytes, image_url)
    tries = 0
    while True:
      tries += 1
      try:
        res = client.run(
          model_ref,
          input={
            "image": image_input,
            "prompt": sys_prompt + "\n\n" + user_prompt,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 400,
          },
        )
        break
      except replicate.exceptions.ReplicateError as exc:
        if exc.status == 429 and tries == 1:
          import time; time.sleep(4); continue
        raise
    if isinstance(res, (list, tuple)):
      return "".join(str(x) for x in res)
    if hasattr(res, "__iter__") and not isinstance(res, (str, bytes)):
      return "".join(str(x) for x in res)
    return str(res)

  try:
    raw = await run_in_threadpool(_call)
    if "{" in raw and "}" in raw:
      raw = raw[raw.find("{") : raw.rfind("}") + 1]
    raw = raw.replace("\\_", "_").strip()
    data = json.loads(raw)
    penalties = data.get("penalties") or {}
    # Map new structure to existing fields
    scores = {
      "color_match": _label_to_score(data.get("color_balance", "average")),
      "fit_quality": _label_to_score(data.get("silhouette_fit", "average")),
      "body_compatibility": _label_to_score(data.get("silhouette_fit", "average")),
      "trend_score": _label_to_score(data.get("trend_alignment", "average")),
      "style_match": _label_to_score(data.get("outfit_cohesion", "average")),
    }
    scores = _apply_penalties(scores, penalties)
    # Add slight noise without normalization for label-based scores
    scores = {k: _apply_noise(v) for k, v in scores.items()}
    return ScoreBreakdown(
      color_match=scores["color_match"],
      fit_quality=scores["fit_quality"],
      body_compatibility=scores["body_compatibility"],
      trend_score=scores["trend_score"],
      style_match=scores["style_match"],
    ), "labels", penalties
  except Exception as exc:
    logger.error(f"VLM scoring failed; raw='{raw[:400] if 'raw' in locals() else ''}' err={exc}")
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail="VLM scoring unavailable; please retry shortly.",
    ) from exc


async def score_with_ai(
  image_bytes: bytes, user_ctx: UserContext, image_url: str | None = None
) -> ScoreResponse:
  """Generate clip-like embeddings via Replicate, derive Drip Score, and emit UX suggestions."""
  image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  width, height = image.size
  if width < 512 or height < 768:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Image resolution too low. Upload a clearer full-body photo.",
    )

  # Segmentation and person validation
  mask = await _remote_mask(image_bytes, image_url)
  if mask is None:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Upload a photo with exactly one person.",
    )
  areas, bboxes, h, w = _component_stats(mask)
  people_detected = len(areas)
  if people_detected == 0:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Upload a photo with exactly one person.",
    )
  if people_detected >= 2:
    second_ratio = areas[1] / (h * w)
    if second_ratio >= 0.08:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Upload a photo with exactly one person.",
      )

  mask_cov = float(np.sum(mask)) / (mask.shape[0] * mask.shape[1])
  # choose the largest component that is not swallowing the whole frame
  chosen_idx = 0
  for idx, area in enumerate(areas):
    ratio = area / (h * w)
    if ratio <= 0.90:  # ignore masks that cover almost the whole image (busy background)
      chosen_idx = idx
      break

  # bounding box area ratio using chosen component
  y1, x1, y2, x2 = bboxes[chosen_idx]
  bbox_area = (y2 - y1 + 1) * (x2 - x1 + 1)
  person_area_ratio = bbox_area / (mask.shape[0] * mask.shape[1])
  if person_area_ratio < 0.18:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Person is too far away",
    )
  # Reject only when the detected person box dominates the frame
  if person_area_ratio > 1.0:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Person is too close to the camera",
    )

  # Pose checks
  pose_skipped = False
  try:
    keypoints, avg_vis, full_body_ok = _pose_metrics(image)
  except RuntimeError as exc:
    pose_skipped = True
    keypoints, avg_vis, full_body_ok = 0, 0.0, True  # allow flow to continue
    warnings = ["Pose check skipped; pose model unavailable on server."]

  if not pose_skipped:
    if not full_body_ok:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Full body must be visible (head to feet).",
      )
    if avg_vis < 0.5:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Image is too unclear for accurate outfit scoring.",
      )

  # Try VLM for grounded numeric scores first (if configured)
  breakdown = None
  breakdown_mode = "numeric"
  breakdown_flags: dict = {}
  if settings.replicate_vlm_model:
    res = await _vlm_breakdown(image_bytes, user_ctx, image_url)
    if res:
      breakdown, breakdown_mode, breakdown_flags = res
      logger.info("score pipeline: using VLM breakdown (model=%s)", settings.replicate_vlm_model)
  top_sim = 0.0
  top_prompt = None

  if breakdown is None:
    logger.info("score pipeline: VLM unavailable or failed; falling back to CLIP embeddings")
    try:
      embedding = await _run_replicate(image_bytes, image_url)
    except HTTPException:
      logger.warning("Replicate token missing; using fake score fallback.")
      return fake_score(user_ctx.style_preferences)
    except Exception as exc:
      logger.exception(f"Replicate call failed: {exc}")
      raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="AI scoring service is unavailable right now.",
      )

    if not embedding:
      raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="AI service returned an empty embedding.",
      )

    # CLIP text alignment
    style_prompt = user_ctx.style_preferences[0] + " outfit photo" if user_ctx.style_preferences else "outfit photo"
    try:
      tvec = await _text_embeddings(style_prompt)
      img_vec = np.array(embedding, dtype=float)
      tvec_np = np.array(tvec, dtype=float)
      top_sim = float(np.dot(img_vec, tvec_np) / (np.linalg.norm(img_vec) * np.linalg.norm(tvec_np) + 1e-8))
      top_prompt = style_prompt
    except Exception as exc:
      logger.warning(f"text embedding failed: {exc}")

    color_metrics = _compute_color_metrics(image_bytes)
    breakdown = _derive_breakdown(embedding, user_ctx, color_metrics, top_sim)
  else:
    color_metrics = _compute_color_metrics(image_bytes)
  overall_score = _clamp(
    0.30 * breakdown.color_match
    + 0.20 * breakdown.fit_quality
    + 0.20 * breakdown.body_compatibility
    + 0.10 * breakdown.trend_score
    + 0.20 * breakdown.style_match
  )
  if breakdown_mode == "numeric":
    overall_score = _apply_noise(_normalize_score(overall_score))
  else:
    overall_score = _apply_noise(overall_score)

  # Costume or absurd outfits get capped low
  if breakdown_flags.get("costume_like"):
    overall_score = _clamp(min(overall_score, 4.0))
    breakdown.trend_score = _clamp(min(breakdown.trend_score, 3.0))
    breakdown.style_match = _clamp(min(breakdown.style_match, 3.0))

  # Simple outfit protection
  if breakdown_flags.get("simple_clean"):
    overall_score = _clamp(max(overall_score, 7.0))

  quality_tier = _quality_tier(overall_score)

  # LLM suggestions (no heuristic fallback; propagate errors)
  suggestions, summary = await generate_suggestions(breakdown, user_ctx, image_bytes, image_url)
  if not suggestions:
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail="LLM suggestions unavailable; please retry shortly.",
    )

  # No suggestions fallback; any errors would have raised above

  warnings = []
  if summary:
    warnings.append(summary)
  if mask_cov is not None and mask_cov < 0.1:
    warnings.append("Full body not fully visible; results may be less accurate.")
  if color_metrics["brightness"] < 0.35:
    warnings.append("Low lighting detected; scores may be less accurate.")

  mc_term = 1 if (mask_cov is not None and mask_cov >= 0.05) else 0.5 if mask_cov is None else 0
  confidence_score = (
    mc_term
    + min(1, keypoints / 33)
    + top_sim
    + (1 if suggestions else 0)
  ) / 4

  logger.info(
    "drip_score={drip} breakdown={bd} brightness={bright:.2f} contrast={contrast:.2f} mask={mask:.2f} person_area_ratio={par:.2f} keypoints={kpts} avg_vis={avg_vis:.2f} top_prompt={prompt} top_sim={sim:.3f} llm_parse={llm_parse} conf={conf:.3f} img_w={w} img_h={h}",
    drip=overall_score,
    bd=breakdown.model_dump(),
    bright=color_metrics["brightness"],
    contrast=color_metrics["contrast"],
    mask=mask_cov if mask_cov is not None else -1,
    par=person_area_ratio,
    kpts=keypoints,
    avg_vis=avg_vis,
    prompt=top_prompt,
    sim=top_sim,
    llm_parse=bool(suggestions),
    conf=confidence_score,
    w=width,
    h=height,
  )

  return ScoreResponse(
    drip_score=overall_score,
    overall_score=overall_score,
    quality_tier=quality_tier,
    breakdown=breakdown,
    suggestions=suggestions,
    warnings=warnings,
  )


def fake_score(user_styles: List[str]) -> ScoreResponse:
  """Fallback stub when AI is unavailable."""

  def rand():
    return round(random.uniform(6.0, 9.5), 1)

  breakdown = ScoreBreakdown(
    color_match=_apply_noise(_normalize_score(rand())),
    fit_quality=_apply_noise(_normalize_score(rand())),
    body_compatibility=_apply_noise(_normalize_score(rand())),
    trend_score=_apply_noise(_normalize_score(rand())),
    style_match=_apply_noise(_normalize_score(rand())),
  )
  overall_score = _clamp(
    0.30 * breakdown.color_match
    + 0.20 * breakdown.fit_quality
    + 0.20 * breakdown.body_compatibility
    + 0.10 * breakdown.trend_score
    + 0.20 * breakdown.style_match
  )
  overall_score = _apply_noise(_normalize_score(overall_score))
  quality_tier = _quality_tier(overall_score)
  suggestions = [
    SuggestionCard(
      title="Fix lighting",
      type="other",
      description="Use brighter lighting so textures and colors are clear.",
    ),
    SuggestionCard(
      title="Frame the outfit",
      type="other",
      description="Frame the full outfit; avoid random objects in the shot.",
    ),
    SuggestionCard(
      title="Balance silhouette",
      type="fit",
      description="Try balancing top/bottom proportions for a cleaner silhouette.",
    ),
  ]
  warnings = [
    "AI scoring unavailable; showing a fallback score.",
  ]
  return ScoreResponse(
    drip_score=overall_score,
    overall_score=overall_score,
    quality_tier=quality_tier,
    breakdown=breakdown,
    suggestions=suggestions,
    warnings=warnings,
  )
