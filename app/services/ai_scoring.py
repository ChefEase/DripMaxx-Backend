import io
import statistics
import json
from typing import List, Sequence, Dict, Any

import replicate
import numpy as np
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image, ImageStat

from app.core.config import get_settings
from app.schemas.outfits import ScoreBreakdown, ScoreResponse, SuggestionCard, UserContext
from app.services.ai_suggestions import generate_suggestions

settings = get_settings()
DEFAULT_MODEL_REF = (
  "krthr/clip-embeddings:1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4"
)
VLM_MODEL_REF_DEFAULT = "openai/gpt-5.6-terra"
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


def _score_from_level(level: str) -> float:
  return {
    "bad": 2.0,
    "poor": 3.5,
    "average": 5.8,
    "good": 7.8,
    "excellent": 9.2,
  }.get(level.strip().lower(), 5.8)


def _normalize_color(color: str) -> str:
  value = color.strip().lower().replace("gray", "grey")
  aliases = {
    "off white": "white", "off-white": "white", "ivory": "cream",
    "charcoal": "grey", "silver": "grey", "burgundy": "red",
    "maroon": "red", "crimson": "red", "olive": "green",
    "khaki": "tan", "camel": "tan", "denim": "blue",
  }
  return aliases.get(value, value)


def _score_value(value: Any, labels: Dict[str, float], default: float) -> float:
  """Accept granular 0-10 values while remaining compatible with old labels."""
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return _clamp(float(value))
  text = str(value or "").strip().lower()
  try:
    return _clamp(float(text))
  except ValueError:
    return labels.get(text, default)


def _weighted_subscore(
  values: Any,
  weights: Dict[str, float],
  fallback: Any,
  labels: Dict[str, float],
  default: float,
) -> float:
  if isinstance(values, dict):
    valid = []
    for key, weight in weights.items():
      value = values.get(key)
      if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= float(value) <= 10:
        valid.append((float(value), weight))
    if valid:
      weight_total = sum(weight for _, weight in valid)
      return _clamp(sum(value * weight for value, weight in valid) / weight_total)
  return _score_value(fallback, labels, default)


def _eval_color_score(colors: List[str], penalties: Dict[str, Any], harmony_level: Any = "") -> float:
  """Score coordination, not image brightness or raw color count."""
  palette = list(dict.fromkeys(_normalize_color(c) for c in colors if c.strip()))
  count = len(palette)
  level_scores = {"excellent": 9.1, "good": 8.2, "average": 6.8, "poor": 4.8, "bad": 3.2}
  score = _score_value(harmony_level, level_scores, 7.2)

  if penalties.get("clashing_patterns"):
    score -= 1.0
  if penalties.get("neon_colors") and str(harmony_level).strip().lower() not in {"good", "excellent"}:
    score -= 0.8
  if penalties.get("too_many_colors") or count > 4:
    score -= 1.2
  return _clamp(score)


def _eval_fit_score(fit_style: str, silhouette: str, body_type: str, quality_level: Any = "") -> float:
  score = _score_value(
    quality_level,
    {"excellent": 8.9, "good": 7.8, "average": 6.4, "poor": 4.7, "bad": 3.2},
    6.8,
  )
  if fit_style in ("tailored", "fitted", "balanced"):
    score += 0.4
  if fit_style in ("extremely_baggy", "extremely_tight"):
    score -= 2.0
  if silhouette == "balanced":
    score += 0.3
  if silhouette == "imbalanced":
    score -= 1.2
  # basic body compatibility tweaks
  if body_type == "slim" and fit_style in ("layered", "tailored", "balanced"):
    score += 0.2
  if body_type == "athletic" and fit_style in ("relaxed", "streetwear", "balanced"):
    score += 0.2
  if body_type == "broad" and fit_style in ("structured", "tailored"):
    score += 0.3
  if body_type == "plus_size" and silhouette == "balanced":
    score += 0.3
  return _clamp(score)


def _eval_trend_score(items: List[str], trend_hits: List[str], penalties: Dict[str, Any], relevance_level: Any = "") -> float:
  score = _score_value(
    relevance_level,
    {"excellent": 9.0, "good": 8.0, "average": 6.6, "poor": 4.8, "bad": 3.2},
    min(8.5, 6.0 + 0.8 * len(trend_hits)),
  )
  score += min(0.6, 0.2 * len(trend_hits))
  if penalties.get("costume_like"):
    score = min(score, 3.0)
  return _clamp(score)


def _eval_style_match(user_styles: List[str], style_probs: Dict[str, float], match_level: Any = "") -> float:
  if not user_styles:
    return 6.0
  direct_score = _score_value(
    match_level,
    {"excellent": 9.1, "good": 8.0, "average": 6.4, "poor": 4.6, "bad": 3.0},
    -1.0,
  )
  normalized_probs = {
    str(key).strip().lower().replace(" ", "_"): float(value)
    for key, value in style_probs.items()
  }
  total = 0.0
  for s in user_styles:
    key = str(s).strip().lower().replace(" ", "_")
    total += normalized_probs.get(key, 0.0)
  probability_score = 5.0 + min(4.5, total * 5.0)
  # The direct judgment supports custom styles that are not classifier keys.
  return _clamp(direct_score if direct_score >= 0 else probability_score)


def _normalize_height_bucket(height_text: str) -> str:
  text = (height_text or "").strip().lower()
  if not text:
    return "unknown"
  short_markers = ("4'", "5'0", "5'1", "5'2", "5'3", "5'4", "5'5", "5 ft", "160", "161", "162", "163", "164", "165")
  tall_markers = ("6'", "6'0", "6'1", "6'2", "6'3", "6'4", "183", "184", "185", "186", "187", "188", "189", "190")
  if any(marker in text for marker in short_markers):
    return "short"
  if any(marker in text for marker in tall_markers):
    return "tall"
  return "average"


def _eval_body_compatibility(
  body_type: str,
  user_height: str,
  gender_style: str,
  fit_style: str,
  silhouette: str,
  detected_items: List[str],
  trend_hits: List[str],
  layer_count: int,
  compatibility_level: Any = "",
) -> float:
  level_base = _score_value(
    compatibility_level,
    {"excellent": 8.9, "good": 7.8, "average": 6.3, "poor": 4.6, "bad": 3.1},
    -1.0,
  )
  score = level_base if level_base >= 0 else 6.2
  body_type = (body_type or "").strip().lower()
  height_bucket = _normalize_height_bucket(user_height)
  gender_style = (gender_style or "").strip().lower()
  items = {str(item).strip().lower() for item in detected_items if str(item).strip()}
  trend_set = {str(hit).strip().lower() for hit in trend_hits if str(hit).strip()}

  if fit_style in ("balanced", "tailored", "structured"):
    score += 0.8
  elif fit_style in ("relaxed", "fitted"):
    score += 0.3
  elif fit_style in ("extremely_baggy", "extremely_tight"):
    score -= 1.2

  if silhouette == "balanced":
    score += 0.9
  elif silhouette == "imbalanced":
    score -= 1.0

  if body_type == "slim":
    if fit_style in ("tailored", "balanced", "layered"):
      score += 0.6
    if fit_style == "extremely_baggy":
      score -= 0.8
  elif body_type == "athletic":
    if fit_style in ("balanced", "relaxed", "structured"):
      score += 0.6
    if fit_style == "extremely_tight":
      score -= 0.6
  elif body_type == "broad":
    if fit_style in ("structured", "tailored", "balanced"):
      score += 0.7
    if "blazer" in items or "jacket" in items or "coat" in items:
      score += 0.3
    if fit_style == "extremely_tight":
      score -= 0.7
  elif body_type == "plus_size":
    if fit_style in ("balanced", "structured", "tailored"):
      score += 0.8
    if silhouette == "balanced":
      score += 0.4
    if fit_style == "extremely_tight":
      score -= 0.9

  if height_bucket == "short":
    if fit_style == "extremely_baggy":
      score -= 0.8
    if "vertical_stripes" in trend_set or silhouette == "balanced":
      score += 0.4
  elif height_bucket == "tall":
    if fit_style in ("relaxed", "balanced", "structured"):
      score += 0.3
    if layer_count >= 2:
      score += 0.2

  if gender_style in ("menswear", "men", "male"):
    if fit_style in ("structured", "tailored", "balanced"):
      score += 0.2
  elif gender_style in ("womenswear", "women", "female"):
    if fit_style in ("fitted", "tailored", "balanced"):
      score += 0.2

  if level_base >= 0:
    # Let concrete profile/fit evidence refine the visual judgment without
    # overwhelming it and forcing most results into the same high range.
    score = level_base + (score - level_base) * 0.35
  return _clamp(score)


def _styling_adjustments(
  detected_items: List[str],
  trend_hits: List[str],
  top_type: str,
  shoe_type: str,
  layer_count: int,
  fit_style: str,
  silhouette: str,
) -> Dict[str, float]:
  items = {str(item).strip().lower() for item in detected_items if str(item).strip()}
  top = (top_type or "").strip().lower()
  shoes = (shoe_type or "").strip().lower()
  trend_set = {str(hit).strip().lower() for hit in trend_hits if str(hit).strip()}

  accessories_present = any(
    token in items
    for token in {
      "watch", "bracelet", "chain", "necklace", "ring", "earrings", "jewelry",
      "bag", "scarf", "beads", "durag", "head tie", "headtie", "bandana", "belt",
      "hat", "cap", "beanie",
    }
  )
  elevated_accessories = any(
    token in items for token in {"watch", "bracelet", "chain", "necklace", "ring", "earrings", "jewelry", "scarf", "belt", "bag"}
  )
  headwear_present = any(token in items for token in {"hat", "cap", "beanie", "durag", "head tie", "headtie", "bandana"})
  formal_top = any(token in top for token in {"suit", "blazer", "dress shirt", "button-up", "button up", "button-down", "button down", "tie"})
  casual_top = any(token in top for token in {"t-shirt", "tee", "hoodie", "sweatshirt"})
  visible_layering = layer_count >= 2 or "inner layer" in trend_set or "layered" in trend_set

  fit_bonus = 0.0
  trend_bonus = 0.0
  style_bonus = 0.0

  if accessories_present:
    style_bonus += 0.3
    trend_bonus += 0.2
  if elevated_accessories and silhouette == "balanced":
    style_bonus += 0.4
  if visible_layering and casual_top:
    fit_bonus += 0.3
    style_bonus += 0.5
    trend_bonus += 0.4
  if visible_layering and fit_style in ("balanced", "tailored", "structured"):
    style_bonus += 0.2

  if formal_top and headwear_present and not any(token in items for token in {"beret", "fedora"}):
    style_bonus -= 0.9
    trend_bonus -= 0.6
  if formal_top and shoes in {"", "none", "barefoot", "socks"}:
    fit_bonus -= 0.5
    style_bonus -= 0.6
  elif shoes in {"", "none", "barefoot", "socks"}:
    fit_bonus -= 0.3
    style_bonus -= 0.2
  if casual_top and visible_layering:
    trend_bonus += 0.2

  return {
    "fit": fit_bonus,
    "trend": trend_bonus,
    "style": style_bonus,
  }


def _quality_tier(overall: float) -> str:
  if overall >= 9.0:
    return "Top_Notch"
  if overall >= 7.0:
    return "Good"
  if overall >= 5.0:
    return "Mid"
  return "Bad"


def _overall_from_available_metrics(
  breakdown: ScoreBreakdown,
  *,
  has_style_target: bool,
  has_body_profile: bool,
) -> float:
  """Combine only metrics the user supplied enough context to personalize."""
  if has_style_target and has_body_profile:
    weights = {
      "color_match": 0.30, "fit_quality": 0.20, "body_compatibility": 0.20,
      "trend_score": 0.10, "style_match": 0.20,
    }
  elif has_style_target:
    weights = {
      "color_match": 0.36, "fit_quality": 0.27, "trend_score": 0.15,
      "style_match": 0.22,
    }
  elif has_body_profile:
    weights = {
      "color_match": 0.37, "fit_quality": 0.27, "body_compatibility": 0.20,
      "trend_score": 0.16,
    }
  else:
    weights = {"color_match": 0.44, "fit_quality": 0.34, "trend_score": 0.22}
  return _clamp(sum(getattr(breakdown, metric) * weight for metric, weight in weights.items()))


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


async def _vlm_attributes(
  image_bytes: bytes, user_ctx: UserContext, image_url: str | None = None
) -> tuple[Dict[str, Any], dict] | None:
  """Ask the VLM for visual attributes and penalties; return attributes."""
  if not settings.replicate_api_token:
    return None
  model_ref = settings.replicate_vlm_model or VLM_MODEL_REF_DEFAULT

  sys_prompt = (
    "Rules (follow strictly):\n"
    "- If no clear outfit is visible, set outfit_present=false and leave clothing fields empty.\n"
    "- If multiple people are visible, choose ONE main subject: the clearest full-outfit person closest to the center. Rate only that person's outfit and ignore everyone else.\n"
    "- Do not reject an image only because multiple people are visible.\n"
    "- Do not guess clothing when only a face/close-up is present.\n"
    "- Only list items/colors that are visible.\n"
    "- Monochrome means the SAME color family (1 color). Black+white+blue is NOT monochrome.\n"
    "- Judge color_harmony as fashion coordination, not brightness: intentional monochrome can be excellent; "
    "a neutral base with one repeated accent (for example a red hat and red/white shoes) can be excellent; "
    "black/white and black-on-black are usually good or excellent unless materials/patterns visibly clash.\n"
    "- Reserve poor/bad color_harmony for genuinely competing hues, uncontrolled color count, or clashing patterns.\n"
    "- fit_quality judges whether garments sit intentionally on this person: proportions, stacking, breaks, "
    "silhouette and visible bunching. Do not call every balanced outfit good; use the full scale.\n"
    "- trend_relevance judges current styling and execution, not merely the presence of common clothes. "
    "A timeless coherent outfit may be good; excellent requires notably current or distinctive execution.\n"
    "- Score every visible criterion independently from 0.0 to 10.0 with one decimal. Do not use words such "
    "as good or excellent and do not round several criteria to the same convenient number.\n"
    "- Calibration anchors: 5.0 is ordinary/functional, 6.5 is solid, 7.5 is clearly strong, 8.5 is exceptional, "
    "9.3+ is rare editorial-level execution. A clean or neutral outfit is not automatically 8+.\n"
    "- selected_style_match judges the outfit against the user's exact requested styles, including custom "
    "free-text styles not listed in style_probs. If no style was requested, return null for style subscores.\n"
    "- body_compatibility judges how the visible proportions, silhouette, garment lengths and fit work on "
    "the photographed person. Use supplied height/body/gender-style context when available; otherwise judge "
    "only when profile context was supplied. If it was not supplied, return null for body subscores.\n"
    "- If inner_layer_visible is true, layer_count must be >= 1.\n"
    "- Use collar_visible only as a supporting hint.\n"
    "- style_probs are probabilities 0-1.\n"
    "- Each style probability must be present (even if very low like 0.05).\n"
    "- If unsure about colors, lower color_confidence.\n"
    "Output format (JSON only):\n"
    "{"
    "\"outfit_present\": true|false,"
    "\"selected_person\": \"brief description of the person/outfit you chose\","
    "\"top_type\": \"\","
    "\"pants_type\": \"\","
    "\"shoe_type\": \"\","
    "\"accessories\": [\"watch\",\"chain\"],"
    "\"top_color\": \"\","
    "\"pants_color\": \"\","
    "\"shoe_color\": \"\","
    "\"primary_colors\": [\"black\",\"white\"],"
    "\"color_confidence\": 0.0,"
    "\"color_harmony\": 0.0,"
    "\"color_scores\": {\"harmony\":0.0,\"palette_intentionality\":0.0,\"contrast_balance\":0.0,\"material_coordination\":0.0},"
    "\"fit_style\": \"relaxed|tailored|balanced|extremely_baggy|extremely_tight\","
    "\"fit_quality\": 0.0,"
    "\"fit_scores\": {\"proportion_balance\":0.0,\"garment_fit\":0.0,\"silhouette_execution\":0.0,\"layering_and_finishing\":0.0},"
    "\"layer_count\": 0,"
    "\"collar_visible\": true|false,"
    "\"inner_layer_visible\": true|false,"
    "\"pattern_type\": \"solid|patterned\","
    "\"silhouette_balance\": \"balanced|imbalanced\","
    "\"style_probs\": {\"streetwear\":0.0,\"minimal\":0.0,\"casual\":0.0,\"luxury\":0.0,\"vintage\":0.0,\"y2k\":0.0,\"athleisure\":0.0,\"smart_casual\":0.0,\"experimental\":0.0},"
    "\"selected_style_match\": 0.0|null,"
    "\"style_scores\": {\"target_alignment\":0.0,\"styling_execution\":0.0,\"detail_consistency\":0.0},"
    "\"body_compatibility\": 0.0|null,"
    "\"body_scores\": {\"proportion_support\":0.0,\"garment_length\":0.0,\"silhouette_support\":0.0},"
    "\"trend_hits\": [\"relaxed_denim\",\"oversized_hoodies\"],"
    "\"trend_relevance\": 0.0,"
    "\"trend_scores\": {\"current_relevance\":0.0,\"styling_intent\":0.0,\"distinctiveness\":0.0,\"timeless_execution\":0.0},"
    "\"detected_items\": [\"hoodie\",\"jeans\",\"sneakers\",\"watch\"],"
    "\"penalties\": {"
    "\"excessive_monochrome\": true|false,"
    "\"neon_colors\": true|false,"
    "\"clashing_patterns\": true|false,"
    "\"costume_like\": true|false,"
    "\"poor_layering\": true|false,"
    "\"too_many_colors\": true|false,"
    "\"simple_clean\": true|false"
    "}"
    "}"
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
            "system_prompt": sys_prompt,
            "prompt": user_prompt,
            "image_input": [image_input],
            "verbosity": "low",
            "reasoning_effort": "low",
            "max_completion_tokens": 800,
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
    return data, penalties
  except Exception as exc:
    logger.error(f"VLM scoring failed; raw='{raw[:400] if 'raw' in locals() else ''}' err={exc}")
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail="VLM scoring unavailable; please retry shortly.",
    ) from exc


def _normalize_user_context(user_ctx: UserContext) -> UserContext:
  missing_markers = {"", "n/a", "na", "none", "null", "unspecified"}
  return user_ctx.model_copy(
    update={
      "style_preferences": [
        style for style in user_ctx.style_preferences
        if str(style).strip().lower() not in missing_markers
      ],
      "style_inspirations": [
        inspiration for inspiration in user_ctx.style_inspirations
        if str(inspiration).strip().lower() not in missing_markers
      ],
      "user_height": (
        None
        if str(user_ctx.user_height or "").strip().lower() in missing_markers
        else user_ctx.user_height
      ),
      "user_body_type": (
        None
        if str(user_ctx.user_body_type or "").strip().lower() in missing_markers
        else user_ctx.user_body_type
      ),
      "gender_style_preference": (
        None
        if str(user_ctx.gender_style_preference or "").strip().lower() in missing_markers
        else user_ctx.gender_style_preference
      ),
    }
  )


async def score_with_ai(
  image_bytes: bytes, user_ctx: UserContext, image_url: str | None = None
) -> ScoreResponse:
  """Generate clip-like embeddings via Replicate, derive Drip Score, and emit UX suggestions."""
  user_ctx = _normalize_user_context(user_ctx)
  logger.info("score_with_ai stage=start image_url_present={}", bool(image_url))
  image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  width, height = image.size
  logger.info("score_with_ai stage=image_loaded width={} height={}", width, height)
  if width < 512 or height < 768:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Image resolution too low. Upload a clearer full-body photo.",
    )
  keypoints = 0
  avg_vis = 0.0

  # Try VLM for grounded numeric scores first (if configured)
  breakdown = None
  breakdown_mode = "numeric"
  breakdown_flags: dict = {}
  attr_data: Dict[str, Any] | None = None
  if settings.replicate_vlm_model:
    logger.info("score_with_ai stage=vlm_start model={}", settings.replicate_vlm_model)
    attrs = await _vlm_attributes(image_bytes, user_ctx, image_url)
    if attrs:
      logger.info("score_with_ai stage=vlm_done")
      attr_data, breakdown_flags = attrs
      outfit_present = attr_data.get("outfit_present")
      if outfit_present is False:
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="No outfit, no rating.",
        )
      detected_items = [str(x).lower() for x in (attr_data.get("detected_items") or [])]
      detected_items.extend(
        [str(x).lower() for x in (attr_data.get("accessories") or []) if isinstance(x, str)]
      )
      detected_items = list(dict.fromkeys([x for x in detected_items if x]))
      top_type = str(attr_data.get("top_type") or "").strip().lower()
      pants_type = str(attr_data.get("pants_type") or "").strip().lower()
      shoe_type = str(attr_data.get("shoe_type") or "").strip().lower()
      if not detected_items and not (top_type or pants_type or shoe_type):
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="No outfit, no rating.",
        )
      colors = []
      for key in ("top_color", "pants_color", "shoe_color"):
        val = attr_data.get(key)
        if isinstance(val, str) and val.strip():
          colors.append(val.strip().lower())
      colors.extend([c.lower() for c in (attr_data.get("primary_colors") or []) if isinstance(c, str)])
      # Quick color override for jeans
      if ("jeans" in detected_items or "denim" in detected_items) and "blue" not in colors:
        colors.append("blue")
      # Normalize palette
      palette = []
      for c in colors:
        if c and c not in palette:
          palette.append(c)
      color_conf = float(attr_data.get("color_confidence") or 0.0)
      fit_style = str(attr_data.get("fit_style") or "").lower()
      silhouette = str(attr_data.get("silhouette_balance") or "").lower()
      style_probs = attr_data.get("style_probs") or {}
      trend_hits = [t for t in (attr_data.get("trend_hits") or []) if isinstance(t, str)]
      # Layering heuristic: hoodie + inner layer or visible collar
      layer_count = int(attr_data.get("layer_count") or 0)
      if attr_data.get("inner_layer_visible"):
        layer_count += 1
      if "hoodie" in detected_items:
        layer_count += 1
      layer_count = min(3, layer_count)
      attr_data["layer_count"] = layer_count
      # Monochrome logic override when confidence is high
      neutrals = {"black", "white", "grey", "gray", "beige", "cream", "brown", "tan", "navy"}
      # If colors are uncertain, do not apply harsh penalties.
      if color_conf < 0.7:
        breakdown_flags["excessive_monochrome"] = False
        breakdown_flags["neon_colors"] = False
        breakdown_flags["too_many_colors"] = False
      else:
        if len(palette) == 1:
          breakdown_flags["excessive_monochrome"] = True
        else:
          breakdown_flags["excessive_monochrome"] = False
      # Solid patterns should not trigger clashing patterns
      if str(attr_data.get("pattern_type") or "").lower() == "solid":
        breakdown_flags["clashing_patterns"] = False
      # Too many colors
      if len(palette) > 4:
        breakdown_flags["too_many_colors"] = True
      color_base = _weighted_subscore(
        attr_data.get("color_scores"),
        {"harmony": 0.40, "palette_intentionality": 0.25, "contrast_balance": 0.20, "material_coordination": 0.15},
        attr_data.get("color_harmony"),
        {"excellent": 9.1, "good": 8.2, "average": 6.8, "poor": 4.8, "bad": 3.2},
        7.2,
      )
      fit_base = _weighted_subscore(
        attr_data.get("fit_scores"),
        {"proportion_balance": 0.30, "garment_fit": 0.30, "silhouette_execution": 0.25, "layering_and_finishing": 0.15},
        attr_data.get("fit_quality"),
        {"excellent": 8.9, "good": 7.8, "average": 6.4, "poor": 4.7, "bad": 3.2},
        6.8,
      )
      trend_base = _weighted_subscore(
        attr_data.get("trend_scores"),
        {"current_relevance": 0.35, "styling_intent": 0.25, "distinctiveness": 0.20, "timeless_execution": 0.20},
        attr_data.get("trend_relevance"),
        {"excellent": 9.0, "good": 8.0, "average": 6.6, "poor": 4.8, "bad": 3.2},
        6.6,
      )
      color_score = _eval_color_score(
        palette,
        breakdown_flags,
        color_base,
      )
      fit_score = _eval_fit_score(
        fit_style,
        silhouette,
        (user_ctx.user_body_type or "").lower(),
        fit_base,
      )
      trend_score = _eval_trend_score(
        detected_items,
        trend_hits,
        breakdown_flags,
        trend_base,
      )
      styling_adj = _styling_adjustments(
        detected_items,
        trend_hits,
        top_type,
        shoe_type,
        layer_count,
        fit_style,
        silhouette,
      )
      fit_score = _clamp(fit_score + styling_adj["fit"])
      trend_score = _clamp(trend_score + styling_adj["trend"])
      # Fallback style probs if classifier is empty
      if isinstance(style_probs, dict):
        max_prob = max(style_probs.values() or [0.0])
      else:
        style_probs = {}
        max_prob = 0.0
      if max_prob < 0.1:
        if {"hoodie", "sneakers", "jeans"}.issubset(set(detected_items)):
          style_probs["streetwear"] = 0.7
          style_probs["casual"] = 0.8
          style_probs["minimal"] = 0.4
      # Ensure non-zero probabilities to avoid all-zero outputs
      if not style_probs or max_prob == 0.0:
        style_probs = {
          "streetwear": 0.05,
          "minimal": 0.05,
          "casual": 0.05,
          "luxury": 0.05,
          "vintage": 0.05,
          "y2k": 0.05,
          "athleisure": 0.05,
          "smart_casual": 0.05,
          "experimental": 0.05,
        }
      # Normalize style_probs so total <= 1.0
      if style_probs:
        total = sum(style_probs.values())
        if total > 1.0 and total > 0:
          for k in list(style_probs.keys()):
            style_probs[k] = style_probs[k] / total
      style_base = _weighted_subscore(
        attr_data.get("style_scores"),
        {"target_alignment": 0.50, "styling_execution": 0.30, "detail_consistency": 0.20},
        attr_data.get("selected_style_match"),
        {"excellent": 9.1, "good": 8.0, "average": 6.4, "poor": 4.6, "bad": 3.0},
        6.0,
      )
      style_score = _eval_style_match(
        user_ctx.style_preferences or [],
        style_probs,
        style_base,
      )
      style_score = _clamp(style_score + styling_adj["style"])
      body_base = _weighted_subscore(
        attr_data.get("body_scores"),
        {"proportion_support": 0.40, "garment_length": 0.30, "silhouette_support": 0.30},
        attr_data.get("body_compatibility"),
        {"excellent": 8.9, "good": 7.8, "average": 6.3, "poor": 4.6, "bad": 3.1},
        6.2,
      )
      body_score = _eval_body_compatibility(
        user_ctx.user_body_type or "",
        user_ctx.user_height or "",
        user_ctx.gender_style_preference or "",
        fit_style,
        silhouette,
        detected_items,
        trend_hits,
        layer_count,
        body_base,
      )
      breakdown = ScoreBreakdown(
        color_match=color_score,
        fit_quality=fit_score,
        body_compatibility=body_score,
        trend_score=trend_score,
        style_match=style_score,
      )
      breakdown_mode = "rules"
      logger.info("score pipeline: using rule-based breakdown (model=%s)", settings.replicate_vlm_model)
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
  has_style_target = bool(user_ctx.style_preferences)
  has_body_profile = bool(
    (user_ctx.user_height or "").strip()
    or (user_ctx.user_body_type or "").strip()
    or (user_ctx.gender_style_preference or "").strip()
  )
  overall_score = _overall_from_available_metrics(
    breakdown,
    has_style_target=has_style_target,
    has_body_profile=has_body_profile,
  )
  if breakdown_mode == "numeric":
    overall_score = _normalize_score(overall_score)

  # Costume or absurd outfits get capped low
  if breakdown_flags.get("costume_like"):
    overall_score = _clamp(min(overall_score, 4.0))
    breakdown.trend_score = _clamp(min(breakdown.trend_score, 3.0))
    breakdown.style_match = _clamp(min(breakdown.style_match, 3.0))

  quality_tier = _quality_tier(overall_score)
  unavailable_metrics = []
  if not has_body_profile:
    unavailable_metrics.append("body_compatibility")
  if not has_style_target:
    unavailable_metrics.append("style_match")

  # LLM suggestions (no heuristic fallback; propagate errors)
  logger.info("score_with_ai stage=suggestions_start")
  suggestions, summary, target_look = await generate_suggestions(
    breakdown,
    user_ctx,
    image_bytes,
    image_url,
    visual_context=attr_data,
  )
  if not suggestions:
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail="LLM suggestions unavailable; please retry shortly.",
    )
  logger.info("score_with_ai stage=suggestions_done count={}", len(suggestions))

  # No suggestions fallback; any errors would have raised above

  warnings = []
  if summary:
    warnings.append(summary)
  if color_metrics["brightness"] < 0.35:
    warnings.append("Low lighting detected; scores may be less accurate.")

  confidence_score = (
    min(1, keypoints / 33)
    + top_sim
    + (1 if suggestions else 0)
  ) / 3

  logger.info(
    "drip_score={drip} breakdown={bd} brightness={bright:.2f} contrast={contrast:.2f} keypoints={kpts} avg_vis={avg_vis:.2f} top_prompt={prompt} top_sim={sim:.3f} llm_parse={llm_parse} conf={conf:.3f} img_w={w} img_h={h}",
    drip=overall_score,
    bd=breakdown.model_dump(),
    bright=color_metrics["brightness"],
    contrast=color_metrics["contrast"],
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
    unavailable_metrics=unavailable_metrics,
    visual_analysis=attr_data or {},
    target_look=target_look,
  )


def fake_score(user_styles: List[str]) -> ScoreResponse:
  """Fallback stub when AI is unavailable."""

  breakdown = ScoreBreakdown(
    color_match=7.0,
    fit_quality=7.0,
    body_compatibility=7.0,
    trend_score=7.0,
    style_match=7.0,
  )
  overall_score = _overall_from_available_metrics(
    breakdown,
    has_style_target=bool(user_styles),
    has_body_profile=False,
  )
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
    unavailable_metrics=["body_compatibility"] + ([] if user_styles else ["style_match"]),
  )
