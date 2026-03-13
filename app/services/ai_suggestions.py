import json
import io
import time
import re
from typing import List, Tuple, Dict, Any

import replicate
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.core.config import get_settings
from app.schemas.outfits import ScoreBreakdown, SuggestionCard, UserContext

settings = get_settings()


def _parse_detection(raw: str) -> Dict[str, Any]:
  text = raw.strip()
  if "{" in text and "}" in text:
    text = text[text.index("{") : text.rindex("}") + 1]
  text = text.replace("\\_", "_").strip()
  try:
    data = json.loads(text)
    if isinstance(data, dict):
      return data
  except Exception:
    pass
  return {}


def _tokenize(s: str) -> set:
  return set(re.findall(r"[a-z0-9]+", s.lower()))


def _too_similar(a: str, b: str) -> bool:
  ta = _tokenize(a)
  tb = _tokenize(b)
  if not ta or not tb:
    return False
  overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
  return overlap > 0.85


def _safe_summary_from_scores(breakdown: ScoreBreakdown) -> str:
  parts = []
  if breakdown.fit_quality < 6 or breakdown.body_compatibility < 6:
    parts.append("fit could be stronger")
  if breakdown.color_match < 6:
    parts.append("color balance lacks contrast")
  if breakdown.trend_score < 5.5:
    parts.append("trend alignment is a bit behind")
  if breakdown.style_match < 6:
    parts.append("overall cohesion needs polish")
  if not parts:
    parts.append("overall fit and balance are solid")
  return "Summary: " + ", ".join(parts) + "."


def _is_valid_detection(detection: Dict[str, Any]) -> bool:
  if not isinstance(detection, dict):
    return False
  items = detection.get("detected_items")
  problems = detection.get("problems_detected")
  improvements = detection.get("improvements")
  if not isinstance(items, list) or not isinstance(problems, list) or not isinstance(improvements, list):
    return False
  if not any(isinstance(s, str) and s.strip() for s in improvements):
    return False
  return True


async def generate_suggestions(
  breakdown: ScoreBreakdown, user_ctx: UserContext, image_bytes: bytes, image_url: str | None = None
) -> Tuple[List[SuggestionCard], str | None]:
  """Detect items/problems via VLM, then generate 5 templated suggestions."""
  if not settings.replicate_api_token:
    raise HTTPException(
      status_code=503,
      detail="Replicate API token missing for suggestions.",
    )

  sys_prompt = (
    "Look at the image and output ONLY JSON with detected items, problems, improvements, and a short summary:\n"
    "{"
    "\"detected_items\": [\"shirt\",\"pants\",\"shoes\",\"jacket\",\"hat\",\"belt\",\"dress\",\"skirt\",\"boots\",\"sneakers\",\"hoodie\",\"coat\",\"bag\",\"jewelry\"],"
    "\"problems_detected\": [\"monochrome_color\",\"lack_of_layers\",\"clashing_patterns\",\"poor_fit\",\"low_trend\",\"low_cohesion\"],"
    "\"improvements\": [\"short, concrete improvement\"],"
    "\"summary\": \"one sentence\""
    "}\n"
    "Rules:\n"
    "- Only list items that are clearly visible.\n"
    "- problems_detected can be empty.\n"
    "- improvements must be image-based, concrete, and at least one item.\n"
    "- Summary must be one sentence about fit/color/overall balance.\n"
    "Output only JSON."
  )
  user_prompt = (
    f"User style prefs: {', '.join(user_ctx.style_preferences) or 'unspecified'}; "
    f"inspirations: {', '.join(user_ctx.style_inspirations) or 'unspecified'}; "
    f"height: {user_ctx.user_height or 'n/a'}; body_type: {user_ctx.user_body_type or 'n/a'}; "
    f"gender_style: {user_ctx.gender_style_preference or 'n/a'}. "
    f"Focus on weakest scores first: "
    f"color_match={breakdown.color_match}, fit_quality={breakdown.fit_quality}, "
    f"body_compatibility={breakdown.body_compatibility}, trend_score={breakdown.trend_score}, "
    f"style_match={breakdown.style_match}. "
    "Output JSON only."
  )

  def _call_vlm(prompt: str, temperature: float):
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=60)
    model_ref = settings.replicate_vlm_model
    image_input = image_url
    if not image_input:
      file_obj = io.BytesIO(image_bytes)
      file_obj.name = "upload.jpg"
      image_input = file_obj
    tries = 0
    while True:
      tries += 1
      try:
        result = client.run(
          model_ref,
          input={
            "image": image_input,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": 800,
          },
        )
        break
      except replicate.exceptions.ReplicateError as exc:
        if exc.status == 429 and tries == 1:
          time.sleep(4)
          continue
        raise
    if isinstance(result, (list, tuple)):
      return "".join(str(x) for x in result)
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
      return "".join(str(x) for x in result)
    return str(result)

  base_prompt = sys_prompt + "\n\n" + user_prompt
  raw = await run_in_threadpool(lambda: _call_vlm(base_prompt, 0.2))
  detection = _parse_detection(raw)
  if not _is_valid_detection(detection):
    strict_prompt = base_prompt + "\n\nReturn ONLY valid JSON. No markdown. No extra text."
    raw = await run_in_threadpool(lambda: _call_vlm(strict_prompt, 0.0))
    detection = _parse_detection(raw)
  if not _is_valid_detection(detection):
    repair_prompt = (
      "Fix and return ONLY valid JSON for this schema:\n"
      "{"
      "\"detected_items\": [\"shirt\",\"pants\",\"shoes\",\"jacket\",\"hat\",\"belt\",\"dress\",\"skirt\",\"boots\",\"sneakers\",\"hoodie\",\"coat\",\"bag\",\"jewelry\"],"
      "\"problems_detected\": [\"monochrome_color\",\"lack_of_layers\",\"clashing_patterns\",\"poor_fit\",\"low_trend\",\"low_cohesion\"],"
      "\"improvements\": [\"short, concrete improvement\"],"
      "\"summary\": \"one sentence\""
      "}\n"
      "Rules:\n"
      "- improvements must be image-based, concrete, and at least one item.\n"
      "- Output only JSON.\n\n"
      f"RAW:\n{raw}"
    )
    raw = await run_in_threadpool(lambda: _call_vlm(repair_prompt, 0.0))
    detection = _parse_detection(raw)
  detected_items = set(i.lower() for i in (detection.get("detected_items") or []) if isinstance(i, str))
  problems = set(p.lower() for p in (detection.get("problems_detected") or []) if isinstance(p, str))
  improvements = [
    re.sub(r"\s+", " ", s).strip()
    for s in (detection.get("improvements") or [])
    if isinstance(s, str) and s.strip()
  ]
  summary = _safe_summary_from_scores(breakdown)

  def has_item(*names: str) -> bool:
    return any(n in detected_items for n in names)

  blocked_items = {"belt", "tie", "hat", "socks", "jacket", "coat", "overshirt"}

  candidates = []
  if breakdown.color_match < 6 or "monochrome_color" in problems:
    candidates.append(("color", "Add color contrast", "Introduce a second color to break the monochrome look.", "overall"))
  if breakdown.fit_quality < 6 or "poor_fit" in problems:
    candidates.append(("fit", "Refine the fit", "Adjust proportions with a cleaner fit on top or bottom.", "overall"))
  if breakdown.trend_score < 5.5 or "low_trend" in problems:
    candidates.append(("other", "Modernize one piece", "Swap one item for a more current silhouette or finish.", "overall"))
  if breakdown.style_match < 6 or "low_cohesion" in problems:
    candidates.append(("layering", "Improve cohesion", "Use a cohesive layer or texture to unify the outfit.", "overall"))

  if ("lack_of_layers" in problems) and has_item("jacket", "coat", "hoodie", "overshirt"):
    candidates.append(("layering", "Add a clean layer", "A light outer layer can add depth without clutter.", "upper"))

  if has_item("jewelry", "bag"):
    candidates.append(("accessory", "Elevate with accessories", "Lean into visible accessories for a more finished look.", "overall"))

  if has_item("shoes", "boots", "sneakers"):
    candidates.append(("fit", "Upgrade footwear texture", "Choose a more textured shoe to add visual interest.", "lower"))

  if has_item("jacket", "coat", "overshirt"):
    candidates.append(("layering", "Structure the outer layer", "Keep outerwear structured to sharpen the silhouette.", "upper"))
  if has_item("hoodie"):
    candidates.append(("fit", "Clean up hoodie shape", "Keep the hoodie fit clean so it doesn't bunch.", "upper"))
  if has_item("dress", "skirt"):
    candidates.append(("fit", "Define the waistline", "A subtle waist definition can improve proportions.", "overall"))
  if has_item("pants", "skirt", "dress"):
    candidates.append(("fit", "Tighten the hem", "A sharper hem length improves the leg line.", "lower"))
  if has_item("sneakers"):
    candidates.append(("other", "Keep sneakers crisp", "Clean, crisp sneakers elevate the base of the look.", "lower"))

  for imp in improvements:
    words = imp.split()
    title = " ".join(words[:6]) if words else imp
    if title:
      title = title[0].upper() + title[1:]
    desc = imp if imp.endswith(".") else f"{imp}."
    candidates.append(("other", title or "Improve the look", desc, "overall"))
  filtered = []
  for t, title, desc, area in candidates:
    lower = f"{title} {desc}".lower()
    if any(b in lower for b in blocked_items) and not has_item("jacket", "coat", "overshirt", "hat", "belt", "tie", "socks"):
      continue
    filtered.append((t, title, desc, area))

  ordered = []
  used_areas = set()
  for t, title, desc, area in filtered:
    if area not in used_areas:
      ordered.append((t, title, desc, area))
      used_areas.add(area)
    if len(ordered) >= 3:
      break
  for t, title, desc, area in filtered:
    if len(ordered) >= 5:
      break
    ordered.append((t, title, desc, area))

  final = []
  for t, title, desc, area in ordered:
    text = f"{title} {desc}"
    if any(_too_similar(text, f"{c.title} {c.description}") for c in final):
      continue
    final.append(SuggestionCard(title=title, type=t, description=desc))
    if len(final) >= 5:
      break

  if not final:
    logger.error(f"VLM suggestion parse failed; raw='{raw[:800]}'")
    raise HTTPException(
      status_code=502,
      detail="LLM suggestions unavailable; please retry shortly.",
    )

  formatted = []
  for idx, card in enumerate(final, start=1):
    if idx <= 3:
      card.title = f"Top Fix {idx}: {card.title}"
    else:
      card.title = f"Optional {idx}: {card.title}"
    formatted.append(card)

  logger.debug(f"VLM suggestions templated {len(formatted)} items")
  return formatted, summary
