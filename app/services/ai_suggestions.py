import json
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


def _looks_placeholder(text: str) -> bool:
  lowered = text.strip().lower()
  banned_phrases = {
    "dynamic title",
    "specific outfit change based on the image",
    "suggestion title",
    "optional suggestion",
    "main suggestion",
  }
  return lowered in banned_phrases or lowered.startswith("dynamic ")


def _is_valid_detection(detection: Dict[str, Any]) -> bool:
  if not isinstance(detection, dict):
    return False
  summary = detection.get("summary")
  main_suggestions = detection.get("main_suggestions")
  optional_suggestions = detection.get("optional_suggestions")
  if not isinstance(summary, str) or not summary.strip():
    return False
  if not isinstance(main_suggestions, list) or not isinstance(optional_suggestions, list):
    return False
  if len(main_suggestions) != 3 or len(optional_suggestions) != 2:
    return False
  color_count = 0
  seen_texts = []
  for item in [*main_suggestions, *optional_suggestions]:
    if not isinstance(item, dict):
      return False
    title = item.get("title")
    type_name = item.get("type")
    description = item.get("description")
    if not (isinstance(title, str) and title.strip() and isinstance(type_name, str) and type_name.strip() and isinstance(description, str) and description.strip()):
      return False
    if _looks_placeholder(title) or _looks_placeholder(description):
      return False
    if type_name.strip().lower() == "color":
      color_count += 1
    text = f"{title} {description}"
    if any(_too_similar(text, existing) for existing in seen_texts):
      return False
    seen_texts.append(text)
  if color_count > 2:
    return False
  return True


async def generate_suggestions(
  breakdown: ScoreBreakdown,
  user_ctx: UserContext,
  image_bytes: bytes,
  image_url: str | None = None,
  visual_context: Dict[str, Any] | None = None,
) -> Tuple[List[SuggestionCard], str | None]:
  """Generate dynamic suggestion cards and summary from GPT-5.4 using grounded visual context."""
  if not settings.replicate_api_token:
    raise HTTPException(
      status_code=503,
      detail="Replicate API token missing for suggestions.",
    )

  sys_prompt = (
    "Look at the image and output ONLY JSON with dynamic suggestion cards and a summary:\n"
    "{"
    "\"summary\": \"two short sentences: one on what works, one on what to improve\","
    "\"main_suggestions\": ["
    "{\"title\": \"Sharpen the boot contrast\", \"type\": \"color\", \"description\": \"Keep the dark boots, but break the blue stack with a lighter or textured mid-layer so the outfit reads intentional instead of flat.\"},"
    "{\"title\": \"Clean up the silhouette\", \"type\": \"fit\", \"description\": \"Tighten the line at either the top or bottom so the shape feels more deliberate and less uniform.\"},"
    "{\"title\": \"Add one focal detail\", \"type\": \"accessory\", \"description\": \"Introduce one visible detail such as a watch, chain, or structured outer layer so the outfit has a point of interest.\"}"
    "],"
    "\"optional_suggestions\": ["
    "{\"title\": \"Break up the palette\", \"type\": \"color\", \"description\": \"Swap one blue piece for charcoal, grey, or off-white so the outfit keeps its clean mood without looking one-note.\"},"
    "{\"title\": \"Refine the finish\", \"type\": \"other\", \"description\": \"Choose one cleaner fabric or sharper hem so the look feels more premium overall.\"}"
    "]"
    "}\n"
    "Rules:\n"
    "- Base every summary and suggestion on the provided visual context plus the score breakdown.\n"
    "- The summary must mention at least one strength and one weakness.\n"
    "- Do not use placeholder words like 'dynamic title' or schema wording.\n"
    "- Do not use canned wording, templates, or generic fallback advice.\n"
    "- Suggestions must be concrete and visually grounded in the provided outfit details.\n"
    "- Return exactly 3 main_suggestions and exactly 2 optional_suggestions.\n"
    "- Keep titles short, natural, and editorial. Do not repeat the same noun across multiple suggestions unless absolutely necessary.\n"
    "- Avoid giving all five suggestions about the same issue. Spread them across color, fit, accessory, layering, or finish when possible.\n"
    "- If color is the weakest area, do not make more than two suggestions purely about color.\n"
    "- Use only the allowed types: fit, layering, color, accessory, other.\n"
    "- Do not mention items that are not visible in the visual context.\n"
    "Output only JSON."
  )
  visual_context_json = json.dumps(visual_context or {}, ensure_ascii=True, separators=(",", ":"))
  user_prompt = (
    f"User style prefs: {', '.join(user_ctx.style_preferences) or 'unspecified'}; "
    f"inspirations: {', '.join(user_ctx.style_inspirations) or 'unspecified'}; "
    f"height: {user_ctx.user_height or 'n/a'}; body_type: {user_ctx.user_body_type or 'n/a'}; "
    f"gender_style: {user_ctx.gender_style_preference or 'n/a'}. "
    f"Focus on weakest scores first: "
    f"color_match={breakdown.color_match}, fit_quality={breakdown.fit_quality}, "
    f"body_compatibility={breakdown.body_compatibility}, trend_score={breakdown.trend_score}, "
    f"style_match={breakdown.style_match}. "
    f"Visual context JSON: {visual_context_json}. "
    "Your response must be fully dynamic for this outfit and must not reuse stock phrases. "
    "If the outfit already has a strong base, acknowledge that and suggest refinements instead of acting like the whole look is bad. "
    "Output JSON only."
  )

  def _call_vlm(prompt: str, temperature: float):
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=60)
    model_ref = settings.replicate_llm_model
    tries = 0
    while True:
      tries += 1
      try:
        result = client.run(
          model_ref,
          input={
            "prompt": prompt,
            "reasoning_effort": "medium",
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
      "\"summary\": \"two short sentences: one on what works, one on what to improve\","
      "\"main_suggestions\": ["
      "{\"title\": \"natural short title\", \"type\": \"fit|layering|color|accessory|other\", \"description\": \"specific outfit change based on the image\"},"
      "{\"title\": \"natural short title\", \"type\": \"fit|layering|color|accessory|other\", \"description\": \"specific outfit change based on the image\"},"
      "{\"title\": \"natural short title\", \"type\": \"fit|layering|color|accessory|other\", \"description\": \"specific outfit change based on the image\"}"
      "],"
      "\"optional_suggestions\": ["
      "{\"title\": \"natural short title\", \"type\": \"fit|layering|color|accessory|other\", \"description\": \"specific outfit change based on the image\"},"
      "{\"title\": \"natural short title\", \"type\": \"fit|layering|color|accessory|other\", \"description\": \"specific outfit change based on the image\"}"
      "]"
      "}\n"
      "Rules:\n"
      "- summary must be non-empty.\n"
      "- main_suggestions must contain exactly 3 valid suggestion objects.\n"
      "- optional_suggestions must contain exactly 2 valid suggestion objects.\n"
      "- reject placeholder titles or repeated suggestions.\n"
      "- Output only JSON.\n\n"
      f"RAW:\n{raw}"
    )
    raw = await run_in_threadpool(lambda: _call_vlm(repair_prompt, 0.0))
    detection = _parse_detection(raw)
  summary = re.sub(r"\s+", " ", str(detection.get("summary") or "")).strip()
  suggestions_raw = [
    *(detection.get("main_suggestions") or []),
    *(detection.get("optional_suggestions") or []),
  ]
  final = []
  for item in suggestions_raw:
    if not isinstance(item, dict):
      continue
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
    type_name = re.sub(r"\s+", " ", str(item.get("type") or "other")).strip().lower()
    desc = re.sub(r"\s+", " ", str(item.get("description") or "")).strip()
    if not title or not desc:
      continue
    if type_name not in {"fit", "layering", "color", "accessory", "other"}:
      type_name = "other"
    text = f"{title} {desc}"
    if any(_too_similar(text, f"{c.title} {c.description}") for c in final):
      continue
    final.append(SuggestionCard(title=title, type=type_name, description=desc))
    if len(final) >= 5:
      break

  if not final:
    logger.error(f"VLM suggestion parse failed; raw='{raw[:800]}'")
    raise HTTPException(
      status_code=502,
      detail="LLM suggestions unavailable; please retry shortly.",
    )
  logger.debug(f"LLM suggestions dynamic {len(final)} items")
  return final, summary or None
