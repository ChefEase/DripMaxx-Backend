import json
import io
import time
from typing import List

import replicate
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.core.config import get_settings
from app.schemas.outfits import ScoreBreakdown, SuggestionCard, UserContext

settings = get_settings()


def _parse_suggestions(raw: str) -> List[SuggestionCard]:
  text = raw.strip()
  if "[" in text and "]" in text:
    text = text[text.index("[") : text.rindex("]") + 1]
  text = text.replace("\\_", "_").strip()
  try:
    data = json.loads(text)
  except Exception:
    data = None
    import re
    # Fallback: extract repeating title/type/description triples even if braces are broken
    titles = re.findall(r'"title"\s*:\s*"([^"]+)"', text)
    types = re.findall(r'"type"\s*:\s*"([^"]+)"', text)
    descs = re.findall(r'"description"\s*:\s*"([^"]+)"', text)
    items = []
    for idx, title in enumerate(titles):
      desc = descs[idx] if idx < len(descs) else ""
      typ = types[idx] if idx < len(types) else "other"
      items.append({"title": title, "type": typ, "description": desc})
    if items:
      data = items

  cards: List[SuggestionCard] = []
  if isinstance(data, list):
    for item in data[:15]:
      if not isinstance(item, dict):
        continue
      title = (item.get("title") or "").strip()
      desc = (item.get("description") or item.get("desc") or item.get("suggestion") or "").strip()
      type_tag = (item.get("type") or item.get("category") or "other").strip()
      if title and desc:
        cards.append(SuggestionCard(title=title, type=type_tag, description=desc))
  return cards


async def generate_suggestions(
  breakdown: ScoreBreakdown, user_ctx: UserContext, image_bytes: bytes, image_url: str | None = None
) -> List[SuggestionCard]:
  """Use a vision-language model (VLM) to generate grounded suggestions from the actual image."""
  if not settings.replicate_api_token:
    raise HTTPException(
      status_code=503,
      detail="Replicate API token missing for suggestions.",
    )

  sys_prompt = (
    "You are a fashion assistant. Look at the image and provide 15 actionable outfit improvement suggestions. "
    "Return ONLY a valid JSON array with up to 15 objects. "
    "Each object must be: {\"title\": \"<=8 words\", \"type\": \"fit|layering|color|accessory|other\", "
    "\"description\": \"<=25 words\", and it must be a specific suggestion, not a description. "
    "All items must be unique and non-redundant. Do not repeat the same idea or wording. "
    "Use varied verbs and phrasing; avoid repeating sentence patterns. "
    "Be creative within what is visible; suggest distinct angles (fit, proportions, texture, contrast, silhouette, accessories). "
    "Do NOT use score category names as titles (no 'Color Match', 'Fit Quality', 'Trend Score', etc). "
    "Do NOT use types outside the allowed list. "
    "Ground every tip in what is visible; do NOT invent garments. "
    "Do NOT output captions, summaries, or image descriptions. Suggestions only. "
    "No extra text, no markdown, no trailing commas."
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
    "Output JSON array only."
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
            "top_p": 0.95,
            "max_tokens": 1200,
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
  raw = await run_in_threadpool(lambda: _call_vlm(base_prompt, 0.8))
  cards = _parse_suggestions(raw)
  # post-filter: normalize types, cap 15 (do not dedupe to avoid shrinking output)
  normalized = []
  allowed_types = {"fit", "layering", "color", "accessory", "other"}
  for c in cards:
    c.type = c.type.lower()
    if c.type not in allowed_types:
      c.type = "other"
    normalized.append(c)
  cards = normalized[:15]

  # Reject repeated ideas (title + description) to enforce uniqueness.
  seen = set()
  unique_cards = []
  for c in cards:
    key = (c.title.strip().lower(), c.description.strip().lower())
    if key in seen:
      continue
    seen.add(key)
    unique_cards.append(c)
  cards = unique_cards

  if not cards:
    logger.error(f"VLM suggestion parse failed; raw='{raw[:800]}'")
    raise HTTPException(
      status_code=502,
      detail="LLM suggestions unavailable; please retry shortly.",
    )

  logger.debug(f"VLM suggestions parsed {len(cards)} items")
  return cards
