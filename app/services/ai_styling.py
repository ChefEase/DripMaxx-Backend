import io
import json
import re

import replicate
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.schemas.styling import Occasion, StylingRecommendation, StylingResponse, WeatherSnapshot


settings = get_settings()


def _clean(value: object) -> str:
  return re.sub(r"\s+", " ", str(value or "")).strip()


async def style_outfit_with_ai(
  image_bytes: bytes,
  occasion: Occasion,
  weather: WeatherSnapshot,
) -> StylingResponse:
  if not settings.replicate_api_token:
    raise HTTPException(status_code=503, detail="AI styling is temporarily unavailable.")

  weather_json = weather.model_dump_json()
  system_prompt = (
    "You are DripMaxx's Style My Outfit assistant. Analyze the uploaded outfit photo and provide practical "
    "ways to adapt the visible look for the supplied occasion and CURRENT weather. This is not an outfit rating. "
    "Never give a number, score, grade, tier, or rating. The photo may be old and may not show something worn today. "
    "Never claim the person is wearing it today and never say 'you should not be wearing this today'. "
    "Separate aesthetic advice from weather/practicality advice. Do not recommend items from a wardrobe or claim to "
    "know what the user owns. Suggest only general additions, swaps, layers, footwear, or accessories. "
    "Ground aesthetic advice in visible details. Use short, clear language. Output JSON only with this shape: "
    '{"outfit_present":true,"summary":"one sentence about the visible look","aesthetic_recommendations":['
    '{"title":"short title","description":"specific visible style improvement"}],"weather_recommendations":['
    '{"title":"short title","description":"practical adaptation tied to current conditions"}],'
    '"occasion_note":"one concise occasion-specific note"}. '
    "Return 2 or 3 aesthetic recommendations and 1 to 3 weather recommendations."
  )
  prompt = (
    f"Occasion: {occasion}. Current local weather JSON: {weather_json}. "
    "Use the framing 'Based on today's weather, here's how I'd adapt this look' for practical advice."
  )

  def _call():
    client = replicate.Client(api_token=settings.replicate_api_token, timeout=75)
    image = io.BytesIO(image_bytes)
    image.name = "outfit.jpg"
    return client.run(
      settings.replicate_vlm_model,
      input={
        "system_prompt": system_prompt,
        "prompt": prompt,
        "image_input": [image],
        "verbosity": "low",
        "reasoning_effort": "low",
        "max_completion_tokens": 900,
      },
    )

  try:
    raw_result = await run_in_threadpool(_call)
    if isinstance(raw_result, (list, tuple)):
      raw = "".join(str(part) for part in raw_result)
    elif hasattr(raw_result, "__iter__") and not isinstance(raw_result, (str, bytes)):
      raw = "".join(str(part) for part in raw_result)
    else:
      raw = str(raw_result)
    if "{" in raw and "}" in raw:
      raw = raw[raw.find("{"):raw.rfind("}") + 1]
    data = json.loads(raw.replace("\\_", "_"))
  except Exception as exc:
    raise HTTPException(status_code=502, detail="AI styling could not analyze this photo. Please try again.") from exc

  if data.get("outfit_present") is not True:
    raise HTTPException(status_code=422, detail="No clear outfit was detected. Use a clear full-body outfit photo.")

  def recommendations(key: str, minimum: int) -> list[StylingRecommendation]:
    items = []
    for item in data.get(key) or []:
      if not isinstance(item, dict):
        continue
      title = _clean(item.get("title"))
      description = _clean(item.get("description"))
      if title and description:
        items.append(StylingRecommendation(title=title, description=description))
    if len(items) < minimum:
      raise HTTPException(status_code=502, detail="AI styling returned incomplete advice. Please try again.")
    return items[:3]

  summary = _clean(data.get("summary"))
  occasion_note = _clean(data.get("occasion_note"))
  aesthetic = recommendations("aesthetic_recommendations", 2)
  practical = recommendations("weather_recommendations", 1)
  all_copy = " ".join([
    summary,
    occasion_note,
    *(f"{item.title} {item.description}" for item in [*aesthetic, *practical]),
  ]).lower()
  if not summary or not occasion_note or any(term in all_copy for term in ("drip score", "out of 10", "rating:", "grade:")):
    raise HTTPException(status_code=502, detail="AI styling returned an invalid rating-style response. Please try again.")

  return StylingResponse(
    occasion=occasion,
    weather=weather,
    summary=summary,
    aesthetic_recommendations=aesthetic,
    weather_recommendations=practical,
    occasion_note=occasion_note,
  )
