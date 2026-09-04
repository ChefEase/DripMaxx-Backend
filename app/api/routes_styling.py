import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image as PILImage, UnidentifiedImageError
from pydantic import ValidationError

from app.core.auth import AuthContext, require_auth
from app.schemas.styling import Occasion, StylingResponse, WeatherSnapshot
from app.services.ai_styling import style_outfit_with_ai
from app.services.weather import get_current_weather


router = APIRouter(prefix="/v1/styling", tags=["styling"])


@router.get("/weather", response_model=WeatherSnapshot)
async def current_weather(
  latitude: float,
  longitude: float,
  _auth: AuthContext = Depends(require_auth),
):
  return await get_current_weather(latitude, longitude)


@router.post("/advice", response_model=StylingResponse)
async def style_outfit(
  image: UploadFile = File(...),
  occasion: Occasion = Form(...),
  latitude: float = Form(...),
  longitude: float = Form(...),
  weather_json: str | None = Form(None),
  ai_processing_consent: bool = Form(False),
  _auth: AuthContext = Depends(require_auth),
):
  if not ai_processing_consent:
    raise HTTPException(status_code=400, detail="Consent to Replicate AI processing is required.")
  if not image.content_type or not image.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="Upload must be an image.")
  image_bytes = await image.read()
  if not image_bytes:
    raise HTTPException(status_code=400, detail="Image upload is empty.")
  if len(image_bytes) > 12 * 1024 * 1024:
    raise HTTPException(status_code=413, detail="Image is too large.")
  try:
    with PILImage.open(io.BytesIO(image_bytes)) as decoded:
      detected_format = str(decoded.format or "").upper()
      if decoded.width * decoded.height > 40_000_000:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")
      decoded.verify()
  except (UnidentifiedImageError, OSError, ValueError) as exc:
    raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
  if detected_format not in {"JPEG", "PNG", "WEBP"}:
    raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP outfit images are supported.")
  if weather_json:
    try:
      weather = WeatherSnapshot.model_validate_json(weather_json)
    except ValidationError as exc:
      raise HTTPException(status_code=422, detail="Invalid weather snapshot.") from exc
  else:
    weather = await get_current_weather(latitude, longitude)
  return await style_outfit_with_ai(image_bytes, occasion, weather)
