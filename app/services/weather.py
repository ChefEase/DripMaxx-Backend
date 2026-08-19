import requests
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.schemas.styling import WeatherSnapshot


def weather_condition(code: int) -> str:
  if code == 0:
    return "Clear"
  if code in (1, 2):
    return "Partly cloudy"
  if code == 3:
    return "Overcast"
  if code in (45, 48):
    return "Foggy"
  if code in (51, 53, 55, 56, 57):
    return "Drizzle"
  if code in (61, 63, 65, 66, 67, 80, 81, 82):
    return "Rain"
  if code in (71, 73, 75, 77, 85, 86):
    return "Snow"
  if code in (95, 96, 99):
    return "Thunderstorms"
  return "Mixed conditions"


async def get_current_weather(latitude: float, longitude: float) -> WeatherSnapshot:
  if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
    raise HTTPException(status_code=422, detail="Invalid location coordinates.")

  base_params = {
    "latitude": latitude,
    "longitude": longitude,
    "temperature_unit": "celsius",
    "timezone": "auto",
  }

  def _fetch():
    errors = []
    variable_sets = [
      "temperature_2m,apparent_temperature,precipitation,rain,snowfall,weather_code,is_day",
      # A reduced request still provides useful styling context if a weather
      # model temporarily rejects an optional precipitation variable.
      "temperature_2m,apparent_temperature,weather_code,is_day",
    ]
    with requests.Session() as session:
      session.headers.update({"User-Agent": "DripMaxx/1.0 weather-support@dripmaxx.com"})
      for variables in variable_sets:
        for attempt in range(2):
          try:
            response = session.get(
              "https://api.open-meteo.com/v1/forecast",
              params={**base_params, "current": variables},
              timeout=(5, 15),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("current"), dict):
              raise ValueError("provider response missing current conditions")
            return payload
          except (ValueError, requests.RequestException) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            errors.append(f"fields={variables} attempt={attempt + 1} status={status_code} error={exc}")
    raise RuntimeError(" | ".join(errors))

  try:
    payload = await run_in_threadpool(_fetch)
    current = payload.get("current") or {}
    code = int(current["weather_code"])
    return WeatherSnapshot(
      temperature_c=round(float(current["temperature_2m"]), 1),
      apparent_temperature_c=round(float(current["apparent_temperature"]), 1),
      precipitation_mm=max(0, float(current.get("precipitation") or 0)),
      rain_mm=max(0, float(current.get("rain") or 0)),
      snowfall_cm=max(0, float(current.get("snowfall") or 0)),
      weather_code=code,
      condition=weather_condition(code),
      is_day=bool(current.get("is_day", 1)),
    )
  except (KeyError, TypeError, ValueError, RuntimeError, requests.RequestException) as exc:
    logger.error("weather_provider_failed latitude={} longitude={} error={}", latitude, longitude, exc)
    raise HTTPException(status_code=503, detail="Current weather is temporarily unavailable.") from exc
