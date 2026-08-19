import requests
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

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

  def _fetch():
    return requests.get(
      "https://api.open-meteo.com/v1/forecast",
      params={
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,apparent_temperature,precipitation,rain,snowfall,weather_code,is_day",
        "temperature_unit": "celsius",
        "timezone": "auto",
      },
      timeout=12,
    )

  try:
    response = await run_in_threadpool(_fetch)
    response.raise_for_status()
    current = response.json().get("current") or {}
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
  except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
    raise HTTPException(status_code=503, detail="Current weather is temporarily unavailable.") from exc
