import threading
import time

import requests
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.schemas.styling import WeatherSnapshot


USER_AGENT = "DripMaxx/1.0 weather-support@dripmaxx.com"
CACHE_TTL_SECONDS = 15 * 60
_weather_cache: dict[tuple[float, float], tuple[float, WeatherSnapshot]] = {}
_cache_lock = threading.Lock()


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


def _cache_key(latitude: float, longitude: float) -> tuple[float, float]:
  # Nearby users receive equivalent current conditions and share one lookup.
  return round(latitude, 2), round(longitude, 2)


def _cached_weather(key: tuple[float, float]) -> WeatherSnapshot | None:
  with _cache_lock:
    entry = _weather_cache.get(key)
    if not entry:
      return None
    stored_at, snapshot = entry
    if time.monotonic() - stored_at >= CACHE_TTL_SECONDS:
      _weather_cache.pop(key, None)
      return None
    return snapshot


def _store_weather(key: tuple[float, float], snapshot: WeatherSnapshot) -> None:
  with _cache_lock:
    _weather_cache[key] = (time.monotonic(), snapshot)


def _open_meteo_snapshot(payload: dict) -> WeatherSnapshot:
  current = payload.get("current") or {}
  code = int(current["weather_code"])
  temperature = round(float(current["temperature_2m"]), 1)
  return WeatherSnapshot(
    temperature_c=temperature,
    apparent_temperature_c=round(float(current.get("apparent_temperature", temperature)), 1),
    precipitation_mm=max(0, float(current.get("precipitation") or 0)),
    rain_mm=max(0, float(current.get("rain") or 0)),
    snowfall_cm=max(0, float(current.get("snowfall") or 0)),
    weather_code=code,
    condition=weather_condition(code),
    is_day=bool(current.get("is_day", 1)),
  )


def _met_condition(symbol: str) -> tuple[int, str]:
  symbol = symbol.lower()
  if "thunder" in symbol:
    return 95, "Thunderstorms"
  if "snow" in symbol or "sleet" in symbol:
    return 73, "Snow"
  if "rain" in symbol:
    return 63, "Rain"
  if "fog" in symbol:
    return 45, "Foggy"
  if "partlycloudy" in symbol:
    return 2, "Partly cloudy"
  if "cloudy" in symbol:
    return 3, "Overcast"
  if "clearsky" in symbol or "fair" in symbol:
    return 0, "Clear"
  return 999, "Mixed conditions"


def _met_snapshot(payload: dict) -> WeatherSnapshot:
  point = payload["properties"]["timeseries"][0]["data"]
  details = point["instant"]["details"]
  forecast = point.get("next_1_hours") or point.get("next_6_hours") or point.get("next_12_hours") or {}
  symbol = (forecast.get("summary") or {}).get("symbol_code", "")
  precipitation = max(0, float((forecast.get("details") or {}).get("precipitation_amount", 0)))
  temperature = round(float(details["air_temperature"]), 1)
  code, condition = _met_condition(symbol)
  return WeatherSnapshot(
    temperature_c=temperature,
    apparent_temperature_c=temperature,
    precipitation_mm=precipitation,
    rain_mm=precipitation if condition in ("Rain", "Thunderstorms") else 0,
    # MET reports liquid-equivalent precipitation, not snow depth.
    snowfall_cm=0,
    weather_code=code,
    condition=condition,
    is_day=not symbol.endswith("_night"),
  )


def _fetch_weather(latitude: float, longitude: float) -> WeatherSnapshot:
  errors: list[str] = []
  variable_sets = [
    "temperature_2m,apparent_temperature,precipitation,rain,snowfall,weather_code,is_day",
    "temperature_2m,apparent_temperature,weather_code,is_day",
  ]
  with requests.Session() as session:
    session.headers.update({"User-Agent": USER_AGENT})
    for variables in variable_sets:
      try:
        response = session.get(
          "https://api.open-meteo.com/v1/forecast",
          params={
            "latitude": latitude,
            "longitude": longitude,
            "temperature_unit": "celsius",
            "timezone": "auto",
            "current": variables,
          },
          timeout=(5, 15),
        )
        # Immediate retries make provider throttling worse. Move directly to
        # the independent fallback when this Render IP is rate-limited.
        if response.status_code == 429:
          errors.append(f"open-meteo status=429 fields={variables}")
          break
        response.raise_for_status()
        return _open_meteo_snapshot(response.json())
      except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
        errors.append(f"open-meteo fields={variables} error={exc}")

    try:
      response = session.get(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params={"lat": round(latitude, 4), "lon": round(longitude, 4)},
        timeout=(5, 15),
      )
      response.raise_for_status()
      return _met_snapshot(response.json())
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as exc:
      errors.append(f"met-norway error={exc}")

  raise RuntimeError(" | ".join(errors))


async def get_current_weather(latitude: float, longitude: float) -> WeatherSnapshot:
  if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
    raise HTTPException(status_code=422, detail="Invalid location coordinates.")

  key = _cache_key(latitude, longitude)
  cached = _cached_weather(key)
  if cached:
    return cached

  try:
    snapshot = await run_in_threadpool(_fetch_weather, latitude, longitude)
    _store_weather(key, snapshot)
    return snapshot
  except (RuntimeError, requests.RequestException) as exc:
    logger.error("weather_provider_failed latitude={} longitude={} error={}", latitude, longitude, exc)
    raise HTTPException(status_code=503, detail="Current weather is temporarily unavailable.") from exc
