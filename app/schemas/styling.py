from typing import List, Literal

from pydantic import BaseModel, Field


Occasion = Literal["casual", "school", "work", "date_night_out", "party", "other"]


class WeatherSnapshot(BaseModel):
  temperature_c: float
  apparent_temperature_c: float
  precipitation_mm: float = 0
  rain_mm: float = 0
  snowfall_cm: float = 0
  weather_code: int
  condition: str
  is_day: bool


class StylingRecommendation(BaseModel):
  title: str
  description: str


class StylingResponse(BaseModel):
  occasion: Occasion
  weather: WeatherSnapshot
  framing: str = "Based on today's weather, here's how I'd adapt this look."
  summary: str
  aesthetic_recommendations: List[StylingRecommendation] = Field(default_factory=list)
  weather_recommendations: List[StylingRecommendation] = Field(default_factory=list)
  occasion_note: str
