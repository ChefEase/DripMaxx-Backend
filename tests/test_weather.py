from app.services.weather import weather_condition


def test_weather_condition_groups_rain_and_snow():
  assert weather_condition(0) == "Clear"
  assert weather_condition(63) == "Rain"
  assert weather_condition(81) == "Rain"
  assert weather_condition(73) == "Snow"
  assert weather_condition(95) == "Thunderstorms"


def test_weather_condition_has_safe_unknown_fallback():
  assert weather_condition(999) == "Mixed conditions"
