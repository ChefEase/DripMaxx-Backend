from app.services.weather import _met_snapshot, weather_condition


def test_weather_condition_groups_rain_and_snow():
  assert weather_condition(0) == "Clear"
  assert weather_condition(63) == "Rain"
  assert weather_condition(81) == "Rain"
  assert weather_condition(73) == "Snow"
  assert weather_condition(95) == "Thunderstorms"


def test_weather_condition_has_safe_unknown_fallback():
  assert weather_condition(999) == "Mixed conditions"


def test_met_snapshot_maps_current_conditions():
  snapshot = _met_snapshot({
    "properties": {
      "timeseries": [{
        "data": {
          "instant": {"details": {"air_temperature": 12.4}},
          "next_1_hours": {
            "summary": {"symbol_code": "heavyrain_day"},
            "details": {"precipitation_amount": 2.1},
          },
        },
      }],
    },
  })

  assert snapshot.temperature_c == 12.4
  assert snapshot.condition == "Rain"
  assert snapshot.rain_mm == 2.1
  assert snapshot.is_day is True
