from app.schemas.outfits import ScoreBreakdown
from app.services.ai_scoring import (
  _eval_color_score,
  _overall_from_available_metrics,
  _weighted_subscore,
)


def _breakdown(body=9.9, style=9.9):
  return ScoreBreakdown(
    color_match=8.0,
    fit_quality=7.0,
    body_compatibility=body,
    trend_score=6.0,
    style_match=style,
  )


def test_missing_profile_metrics_do_not_change_overall():
  low_missing = _overall_from_available_metrics(
    _breakdown(body=0.0, style=0.0), has_style_target=False, has_body_profile=False
  )
  high_missing = _overall_from_available_metrics(
    _breakdown(body=10.0, style=10.0), has_style_target=False, has_body_profile=False
  )
  assert low_missing == high_missing == 7.2


def test_body_is_excluded_when_only_style_was_selected():
  low_body = _overall_from_available_metrics(
    _breakdown(body=0.0, style=8.0), has_style_target=True, has_body_profile=False
  )
  high_body = _overall_from_available_metrics(
    _breakdown(body=10.0, style=8.0), has_style_target=True, has_body_profile=False
  )
  assert low_body == high_body


def test_style_is_excluded_when_only_body_profile_exists():
  low_style = _overall_from_available_metrics(
    _breakdown(body=8.0, style=0.0), has_style_target=False, has_body_profile=True
  )
  high_style = _overall_from_available_metrics(
    _breakdown(body=8.0, style=10.0), has_style_target=False, has_body_profile=True
  )
  assert low_style == high_style


def test_granular_subscores_preserve_score_differences():
  weights = {"harmony": 0.5, "intent": 0.5}
  first = _weighted_subscore(
    {"harmony": 7.1, "intent": 6.7}, weights, "good", {"good": 8.0}, 6.0
  )
  second = _weighted_subscore(
    {"harmony": 8.6, "intent": 8.2}, weights, "good", {"good": 8.0}, 6.0
  )
  assert first == 6.9
  assert second == 8.4
  assert second - first == 1.5


def test_neutral_palette_does_not_receive_an_automatic_high_floor():
  assert _eval_color_score(["brown", "grey", "black"], {}, 6.4) == 6.4


def test_old_category_labels_remain_backward_compatible():
  score = _weighted_subscore({}, {"harmony": 1.0}, "good", {"good": 8.2}, 6.0)
  assert score == 8.2
