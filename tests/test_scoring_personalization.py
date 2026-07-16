from app.schemas.outfits import ScoreBreakdown
from app.services.ai_scoring import _overall_from_available_metrics


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
