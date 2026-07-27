from datetime import datetime, timedelta, timezone

from app.api.routes_profile import _current_streak, _improvement


def test_improvement_compares_smoothed_early_and_recent_scores():
  assert _improvement([5.0, 5.5, 6.0, 7.0, 7.5, 8.0]) == 2.0


def test_improvement_requires_at_least_two_scans():
  assert _improvement([]) == 0.0
  assert _improvement([7.5]) == 0.0


def test_current_streak_counts_consecutive_unique_days():
  now = datetime.now(timezone.utc)
  assert _current_streak([
    now,
    now - timedelta(hours=2),
    now - timedelta(days=1),
    now - timedelta(days=2),
    now - timedelta(days=4),
  ]) == 3


def test_old_activity_is_not_a_current_streak():
  assert _current_streak([datetime.now(timezone.utc) - timedelta(days=3)]) == 0
