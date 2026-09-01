from datetime import datetime, timedelta, timezone

import pytest

from app.api.routes_challenges import _results_visible, _validate_admin_placements


def test_one_submission_only_requires_first_place():
  assert _validate_admin_placements("a", None, None, {"a"}) == ["a"]


def test_two_submissions_require_first_and_second_only():
  assert _validate_admin_placements("a", "b", None, {"a", "b"}) == ["a", "b"]
  with pytest.raises(ValueError):
    _validate_admin_placements("a", None, None, {"a", "b"})


def test_three_or_more_submissions_require_unique_top_three():
  assert _validate_admin_placements("a", "b", "c", {"a", "b", "c", "d"}) == ["a", "b", "c"]
  with pytest.raises(ValueError):
    _validate_admin_placements("a", "a", "c", {"a", "b", "c"})


def test_results_expire_exactly_after_24_hours():
  now = datetime.now(timezone.utc)
  assert _results_visible(now - timedelta(hours=23, minutes=59), now)
  assert not _results_visible(now - timedelta(hours=24), now)
