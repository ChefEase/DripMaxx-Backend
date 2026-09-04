from types import SimpleNamespace

from app.schemas.outfits import SuggestionCard
from app.core.config import Settings
from app.services.outfit_evolution import calculate_potential_score, calculate_revision_score


def _suggestion(importance: str, impact: float) -> SuggestionCard:
  return SuggestionCard(title="Change", type="fit", description="Change it", importance=importance, impact=impact)


def _recommendation(rec_id: str, impact: float):
  return SimpleNamespace(id=rec_id, impact=impact)


def test_potential_uses_weighted_opportunities_without_promising_ten():
  suggestions = [_suggestion("high", 0.55), _suggestion("medium", 0.35), _suggestion("low", 0.2)]
  assert calculate_potential_score(6.2, suggestions) == 7.3
  assert calculate_potential_score(9.4, suggestions) == 9.8


def test_verified_improvements_raise_revision_score():
  recommendations = [_recommendation("fit", 0.6), _recommendation("shoes", 0.3)]
  score = calculate_revision_score(
    6.2, 6.2, 7.1, 5.8, recommendations,
    [
      {"id": "fit", "status": "completed", "confidence": 0.95},
      {"id": "shoes", "status": "partial", "confidence": 0.8},
    ],
    0, 0.9,
  )
  assert score > 6.2
  assert score <= 7.1


def test_unchanged_revision_stays_stable_despite_objective_photo_noise():
  recommendations = [_recommendation("fit", 0.6)]
  score = calculate_revision_score(
    6.2, 6.2, 6.8, 5.1, recommendations,
    [{"id": "fit", "status": "remaining", "confidence": 0.9}],
    0, 0.9,
  )
  assert 6.0 <= score <= 6.4


def test_real_regression_can_lower_score():
  recommendations = [_recommendation("fit", 0.6)]
  score = calculate_revision_score(
    7.2, 7.5, 8.0, 6.3, recommendations,
    [{"id": "fit", "status": "regressed", "confidence": 0.95}],
    0.3, 0.9,
  )
  assert score < 7.5


def test_major_off_target_change_lowers_score_without_crushing_it():
  recommendations = [_recommendation("pants", 0.5), _recommendation("shoes", 0.3)]
  score = calculate_revision_score(
    7.0, 7.4, 8.1, 5.8, recommendations,
    [
      {"id": "pants", "status": "regressed", "confidence": 0.95},
      {"id": "shoes", "status": "remaining", "confidence": 0.9},
    ],
    1.0, 0.92,
  )
  assert 5.5 <= score < 7.4


def test_default_models_use_terra_and_gpt_image_2():
  assert Settings.model_fields["replicate_llm_model"].default == "openai/gpt-5.6-terra"
  assert Settings.model_fields["replicate_vlm_model"].default == "openai/gpt-5.6-terra"
  assert Settings.model_fields["replicate_image_model"].default == "openai/gpt-image-2"
