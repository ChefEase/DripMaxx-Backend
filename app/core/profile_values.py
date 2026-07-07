from __future__ import annotations

from app.models.entities import BodyTypeEnum, GenderStyleEnum


def _normalize_key(value: str) -> str:
  return value.strip().lower().replace("-", "_").replace(" ", "_")


def parse_body_type(value: str | BodyTypeEnum | None) -> BodyTypeEnum | None:
  if value is None:
    return None
  if isinstance(value, BodyTypeEnum):
    return value
  raw = str(value).strip()
  if not raw or raw.lower() in {"n/a", "na", "none", "null"}:
    return None
  key = _normalize_key(raw)
  try:
    return BodyTypeEnum(key)
  except ValueError:
    return None


def parse_gender_style(value: str | GenderStyleEnum | None) -> GenderStyleEnum | None:
  if value is None:
    return None
  if isinstance(value, GenderStyleEnum):
    return value
  raw = str(value).strip()
  if not raw or raw.lower() in {"n/a", "na", "none", "null"}:
    return None
  key = _normalize_key(raw)
  try:
    return GenderStyleEnum(key)
  except ValueError:
    return None
