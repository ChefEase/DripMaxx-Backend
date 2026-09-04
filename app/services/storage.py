"""Upload outfit images to Supabase Storage."""

import base64
import json
import logging
import time
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

from app.core.config import get_settings

logger = logging.getLogger(__name__)
CLIENT_SIGNED_URL_TTL_SECONDS = 300
AI_SIGNED_URL_TTL_SECONDS = 300


def storage_path(reference: str | None, bucket: str | None = None) -> str | None:
  """Return an object path from a new path or a legacy public/signed URL."""
  if not reference or reference.startswith("uploaded://"):
    return None
  if not reference.startswith(("http://", "https://")):
    return reference.lstrip("/")
  bucket = bucket or get_settings().supabase_bucket or "outfits"
  parsed = urlparse(reference)
  for marker in (
    f"/storage/v1/object/public/{bucket}/",
    f"/storage/v1/object/sign/{bucket}/",
    f"/storage/v1/object/authenticated/{bucket}/",
  ):
    if marker in parsed.path:
      return unquote(parsed.path.split(marker, 1)[1])
  return None


def create_signed_image_url(reference: str | None, expires_in: int = CLIENT_SIGNED_URL_TTL_SECONDS) -> str | None:
  """Grant temporary read access to one private object; never persist or log this URL."""
  path = storage_path(reference)
  settings = get_settings()
  if not path or not settings.supabase_url or not settings.supabase_service_key:
    return None
  from supabase import create_client
  client = create_client(settings.supabase_url, settings.supabase_service_key)
  response = client.storage.from_(settings.supabase_bucket or "outfits").create_signed_url(
    path, max(30, min(int(expires_in), 600))
  )
  if isinstance(response, dict):
    return response.get("signedURL") or response.get("signedUrl") or response.get("signed_url")
  return getattr(response, "signed_url", None) or getattr(response, "signedURL", None)


def delete_storage_objects(references: Iterable[str | None]) -> None:
  """Delete originals, revisions, targets, and legacy URL-backed objects."""
  settings = get_settings()
  paths = list(dict.fromkeys(path for path in (storage_path(ref) for ref in references) if path))
  if not paths or not settings.supabase_url or not settings.supabase_service_key:
    return
  from supabase import create_client
  client = create_client(settings.supabase_url, settings.supabase_service_key)
  bucket = settings.supabase_bucket or "outfits"
  for index in range(0, len(paths), 100):
    batch = paths[index:index + 100]
    for attempt in range(3):
      try:
        client.storage.from_(bucket).remove(batch)
        break
      except Exception:
        if attempt == 2:
          logger.exception("private storage deletion failed after retries batch_size=%s", len(batch))
          raise
        time.sleep(0.25 * (attempt + 1))

def upload_outfit_image(
  image_bytes: bytes,
  outfit_id: str,
  user_id: Optional[str] = None,
  content_type: str = "image/jpeg",
) -> Optional[str]:
  """
  Upload to the private outfit bucket and return only the object path.
  """
  settings = get_settings()
  if not settings.supabase_url or not settings.supabase_service_key:
    logger.warning("Supabase storage not configured: missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
    return None
  bucket = settings.supabase_bucket or "outfits"
  folder = user_id or "anonymous"
  extension = {"image/png": "png", "image/webp": "webp"}.get(content_type.lower(), "jpg")
  path = f"{folder}/{outfit_id}.{extension}"
  client = None

  def _jwt_role(token: str) -> Optional[str]:
    try:
      parts = token.split(".")
      if len(parts) < 2:
        return None
      payload = parts[1] + "==="
      data = json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")))
      return data.get("role")
    except Exception:
      return None

  try:
    from supabase import create_client
    role = _jwt_role(settings.supabase_service_key)
    if role and role != "service_role":
      logger.error("Supabase key role is %s, expected service_role for uploads.", role)
    client = create_client(settings.supabase_url, settings.supabase_service_key)

    upload_resp = client.storage.from_(bucket).upload(
      path=path,
      file=image_bytes,
      # supabase-py storage upload options
      file_options={"content-type": content_type, "upsert": "true", "cache-control": "60"},
    )
    if isinstance(upload_resp, dict) and upload_resp.get("error"):
      raise RuntimeError(f"Supabase upload error: {upload_resp['error']}")

    logger.info("upload_outfit_image bucket=%s path=%s", bucket, path)
    return path
  except Exception as e:
    logger.exception("upload_outfit_image failed: %s", e)
    # The network may fail after Supabase accepted the object. Compensate using
    # the deterministic path so an ambiguous upload cannot become an orphan.
    try:
      if client is not None:
        client.storage.from_(bucket).remove([path])
    except Exception:
      logger.exception("ambiguous outfit upload cleanup failed path=%s", path)
    return None


def upload_target_image(image_bytes: bytes, session_id: str, user_id: str) -> Optional[str]:
  """Persist generated target art because Replicate output URLs expire."""
  settings = get_settings()
  if not settings.supabase_url or not settings.supabase_service_key:
    return None
  bucket = settings.supabase_bucket or "outfits"
  path = f"{user_id}/evolution/{session_id}-target.webp"
  client = None
  try:
    from supabase import create_client
    client = create_client(settings.supabase_url, settings.supabase_service_key)
    client.storage.from_(bucket).upload(
      path=path, file=image_bytes,
      file_options={"content-type": "image/webp", "upsert": "true", "cache-control": "60"},
    )
    return path
  except Exception as exc:
    logger.exception("upload_target_image failed: %s", exc)
    try:
      if client is not None:
        client.storage.from_(bucket).remove([path])
    except Exception:
      logger.exception("ambiguous target upload cleanup failed path=%s", path)
    return None
