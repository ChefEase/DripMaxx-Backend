"""Upload outfit images to Supabase Storage."""

import base64
import json
import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

def upload_outfit_image(
  image_bytes: bytes,
  outfit_id: str,
  user_id: Optional[str] = None,
  content_type: str = "image/jpeg",
) -> Optional[str]:
  """
  Upload outfit image to Supabase Storage. Returns public URL or None on failure.
  Bucket must exist and be public. Create in Supabase: Storage > New bucket > "outfits" (public).
  """
  settings = get_settings()
  if not settings.supabase_url or not settings.supabase_service_key:
    logger.warning("Supabase storage not configured: missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
    return None

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

    bucket = settings.supabase_bucket or "outfits"
    folder = user_id or "anonymous"
    path = f"{folder}/{outfit_id}.jpg"

    upload_resp = client.storage.from_(bucket).upload(
      path=path,
      file=image_bytes,
      # supabase-py storage upload options
      file_options={"content-type": content_type, "upsert": "true"},
    )
    if isinstance(upload_resp, dict) and upload_resp.get("error"):
      raise RuntimeError(f"Supabase upload error: {upload_resp['error']}")

    public_url_resp = client.storage.from_(bucket).get_public_url(path)
    if isinstance(public_url_resp, dict):
      url = (
        public_url_resp.get("publicUrl")
        or public_url_resp.get("publicURL")
        or public_url_resp.get("signedURL")
      )
    else:
      url = str(public_url_resp)
    if not url:
      raise RuntimeError("Supabase returned empty public URL")

    logger.info("upload_outfit_image bucket=%s path=%s url=%s", bucket, path, url)
    return url
  except Exception as e:
    logger.exception("upload_outfit_image failed: %s", e)
    return None
