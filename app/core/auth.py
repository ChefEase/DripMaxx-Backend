from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests
from fastapi import Depends, Header, HTTPException, status
from jose import jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import get_settings
from app.db.session import get_db
from app.models import User


settings = get_settings()


@dataclass
class AuthContext:
  auth_user_id: str
  app_user_id: str
  email: str | None = None


def _jwks_url() -> str:
  if not settings.supabase_url:
    raise HTTPException(status_code=500, detail="Supabase auth is not configured")
  return f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


def _user_verify_url() -> str:
  if not settings.supabase_url:
    raise HTTPException(status_code=500, detail="Supabase auth is not configured")
  return f"{settings.supabase_url.rstrip('/')}/auth/v1/user"


@lru_cache(maxsize=1)
def _fetch_jwks() -> dict[str, Any]:
  resp = requests.get(_jwks_url(), timeout=10)
  resp.raise_for_status()
  return resp.json()


def _verify_with_supabase_userinfo(token: str) -> dict[str, Any]:
  api_key = settings.supabase_anon_key or settings.supabase_service_key
  if not api_key:
    logger.warning("auth_failed reason=missing_supabase_api_key_for_userinfo_fallback")
    raise HTTPException(status_code=401, detail="Invalid token")

  key_mode = "anon_key" if settings.supabase_anon_key else "service_key"
  resp = requests.get(
    _user_verify_url(),
    headers={
      "apikey": api_key,
      "Authorization": f"Bearer {token}",
    },
    timeout=10,
  )
  try:
    resp.raise_for_status()
  except Exception as exc:
    logger.warning(
      "auth_failed reason=userinfo_verify_failed key_mode={} status_code={} message={}",
      key_mode,
      getattr(resp, "status_code", None),
      str(exc),
    )
    raise HTTPException(status_code=401, detail="Invalid token") from exc
  logger.info("auth_verify mode=userinfo_fallback key_mode={} status_code={}", key_mode, resp.status_code)
  data = resp.json()
  if not isinstance(data, dict) or not data.get("id"):
    logger.warning("auth_failed reason=userinfo_missing_id")
    raise HTTPException(status_code=401, detail="Invalid token")
  return {"sub": str(data["id"]), "email": data.get("email")}


def verify_supabase_jwt(token: str) -> dict[str, Any]:
  try:
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    alg = header.get("alg")
    jwks = _fetch_jwks()
    keys = jwks.get("keys", [])
    if not keys:
      logger.warning("auth_verify mode=userinfo_fallback reason=empty_jwks kid={} alg={}", kid, alg)
      return _verify_with_supabase_userinfo(token)
    jwk_data = next((key for key in keys if key.get("kid") == kid), None)
    if not jwk_data:
      logger.warning("auth_verify mode=userinfo_fallback reason=unknown_kid kid={} alg={}", kid, alg)
      return _verify_with_supabase_userinfo(token)

    issuer = f"{settings.supabase_url.rstrip('/')}/auth/v1" if settings.supabase_url else None
    options = {"verify_aud": False}
    allowed_algs = ["RS256", "ES256"]
    decode_algs = [alg] if isinstance(alg, str) and alg in allowed_algs else allowed_algs
    return jwt.decode(
      token,
      jwk_data,
      algorithms=decode_algs,
      issuer=issuer,
      options=options,
    )
  except HTTPException:
    raise
  except Exception as exc:
    logger.warning("auth_verify mode=userinfo_fallback reason=jwks_decode_error error_type={} message={}", type(exc).__name__, str(exc))
    return _verify_with_supabase_userinfo(token)


async def get_user_by_auth_id(db: AsyncSession, auth_id: str) -> User | None:
  stmt = select(User).where(User.auth_id == auth_id)
  res = await db.execute(stmt)
  user = res.scalar_one_or_none()
  if user:
    return user

  # Backfill older rows where id was set directly to Supabase auth user id.
  legacy_stmt = select(User).where(User.id == auth_id)
  legacy_res = await db.execute(legacy_stmt)
  legacy_user = legacy_res.scalar_one_or_none()
  if legacy_user:
    legacy_user.auth_id = auth_id
    await db.flush()
    return legacy_user

  return None


async def require_auth(
  authorization: str | None = Header(default=None),
  db: AsyncSession = Depends(get_db),
) -> AuthContext:
  if not authorization or not authorization.startswith("Bearer "):
    logger.warning("auth_failed reason=missing_bearer_header header_present={}", bool(authorization))
    raise HTTPException(status_code=401, detail="Missing token")

  token = authorization.split(" ", 1)[1].strip()
  if not token:
    logger.warning("auth_failed reason=empty_bearer_token")
    raise HTTPException(status_code=401, detail="Missing token")
  claims = verify_supabase_jwt(token)

  auth_user_id = str(claims["sub"])
  user = await get_user_by_auth_id(db, auth_user_id)
  if not user:
    user = User(id=auth_user_id, auth_id=auth_user_id)
    db.add(user)
    try:
      await db.flush()
    except IntegrityError:
      await db.rollback()
      user = await get_user_by_auth_id(db, auth_user_id)
      if not user:
        logger.error("auth_failed reason=user_create_race auth_user_id={}", auth_user_id)
        raise HTTPException(status_code=500, detail="Could not resolve user account")

  verified_email = str(claims.get("email") or "").strip().lower() or None
  return AuthContext(
    auth_user_id=auth_user_id,
    app_user_id=str(user.id),
    email=verified_email,
  )


async def optional_auth(
  authorization: str | None = Header(default=None),
  db: AsyncSession = Depends(get_db),
) -> AuthContext | None:
  if not authorization:
    return None
  return await require_auth(authorization=authorization, db=db)
