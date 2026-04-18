from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests
from fastapi import Depends, Header, HTTPException, status
from jose import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.models import User


settings = get_settings()


@dataclass
class AuthContext:
  auth_user_id: str
  app_user_id: str


def _jwks_url() -> str:
  if not settings.supabase_url:
    raise HTTPException(status_code=500, detail="Supabase auth is not configured")
  return f"{settings.supabase_url.rstrip('/')}/auth/v1/keys"


@lru_cache(maxsize=1)
def _fetch_jwks() -> dict[str, Any]:
  resp = requests.get(_jwks_url(), timeout=10)
  resp.raise_for_status()
  return resp.json()


def verify_supabase_jwt(token: str) -> dict[str, Any]:
  try:
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    jwks = _fetch_jwks()
    keys = jwks.get("keys", [])
    jwk_data = next((key for key in keys if key.get("kid") == kid), None)
    if not jwk_data:
      raise HTTPException(status_code=401, detail="Invalid token")

    issuer = f"{settings.supabase_url.rstrip('/')}/auth/v1" if settings.supabase_url else None
    options = {"verify_aud": False}
    return jwt.decode(
      token,
      jwk_data,
      algorithms=["RS256"],
      issuer=issuer,
      options=options,
    )
  except HTTPException:
    raise
  except Exception as exc:
    raise HTTPException(status_code=401, detail="Invalid token") from exc


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
    raise HTTPException(status_code=401, detail="Missing token")

  token = authorization.split(" ", 1)[1].strip()
  claims = verify_supabase_jwt(token)

  auth_user_id = str(claims["sub"])
  user = await get_user_by_auth_id(db, auth_user_id)
  if not user:
    user = User(id=auth_user_id, auth_id=auth_user_id)
    db.add(user)
    await db.flush()

  return AuthContext(auth_user_id=auth_user_id, app_user_id=str(user.id))


async def optional_auth(
  authorization: str | None = Header(default=None),
  db: AsyncSession = Depends(get_db),
) -> AuthContext | None:
  if not authorization:
    return None
  return await require_auth(authorization=authorization, db=db)
