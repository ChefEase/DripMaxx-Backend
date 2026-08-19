from app.db.session import engine
from app.models import Base
from sqlalchemy import text


async def _ensure_privacy_social_columns(conn):
  """Apply the additive onboarding migration to existing PostgreSQL databases.

  `create_all` creates fresh schemas but intentionally does not alter existing
  tables. These statements are idempotent so a Render restart can safely repair
  a deployment where application code arrived before the SQL migration.
  """
  if conn.dialect.name != "postgresql":
    return
  statements = [
    "alter table user_profile add column if not exists onboarding_privacy_completed boolean not null default true",
    "alter table user_profile add column if not exists profile_visibility_choice text",
    "alter table user_profile add column if not exists community_feed_choice text not null default 'true'",
    "alter table user_profile add column if not exists leaderboard_choice text not null default 'true'",
    """update user_profile
       set profile_visibility_choice = case
         when profile_visibility_mode in ('private', 'friends_only') then 'private'
         else 'public'
       end
       where profile_visibility_choice is null""",
    "alter table user_profile alter column profile_visibility_choice set default 'public'",
    "alter table user_profile alter column profile_visibility_choice set not null",
  ]
  for statement in statements:
    await conn.execute(text(statement))


async def init_db():
  """Create tables if they don't exist (safe for local dev)."""
  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
    await _ensure_privacy_social_columns(conn)
