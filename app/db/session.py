import asyncio
from contextlib import asynccontextmanager, suppress

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
  pass


engine = create_async_engine(
  settings.database_url,
  echo=settings.debug,
  future=True,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def lifespan(app):
  """FastAPI lifespan hook for clean startup/shutdown."""
  async def settle_leaderboard_awards():
    # Allow startup schema initialization to finish before the first pass.
    await asyncio.sleep(60)
    while True:
      try:
        from app.services.leaderboard_rewards import (
          settle_recent_leaderboard_periods,
          sync_all_time_badges,
        )

        async with AsyncSessionLocal() as session:
          await settle_recent_leaderboard_periods(session)
          await sync_all_time_badges(session)
      except asyncio.CancelledError:
        raise
      except Exception:
        # A later hourly pass retries safely; award rows are idempotent.
        pass
      await asyncio.sleep(3600)

  settlement_task = asyncio.create_task(settle_leaderboard_awards())
  try:
    yield
  finally:
    settlement_task.cancel()
    with suppress(asyncio.CancelledError):
      await settlement_task
    await engine.dispose()


async def get_db() -> AsyncSession:
  async with AsyncSessionLocal() as session:
    yield session
