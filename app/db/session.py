from contextlib import asynccontextmanager

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
  yield
  await engine.dispose()


async def get_db() -> AsyncSession:
  async with AsyncSessionLocal() as session:
    yield session
