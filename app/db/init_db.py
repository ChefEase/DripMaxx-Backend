from app.db.session import engine
from app.models import Base


async def init_db():
  """Create tables if they don't exist (safe for local dev)."""
  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
