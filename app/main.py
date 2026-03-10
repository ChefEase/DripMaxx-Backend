from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_health, routes_outfits, routes_profile, routes_events, routes_rankings, routes_users, routes_billing
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.init_db import init_db
from app.db.session import lifespan

app = FastAPI(title="DripMaxx API", lifespan=lifespan)
logger = setup_logging()
settings = get_settings()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
  logger.info(f"starting {settings.app_name} [{settings.environment}] (debug={settings.debug})")
  try:
    await init_db()
    logger.info("database schema ready")
  except Exception as exc:  # pragma: no cover - startup logging only
    logger.exception(f"failed to initialize database: {exc}")


app.include_router(routes_health.router)
app.include_router(routes_outfits.router)
app.include_router(routes_profile.router)
app.include_router(routes_events.router)
app.include_router(routes_rankings.router)
app.include_router(routes_users.router)
app.include_router(routes_billing.router)
