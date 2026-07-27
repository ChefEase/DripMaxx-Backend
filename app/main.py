from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_health, routes_outfits, routes_profile, routes_events, routes_rankings, routes_users, routes_billing, routes_challenges, routes_rewards, routes_features, routes_news
from app.core.config import get_settings
from app.db.session import lifespan

api = FastAPI(title="DripMaxx API", lifespan=lifespan)
settings = get_settings()


@api.get("/", summary="Root health check")
async def root():
  return {"ok": True, "service": settings.app_name, "environment": settings.environment}


api.include_router(routes_health.router)
api.include_router(routes_outfits.router)
api.include_router(routes_profile.router)
api.include_router(routes_events.router)
api.include_router(routes_rankings.router)
api.include_router(routes_users.router)
api.include_router(routes_billing.router)
api.include_router(routes_challenges.router)
api.include_router(routes_rewards.router)
api.include_router(routes_features.router)
api.include_router(routes_news.router)

# Keep CORS outside FastAPI's error middleware so unhandled 500 responses also
# include CORS headers and remain inspectable by browser clients.
app = CORSMiddleware(
  app=api,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
