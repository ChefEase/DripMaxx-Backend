from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  """Application settings pulled from environment."""

  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

  app_name: str = Field(default="DripMaxx API")
  environment: str = Field(default="local")
  debug: bool = Field(default=True)

  replicate_api_token: str | None = Field(default=None)
  replicate_model: str = Field(
    default="krthr/clip-embeddings:1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4"
  )
  replicate_llm_model: str = Field(default="meta/meta-llama-3-70b-instruct")
  replicate_vlm_model: str = Field(
    default="chefease/fashionvlmodel:7f22da0c6e72f395e33b12c0467c9d9d158cbed71ca391e84c8704ef656c2609"
  )
  database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")
  supabase_url: str | None = Field(default=None)
  supabase_service_key: str | None = Field(default=None)
  supabase_bucket: str = Field(default="outfits")

  stripe_secret_key: str | None = Field(default=None)
  stripe_webhook_secret: str | None = Field(default=None)
  stripe_monthly_price_id: str | None = Field(default=None)
  stripe_success_url: str = Field(default="https://example.com/success")
  stripe_cancel_url: str = Field(default="https://example.com/cancel")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  return Settings()
