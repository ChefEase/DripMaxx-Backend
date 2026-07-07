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
  replicate_llm_model: str = Field(default="openai/gpt-5.4")
  replicate_vlm_model: str = Field(default="openai/gpt-5.4")
  database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")
  supabase_url: str | None = Field(default=None)
  supabase_service_key: str | None = Field(default=None)
  supabase_anon_key: str | None = Field(default=None)
  supabase_bucket: str = Field(default="outfits")
  billing_dev_mode: bool = Field(default=False)
  google_play_package_name: str | None = Field(default=None)
  google_play_service_account_file: str | None = Field(default=None)
  premium_monthly_product_id_android: str = Field(default="dripmaxx_premium_monthly")
  premium_monthly_product_id_ios: str = Field(default="dripmaxx_premium_monthly")
  challenge_admin_user_ids: str = Field(default="")
  challenge_admin_emails: str = Field(default="onyiakamsy74@gmail.com")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  return Settings()
