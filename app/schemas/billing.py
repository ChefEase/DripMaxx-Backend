from pydantic import BaseModel


class BillingStatusResponse(BaseModel):
  user_id: str
  plan: str
  subscription_status: str
  limit_type: str
  limit: int
  used: int
  remaining: int


class RevenueCatSyncRequest(BaseModel):
  user_id: str | None = None
  platform: str


class RevenueCatSyncResponse(BaseModel):
  ok: bool
  plan: str
  subscription_status: str
  current_period_end: str | None = None
  mode: str
