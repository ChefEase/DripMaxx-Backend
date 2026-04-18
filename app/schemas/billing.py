from pydantic import BaseModel


class BillingStatusResponse(BaseModel):
  user_id: str
  plan: str
  subscription_status: str
  limit_type: str
  limit: int
  used: int
  remaining: int


class VerifyPurchaseRequest(BaseModel):
  user_id: str | None = None
  platform: str
  product_id: str
  token: str | None = None
  purchase_token: str | None = None
  transaction_id: str | None = None


class VerifyPurchaseResponse(BaseModel):
  ok: bool
  plan: str
  subscription_status: str
  current_period_end: str | None = None
  mode: str
