from pydantic import BaseModel


class BillingStatusResponse(BaseModel):
  user_id: str
  plan: str
  subscription_status: str
  limit_type: str
  limit: int
  used: int
  remaining: int


class CheckoutSessionRequest(BaseModel):
  user_id: str
  email: str | None = None


class CheckoutSessionResponse(BaseModel):
  checkout_url: str
  session_id: str

