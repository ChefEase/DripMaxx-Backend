from pydantic import BaseModel


class BillingStatusResponse(BaseModel):
  user_id: str
  plan: str
  subscription_status: str
  limit_type: str
  limit: int
  used: int
  remaining: int
