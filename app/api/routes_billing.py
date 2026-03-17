from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.billing import BillingStatusResponse
from app.services.usage_limits import get_scan_quota

router = APIRouter(prefix="/v1/billing", tags=["billing"])


@router.get("/status", response_model=BillingStatusResponse)
async def billing_status(user_id: str = Query(...), db: AsyncSession = Depends(get_db)):
  quota = await get_scan_quota(db, user_id)
  return BillingStatusResponse(
    user_id=user_id,
    plan=quota["plan"],
    subscription_status=quota["subscription_status"],
    limit_type=quota["limit_type"],
    limit=quota["limit"],
    used=quota["used"],
    remaining=quota["remaining"],
  )
