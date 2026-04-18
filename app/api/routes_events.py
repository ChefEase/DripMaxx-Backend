from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, optional_auth
from app.db.session import get_db
from app.models.entities import EventLog
from app.schemas.events import EventIn, EventOut

router = APIRouter(prefix="/v1/events", tags=["events"])


@router.post("", response_model=EventOut, status_code=status.HTTP_202_ACCEPTED)
async def track_event(
  event: EventIn,
  auth: AuthContext | None = Depends(optional_auth),
  db: AsyncSession = Depends(get_db),
):
  user_id = auth.app_user_id if auth else event.user_id
  db.add(EventLog(user_id=user_id, name=event.name, payload=event.payload))
  await db.commit()
  return EventOut()
