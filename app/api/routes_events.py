from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.entities import EventLog
from app.schemas.events import EventIn, EventOut

router = APIRouter(prefix="/v1/events", tags=["events"])


@router.post("", response_model=EventOut, status_code=status.HTTP_202_ACCEPTED)
async def track_event(event: EventIn, db: AsyncSession = Depends(get_db)):
  db.add(EventLog(user_id=event.user_id, name=event.name, payload=event.payload))
  await db.commit()
  return EventOut()
