import json

from fastapi import APIRouter, Depends, HTTPException, status
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
  if len(json.dumps(event.payload, separators=(",", ":"), default=str)) > 16_384:
    raise HTTPException(status_code=413, detail="Event payload is too large.")
  # Never accept a caller-supplied user ID. Authenticated events are tied to
  # the verified token; pre-auth events are joined through anonymous_id.
  user_id = auth.app_user_id if auth else None
  db.add(EventLog(
    user_id=user_id,
    anonymous_id=event.anonymous_id,
    name=event.name,
    payload=event.payload,
  ))
  await db.commit()
  return EventOut()
