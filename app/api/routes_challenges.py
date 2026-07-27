from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthContext, optional_auth, require_auth
from app.core.config import get_settings
from app.db.session import get_db
from app.models import (
  Announcement,
  Challenge,
  ChallengeSubmission,
  ChallengeVote,
  CommunityNews,
  Outfit,
  OutfitScore,
  User,
)
from app.schemas.challenges import (
  ActiveChallengeResponse,
  AdminRankRequest,
  AnnouncementResponse,
  ChallengeResponse,
  ChallengeResultsResponse,
  ChallengeSubmissionResponse,
  SubmitChallengeRequest,
  SubmitChallengeResponse,
  VoteRequest,
  WinnerRequest,
)
from app.services.rewards import (
  CHALLENGE_WIN_XP,
  DAILY_CHALLENGE_PARTICIPATION_XP,
  OUTFIT_VOTE_XP,
  add_scan_credits,
  award_xp_once,
)

router = APIRouter(prefix="/v1/challenges", tags=["challenges"])


def _now() -> datetime:
  return datetime.now(timezone.utc)


def _split_env_csv(value: str) -> set[str]:
  return {part.strip().lower() for part in value.split(",") if part.strip()}


async def _require_challenge_admin(auth: AuthContext, db: AsyncSession) -> None:
  settings = get_settings()
  admin_user_ids = _split_env_csv(settings.challenge_admin_user_ids)
  if auth.app_user_id.lower() in admin_user_ids or auth.auth_user_id.lower() in admin_user_ids:
    return
  admin_emails = _split_env_csv(settings.challenge_admin_emails)
  if admin_emails:
    res = await db.execute(select(User.email).where(User.id == auth.app_user_id))
    email = str(res.scalar() or "").strip().lower()
    if email in admin_emails:
      return
  raise HTTPException(status_code=403, detail="Challenge admin access required")


def _announcement_response(row: Announcement | None) -> AnnouncementResponse | None:
  if not row:
    return None
  return AnnouncementResponse(
    id=row.id,
    title=row.title,
    body=row.body,
    cta_label=row.cta_label,
    cta_url=row.cta_url,
    starts_at=row.starts_at,
    ends_at=row.ends_at,
  )


def _challenge_response(row: Challenge | None) -> ChallengeResponse | None:
  if not row:
    return None
  return ChallengeResponse(
    id=row.id,
    title=row.title,
    description=row.description,
    reward_scans=int(row.reward_scans or 0),
    reward_xp=int(row.reward_xp or 0),
    participation_xp=int(row.participation_xp or DAILY_CHALLENGE_PARTICIPATION_XP),
    winner_xp=int(row.winner_xp or CHALLENGE_WIN_XP),
    starts_at=row.starts_at,
    ends_at=row.ends_at,
    winner_submission_id=row.winner_submission_id,
    winner_selected_at=row.winner_selected_at,
  )


async def _get_active_challenge(db: AsyncSession) -> Challenge | None:
  now = _now()
  res = await db.execute(
    select(Challenge)
    .where(
      Challenge.is_active.is_(True),
      Challenge.starts_at <= now,
    )
    .order_by(Challenge.starts_at.desc())
    .limit(1)
  )
  return res.scalar_one_or_none()


async def _recalculate_challenge_scores(db: AsyncSession, challenge_id: str) -> None:
  vote_counts_res = await db.execute(
    select(ChallengeVote.submission_id, func.count(ChallengeVote.id))
    .where(ChallengeVote.challenge_id == challenge_id)
    .group_by(ChallengeVote.submission_id)
  )
  vote_counts = {row[0]: int(row[1] or 0) for row in vote_counts_res.fetchall()}
  total_votes = sum(vote_counts.values())

  submissions_res = await db.execute(
    select(ChallengeSubmission).where(ChallengeSubmission.challenge_id == challenge_id)
  )
  submissions = submissions_res.scalars().all()
  for submission in submissions:
    vote_share = vote_counts.get(submission.id, 0) / total_votes if total_votes else 0
    submission.user_vote_points = round(vote_share * 40, 2)
    submission.final_points = round(float(submission.admin_points or 0) + float(submission.user_vote_points or 0), 2)


@router.get("/active", response_model=ActiveChallengeResponse)
async def get_active_challenge(db: AsyncSession = Depends(get_db)):
  now = _now()
  announcement_res = await db.execute(
    select(Announcement)
    .where(
      Announcement.is_active.is_(True),
      (Announcement.starts_at.is_(None) | (Announcement.starts_at <= now)),
      (Announcement.ends_at.is_(None) | (Announcement.ends_at > now)),
    )
    .order_by(Announcement.priority.desc(), Announcement.created_at.desc())
    .limit(1)
  )
  challenge = await _get_active_challenge(db)
  return ActiveChallengeResponse(
    announcement=_announcement_response(announcement_res.scalar_one_or_none()),
    challenge=_challenge_response(challenge),
  )


@router.post("/active/submissions", response_model=SubmitChallengeResponse)
async def submit_to_active_challenge(
  payload: SubmitChallengeRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  challenge = await _get_active_challenge(db)
  if not challenge:
    raise HTTPException(status_code=404, detail="No active challenge")
  if challenge.ends_at <= _now():
    raise HTTPException(status_code=400, detail="Challenge submissions are closed")
  if challenge.winner_submission_id:
    raise HTTPException(status_code=400, detail="Challenge winner has already been selected")

  outfit_res = await db.execute(
    select(Outfit).where(Outfit.id == payload.outfit_id, Outfit.user_id == auth.app_user_id)
  )
  outfit = outfit_res.scalar_one_or_none()
  if not outfit:
    raise HTTPException(status_code=404, detail="Outfit not found for authenticated user")

  existing_submission_res = await db.execute(
    select(ChallengeSubmission.id).where(
      ChallengeSubmission.challenge_id == challenge.id,
      ChallengeSubmission.user_id == auth.app_user_id,
    )
  )
  if existing_submission_res.scalar_one_or_none():
    raise HTTPException(status_code=409, detail="You already submitted an outfit to this challenge")

  submission = ChallengeSubmission(
    challenge_id=challenge.id,
    user_id=auth.app_user_id,
    outfit_id=outfit.id,
    display_consent=payload.display_consent,
  )
  db.add(submission)
  try:
    await db.flush()
  except IntegrityError as exc:
    raise HTTPException(status_code=409, detail="This outfit was already submitted to the challenge") from exc
  awarded = int(challenge.participation_xp or DAILY_CHALLENGE_PARTICIPATION_XP)
  did_award = await award_xp_once(
    db,
    auth.app_user_id,
    awarded,
    "challenge_participation",
    challenge.id,
    "Daily challenge participation",
  )
  await db.commit()
  return SubmitChallengeResponse(
    submission_id=submission.id,
    challenge_id=challenge.id,
    awarded_xp=awarded if did_award else 0,
  )


@router.post("/{challenge_id}/votes", status_code=204)
async def vote_for_submission(
  challenge_id: str,
  payload: VoteRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  challenge_res = await db.execute(select(Challenge).where(Challenge.id == challenge_id))
  challenge = challenge_res.scalar_one_or_none()
  if not challenge:
    raise HTTPException(status_code=404, detail="Challenge not found")
  if challenge.winner_submission_id:
    raise HTTPException(status_code=400, detail="Challenge winner has already been selected")
  if challenge.ends_at <= _now():
    raise HTTPException(status_code=400, detail="Challenge voting is closed")

  submission_res = await db.execute(
    select(ChallengeSubmission).where(
      ChallengeSubmission.id == payload.submission_id,
      ChallengeSubmission.challenge_id == challenge_id,
    )
  )
  submission = submission_res.scalar_one_or_none()
  if not submission:
    raise HTTPException(status_code=404, detail="Submission not found")

  existing_res = await db.execute(
    select(ChallengeVote).where(
      ChallengeVote.challenge_id == challenge_id,
      ChallengeVote.user_id == auth.app_user_id,
    )
  )
  existing = existing_res.scalar_one_or_none()
  if existing:
    existing.submission_id = submission.id
  else:
    db.add(ChallengeVote(challenge_id=challenge_id, submission_id=submission.id, user_id=auth.app_user_id))
    await award_xp_once(db, auth.app_user_id, OUTFIT_VOTE_XP, "challenge_vote", challenge_id, "Vote on outfits")

  await _recalculate_challenge_scores(db, challenge_id)
  await db.commit()


@router.post("/{challenge_id}/admin-ranks", status_code=204)
async def set_admin_top_three(
  challenge_id: str,
  payload: AdminRankRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  await _require_challenge_admin(auth, db)
  challenge_res = await db.execute(select(Challenge).where(Challenge.id == challenge_id))
  challenge = challenge_res.scalar_one_or_none()
  if not challenge:
    raise HTTPException(status_code=404, detail="Challenge not found")
  if challenge.winner_submission_id:
    raise HTTPException(status_code=409, detail="Winner has already been selected")
  ranked_ids = [payload.first_submission_id, payload.second_submission_id, payload.third_submission_id]
  if len(set(ranked_ids)) != 3:
    raise HTTPException(status_code=400, detail="Top 3 submissions must be unique")

  submissions_res = await db.execute(
    select(ChallengeSubmission).where(ChallengeSubmission.challenge_id == challenge_id)
  )
  submissions = submissions_res.scalars().all()
  by_id = {submission.id: submission for submission in submissions}
  if any(submission_id not in by_id for submission_id in ranked_ids):
    raise HTTPException(status_code=404, detail="One or more submissions were not found")

  for submission in submissions:
    submission.admin_rank = None
    submission.admin_points = 0
  for rank, points, submission_id in ((1, 60, ranked_ids[0]), (2, 40, ranked_ids[1]), (3, 20, ranked_ids[2])):
    by_id[submission_id].admin_rank = rank
    by_id[submission_id].admin_points = points

  await _recalculate_challenge_scores(db, challenge_id)
  await db.commit()


@router.get("/{challenge_id}/results", response_model=ChallengeResultsResponse)
async def get_challenge_results(
  challenge_id: str,
  auth: AuthContext | None = Depends(optional_auth),
  db: AsyncSession = Depends(get_db),
):
  challenge_res = await db.execute(select(Challenge).where(Challenge.id == challenge_id))
  challenge = challenge_res.scalar_one_or_none()
  if not challenge:
    raise HTTPException(status_code=404, detail="Challenge not found")

  stmt = (
    select(ChallengeSubmission, Outfit.image_url, User.username, User.display_name)
    .join(Outfit, Outfit.id == ChallengeSubmission.outfit_id)
    .join(User, User.id == ChallengeSubmission.user_id)
    .where(ChallengeSubmission.challenge_id == challenge_id)
    .order_by(ChallengeSubmission.final_points.desc(), ChallengeSubmission.created_at.asc())
  )
  res = await db.execute(stmt)
  submissions = [
    ChallengeSubmissionResponse(
      id=row[0].id,
      challenge_id=row[0].challenge_id,
      user_id=row[0].user_id,
      outfit_id=row[0].outfit_id,
      image_url=row[1],
      display_name=row[2] or row[3] or "User",
      admin_rank=row[0].admin_rank,
      admin_points=float(row[0].admin_points or 0),
      user_vote_points=float(row[0].user_vote_points or 0),
      final_points=float(row[0].final_points or 0),
      created_at=row[0].created_at,
    )
    for row in res.fetchall()
  ]
  viewer_vote_submission_id = None
  if auth:
    vote_res = await db.execute(
      select(ChallengeVote.submission_id).where(
        ChallengeVote.challenge_id == challenge_id,
        ChallengeVote.user_id == auth.app_user_id,
      )
    )
    viewer_vote_submission_id = vote_res.scalar_one_or_none()
  return ChallengeResultsResponse(
    challenge_id=challenge_id,
    winner_submission_id=challenge.winner_submission_id,
    winner_selected_at=challenge.winner_selected_at,
    viewer_vote_submission_id=viewer_vote_submission_id,
    submissions=submissions,
  )


@router.post("/{challenge_id}/winner", status_code=204)
async def select_winner(
  challenge_id: str,
  payload: WinnerRequest,
  auth: AuthContext = Depends(require_auth),
  db: AsyncSession = Depends(get_db),
):
  await _require_challenge_admin(auth, db)
  challenge_res = await db.execute(select(Challenge).where(Challenge.id == challenge_id))
  challenge = challenge_res.scalar_one_or_none()
  if not challenge:
    raise HTTPException(status_code=404, detail="Challenge not found")
  if challenge.winner_submission_id:
    raise HTTPException(status_code=409, detail="Winner has already been selected")

  submission_res = await db.execute(
    select(ChallengeSubmission).where(
      ChallengeSubmission.id == payload.submission_id,
      ChallengeSubmission.challenge_id == challenge_id,
    )
  )
  submission = submission_res.scalar_one_or_none()
  if not submission:
    raise HTTPException(status_code=404, detail="Submission not found")

  challenge.winner_submission_id = submission.id
  challenge.winner_selected_at = _now()
  winner_user_res = await db.execute(select(User).where(User.id == submission.user_id))
  winner_user = winner_user_res.scalar_one_or_none()
  display_name = (
    winner_user.username
    if winner_user and winner_user.username
    else winner_user.display_name
    if winner_user and winner_user.display_name
    else "A DripMaxx user"
  )
  db.add(
    Announcement(
      title=f"Winner chosen: {display_name}",
      body=f"{display_name} won {challenge.title}. Open the challenge page to see the winning outfit.",
      cta_label="View Winner",
      cta_url="acme://challenge",
      priority=100,
      is_active=True,
      starts_at=challenge.winner_selected_at,
      ends_at=challenge.winner_selected_at + timedelta(days=7),
    )
  )
  winner_outfit_res = await db.execute(
    select(Outfit.image_url, OutfitScore.drip_score)
    .join(OutfitScore, OutfitScore.outfit_id == Outfit.id)
    .where(Outfit.id == submission.outfit_id)
  )
  winner_outfit = winner_outfit_res.first()
  winner_score = round(float(winner_outfit[1]) * 10) if winner_outfit and winner_outfit[1] is not None else None
  score_line = f"\nScore: {winner_score}/100" if winner_score is not None else ""
  db.add(
    CommunityNews(
      news_key=f"challenge-winner:{challenge.id}",
      kind="challenge_winner",
      scope="challenge",
      category=challenge.title,
      eyebrow="🏆 COMMUNITY CHALLENGE",
      title=f"{challenge.title} Winner",
      caption=(
        f"🏆 COMMUNITY CHALLENGE\n\n{challenge.title}\n\n"
        f"Winner: @{display_name}{score_line}\n\n"
        "A standout look chosen by the DripMaxx community.\n\n#DripMaxx"
      ),
      image_url=winner_outfit[0] if winner_outfit else None,
      content={
        "winner": display_name,
        "score": winner_score,
        "challenge_id": challenge.id,
      },
      published_at=challenge.winner_selected_at,
      expires_at=challenge.winner_selected_at + timedelta(days=14),
    )
  )
  await add_scan_credits(db, submission.user_id, int(challenge.reward_scans or 10))
  await award_xp_once(
    db,
    submission.user_id,
    int(challenge.reward_xp or 0),
    "challenge_reward",
    challenge.id,
    "Challenge prize XP",
  )
  await award_xp_once(
    db,
    submission.user_id,
    int(challenge.winner_xp or CHALLENGE_WIN_XP),
    "challenge_win",
    challenge.id,
    "Winning a challenge",
  )
  await db.commit()
