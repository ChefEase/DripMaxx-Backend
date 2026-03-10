import enum
import uuid
from sqlalchemy import (
  Boolean,
  CheckConstraint,
  Column,
  DateTime,
  Enum,
  ForeignKey,
  Integer,
  Numeric,
  String,
  Text,
  UniqueConstraint,
  Index,
  func,
  JSON,
  text,
)
from app.db.session import Base

# SQLite-safe UUID string column
UUID_STR = String(36)
def _uuid():
  return str(uuid.uuid4())


class BodyTypeEnum(str, enum.Enum):
  slim = "slim"
  athletic = "athletic"
  average = "average"
  broad = "broad"
  plus_size = "plus_size"


class GenderStyleEnum(str, enum.Enum):
  menswear = "menswear"
  womenswear = "womenswear"
  neutral = "neutral"


class SuggestionTypeEnum(str, enum.Enum):
  fit = "fit"
  layering = "layering"
  color = "color"
  accessory = "accessory"
  other = "other"


class User(Base):
  __tablename__ = "users"

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  auth_id = Column(Text, unique=True)
  username = Column(Text, unique=True, index=True)
  email = Column(Text, unique=True)
  display_name = Column(Text)
  avatar_url = Column(Text)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
  updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class UserProfile(Base):
  __tablename__ = "user_profile"

  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
  style_preference = Column(Text)
  height_cm = Column(Numeric(5, 2))
  body_type = Column(Enum(BodyTypeEnum, name="body_type_enum", create_constraint=True))
  gender_style_preference = Column(Enum(GenderStyleEnum, name="gender_style_enum", create_constraint=True))
  country = Column(Text)
  locale = Column(Text)
  # True = public, False = private/friends_only
  profile_visibility = Column(Boolean, server_default=text("true"))
  # Canonical visibility mode: public | friends_only | private
  profile_visibility_mode = Column(Text, server_default=text("'public'"))
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
  updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class StyleInspirationCatalog(Base):
  __tablename__ = "style_inspiration_catalog"

  id = Column(Integer, primary_key=True, autoincrement=True)
  name = Column(Text, nullable=False)
  slug = Column(Text, unique=True)
  image_url = Column(Text)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class UserStyleInspiration(Base):
  __tablename__ = "user_style_inspiration"

  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
  inspiration_id = Column(Integer, ForeignKey("style_inspiration_catalog.id", ondelete="CASCADE"), primary_key=True)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class UserCustomInspiration(Base):
  __tablename__ = "user_custom_inspiration"

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"))
  label = Column(Text, nullable=False)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Outfit(Base):
  __tablename__ = "outfits"
  __table_args__ = (
    Index("idx_outfits_user_id_scanned_at", "user_id", "scanned_at"),
  )

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"))
  style_tags = Column(JSON, default=list)
  source = Column(Text, nullable=False)
  image_url = Column(Text, nullable=False)
  thumb_url = Column(Text)
  notes = Column(Text)
  scanned_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
  is_example = Column(Boolean, nullable=False, server_default=text("false"))
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
  updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OutfitScore(Base):
  __tablename__ = "outfit_scores"
  __table_args__ = (
    UniqueConstraint("outfit_id", name="uq_outfit_scores_outfit_id"),
    CheckConstraint("color_match BETWEEN 0 AND 10", name="ck_color_match_bounds"),
    CheckConstraint("fit_quality BETWEEN 0 AND 10", name="ck_fit_quality_bounds"),
    CheckConstraint("body_compatibility BETWEEN 0 AND 10", name="ck_body_compatibility_bounds"),
    CheckConstraint("trend_score BETWEEN 0 AND 10", name="ck_trend_score_bounds"),
    CheckConstraint("style_match BETWEEN 0 AND 10", name="ck_style_match_bounds"),
    CheckConstraint("drip_score BETWEEN 0 AND 10", name="ck_drip_score_bounds"),
    Index("idx_outfit_scores_drip_score", "drip_score"),
  )

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  outfit_id = Column(UUID_STR, ForeignKey("outfits.id", ondelete="CASCADE"), nullable=False)

  color_match = Column(Numeric(4, 2))
  fit_quality = Column(Numeric(4, 2))
  body_compatibility = Column(Numeric(4, 2))
  trend_score = Column(Numeric(4, 2))
  style_match = Column(Numeric(4, 2))
  drip_score = Column(Numeric(4, 2))

  model_version = Column(Text)
  raw_features = Column(JSON)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OutfitSuggestion(Base):
  __tablename__ = "outfit_suggestions"
  __table_args__ = (
    Index("idx_outfit_suggestions_outfit_rank", "outfit_id", "rank"),
  )

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  outfit_id = Column(UUID_STR, ForeignKey("outfits.id", ondelete="CASCADE"))
  type = Column(Enum(SuggestionTypeEnum, name="suggestion_type_enum", create_constraint=True), nullable=False)
  title = Column(Text, nullable=False)
  description = Column(Text)
  rank = Column(Integer, nullable=False, server_default=text("1"))
  is_applied = Column(Boolean)
  is_liked = Column(Boolean)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
  updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class StyleDNA(Base):
  __tablename__ = "style_dna"

  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
  label = Column(Text)
  description = Column(Text)
  tags = Column(JSON, default=list)
  embedding = Column(JSON)  # placeholder until pgvector is available
  metadata_json = Column("metadata", JSON)
  updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class DripScoreHistory(Base):
  __tablename__ = "drip_score_history"
  __table_args__ = (
    Index("idx_drip_score_history_user_time", "user_id", "recorded_at"),
  )

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"))
  outfit_id = Column(UUID_STR, ForeignKey("outfits.id", ondelete="SET NULL"))
  drip_score = Column(Numeric(4, 2))
  recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EventLog(Base):
  __tablename__ = "event_log"
  __table_args__ = (Index("idx_event_log_user_time", "user_id", "created_at"),)

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  user_id = Column(UUID_STR, nullable=True)
  name = Column(Text, nullable=False)
  payload = Column(JSON)
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RankingGroup(Base):
  """Private group for friend rankings. Users join via shareable code."""
  __tablename__ = "ranking_groups"
  __table_args__ = (UniqueConstraint("code", name="uq_ranking_groups_code"),)

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  name = Column(Text, nullable=False)
  code = Column(Text, unique=True, nullable=False, index=True)
  created_by_user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="SET NULL"))
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RankingGroupMember(Base):
  __tablename__ = "ranking_group_members"
  __table_args__ = (UniqueConstraint("group_id", "user_id", name="uq_ranking_group_member"),)

  id = Column(UUID_STR, primary_key=True, default=_uuid)
  group_id = Column(UUID_STR, ForeignKey("ranking_groups.id", ondelete="CASCADE"), nullable=False)
  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  joined_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class UserSubscription(Base):
  __tablename__ = "user_subscriptions"

  user_id = Column(UUID_STR, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
  plan = Column(Text, nullable=False, server_default=text("'free'"))  # free|monthly
  status = Column(Text, nullable=False, server_default=text("'inactive'"))  # inactive|active|trialing|past_due|canceled
  stripe_customer_id = Column(Text, unique=True)
  stripe_subscription_id = Column(Text, unique=True)
  current_period_start = Column(DateTime(timezone=True))
  current_period_end = Column(DateTime(timezone=True))
  created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
  updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
