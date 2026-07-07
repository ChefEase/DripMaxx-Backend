-- Weekly challenges, announcements, XP, voting, and earned scan credits.
-- PostgreSQL/Supabase migration.

CREATE TABLE IF NOT EXISTS announcements (
  id varchar(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
  title text NOT NULL,
  body text,
  cta_label text,
  cta_url text,
  priority integer NOT NULL DEFAULT 0,
  is_active boolean NOT NULL DEFAULT true,
  starts_at timestamptz,
  ends_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_announcements_active_window
  ON announcements (is_active, starts_at, ends_at);

CREATE TABLE IF NOT EXISTS challenges (
  id varchar(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
  title text NOT NULL,
  description text,
  reward_scans integer NOT NULL DEFAULT 10,
  reward_xp integer NOT NULL DEFAULT 250,
  participation_xp integer NOT NULL DEFAULT 25,
  winner_xp integer NOT NULL DEFAULT 100,
  is_active boolean NOT NULL DEFAULT true,
  starts_at timestamptz NOT NULL,
  ends_at timestamptz NOT NULL,
  winner_submission_id varchar(36),
  winner_selected_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT ck_challenges_window CHECK (ends_at > starts_at)
);

CREATE INDEX IF NOT EXISTS idx_challenges_active_window
  ON challenges (is_active, starts_at, ends_at);

CREATE TABLE IF NOT EXISTS challenge_submissions (
  id varchar(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
  challenge_id varchar(36) NOT NULL REFERENCES challenges (id) ON DELETE CASCADE,
  user_id varchar(36) NOT NULL REFERENCES users (id) ON DELETE CASCADE,
  outfit_id varchar(36) NOT NULL REFERENCES outfits (id) ON DELETE CASCADE,
  display_consent boolean NOT NULL DEFAULT false,
  admin_rank integer,
  admin_points numeric(6,2) NOT NULL DEFAULT 0,
  user_vote_points numeric(6,2) NOT NULL DEFAULT 0,
  final_points numeric(6,2) NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT uq_challenge_user_submission UNIQUE (challenge_id, user_id),
  CONSTRAINT uq_challenge_user_outfit UNIQUE (challenge_id, user_id, outfit_id),
  CONSTRAINT ck_challenge_admin_rank CHECK (admin_rank IS NULL OR admin_rank IN (1, 2, 3))
);

CREATE INDEX IF NOT EXISTS idx_challenge_submissions_challenge
  ON challenge_submissions (challenge_id, created_at);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'uq_challenge_user_submission'
  ) THEN
    ALTER TABLE challenge_submissions
      ADD CONSTRAINT uq_challenge_user_submission UNIQUE (challenge_id, user_id);
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS challenge_votes (
  id varchar(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
  challenge_id varchar(36) NOT NULL REFERENCES challenges (id) ON DELETE CASCADE,
  submission_id varchar(36) NOT NULL REFERENCES challenge_submissions (id) ON DELETE CASCADE,
  user_id varchar(36) NOT NULL REFERENCES users (id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT uq_challenge_vote_user UNIQUE (challenge_id, user_id)
);

CREATE TABLE IF NOT EXISTS user_reward_balances (
  user_id varchar(36) PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
  xp integer NOT NULL DEFAULT 0,
  scan_credits integer NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS xp_ledger (
  id varchar(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
  user_id varchar(36) NOT NULL REFERENCES users (id) ON DELETE CASCADE,
  points integer NOT NULL,
  source_type text NOT NULL,
  source_id varchar(36),
  note text,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT uq_xp_source_once UNIQUE (user_id, source_type, source_id)
);

CREATE INDEX IF NOT EXISTS idx_xp_ledger_user_time
  ON xp_ledger (user_id, created_at);

-- Example: create the current weekly challenge from SQL or Supabase Table Editor.
-- INSERT INTO announcements (title, body, priority, starts_at, ends_at)
-- VALUES ('Today''s Challenge', 'Best Summer Outfit. Reward: 10 scans and 250 XP.', 10, now(), now() + interval '7 days');
--
-- INSERT INTO challenges (title, description, starts_at, ends_at, reward_scans, reward_xp)
-- VALUES ('Best Summer Outfit', 'Submit your best summer fit this week.', now(), now() + interval '7 days', 10, 250);
