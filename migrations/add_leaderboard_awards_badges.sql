create table if not exists leaderboard_awards (
  id varchar(36) primary key default gen_random_uuid()::text,
  user_id varchar(36) not null references users(id) on delete cascade,
  scope text not null,
  category text not null default '',
  period_start timestamptz not null,
  period_end timestamptz not null,
  rank integer not null check (rank between 1 and 3),
  xp_awarded integer not null default 0,
  scan_credits_awarded integer not null default 0,
  created_at timestamptz not null default now(),
  constraint uq_leaderboard_period_rank unique (scope, category, period_start, rank)
);

create index if not exists idx_leaderboard_awards_user_time
  on leaderboard_awards(user_id, created_at desc);

create table if not exists user_badges (
  id varchar(36) primary key default gen_random_uuid()::text,
  user_id varchar(36) not null references users(id) on delete cascade,
  award_id varchar(36) references leaderboard_awards(id) on delete cascade,
  badge_key text not null,
  label text not null,
  rank integer not null check (rank between 1 and 3),
  scope text not null,
  category text not null default '',
  is_current boolean not null default false,
  earned_at timestamptz not null default now(),
  constraint uq_user_badges_award unique (award_id)
);

create index if not exists idx_user_badges_user_time
  on user_badges(user_id, earned_at desc);

alter table user_badges alter column award_id drop not null;
alter table user_badges add column if not exists is_current boolean not null default false;

alter table leaderboard_awards enable row level security;
alter table user_badges enable row level security;
revoke all on leaderboard_awards from anon, authenticated;
revoke all on user_badges from anon, authenticated;
