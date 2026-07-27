create table if not exists feature_submissions (
  id uuid primary key default gen_random_uuid(),
  outfit_id uuid not null references outfits(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  feature_username text,
  instagram_url text,
  tiktok_url text,
  display_consent boolean not null default false,
  status text not null default 'pending',
  consented_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint uq_feature_submissions_outfit unique (outfit_id),
  constraint ck_feature_submissions_consent check (display_consent = true)
);

create index if not exists idx_feature_submissions_created_at
  on feature_submissions(created_at desc);

alter table feature_submissions enable row level security;

-- Feature submissions are accessed only through the authenticated API.
revoke all on feature_submissions from anon, authenticated;
