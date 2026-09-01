create table if not exists outfit_evolution_sessions (
  id varchar(36) primary key,
  user_id varchar(36) not null references users(id) on delete cascade,
  original_outfit_id varchar(36) not null references outfits(id) on delete cascade,
  original_score numeric(4,2) not null check (original_score between 0 and 10),
  current_score numeric(4,2) not null check (current_score between 0 and 10),
  potential_score numeric(4,2) not null check (potential_score between 0 and 10),
  original_analysis jsonb not null default '{}'::jsonb,
  target_look jsonb not null default '{}'::jsonb,
  target_image_url text,
  target_generation_status text not null default 'pending',
  target_generation_error text,
  status text not null default 'active',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint uq_evolution_original_outfit unique (original_outfit_id)
);

create index if not exists idx_evolution_user_updated on outfit_evolution_sessions(user_id, updated_at);

create table if not exists outfit_evolution_recommendations (
  id varchar(36) primary key,
  session_id varchar(36) not null references outfit_evolution_sessions(id) on delete cascade,
  position integer not null,
  category text not null,
  title text not null,
  current_state text,
  recommended_change text not null,
  reason text,
  importance text not null default 'medium' check (importance in ('high','medium','low')),
  target_state text,
  impact numeric(4,2) not null check (impact between 0 and 2),
  created_at timestamptz not null default now(),
  constraint uq_evolution_recommendation_position unique (session_id, position)
);

create index if not exists idx_evolution_recommendation_session on outfit_evolution_recommendations(session_id, position);

create table if not exists outfit_evolution_revisions (
  id varchar(36) primary key,
  session_id varchar(36) not null references outfit_evolution_sessions(id) on delete cascade,
  outfit_id varchar(36) not null references outfits(id) on delete cascade,
  revision_number integer not null,
  previous_score numeric(4,2) not null,
  current_score numeric(4,2) not null,
  score_change numeric(4,2) not null,
  completed_recommendation_ids jsonb not null default '[]'::jsonb,
  recommendation_results jsonb not null default '[]'::jsonb,
  new_issues jsonb not null default '[]'::jsonb,
  summary text not null,
  confidence numeric(4,3) not null,
  created_at timestamptz not null default now(),
  constraint uq_evolution_revision_number unique (session_id, revision_number),
  constraint uq_evolution_revision_outfit unique (outfit_id)
);

create index if not exists idx_evolution_revision_session on outfit_evolution_revisions(session_id, revision_number);

alter table outfit_evolution_sessions add column if not exists target_look jsonb not null default '{}'::jsonb;
alter table outfit_evolution_sessions add column if not exists target_generation_status text not null default 'pending';
alter table outfit_evolution_sessions add column if not exists target_generation_error text;
