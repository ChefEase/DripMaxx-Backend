create table if not exists community_news (
  id varchar(36) primary key default gen_random_uuid()::text,
  news_key text not null unique,
  kind text not null,
  scope text not null,
  category text not null default '',
  audience_country text,
  eyebrow text not null default 'COMMUNITY FIT',
  title text not null,
  caption text not null,
  image_url text,
  content jsonb not null default '{}'::jsonb,
  published_at timestamptz not null default now(),
  expires_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists idx_community_news_publish_window
  on community_news(published_at desc, expires_at);

create table if not exists community_news_dismissals (
  id varchar(36) primary key default gen_random_uuid()::text,
  news_id varchar(36) not null references community_news(id) on delete cascade,
  user_id varchar(36) not null references users(id) on delete cascade,
  dismissed_at timestamptz not null default now(),
  constraint uq_community_news_dismissal unique(news_id, user_id)
);

create table if not exists community_news_likes (
  id varchar(36) primary key default gen_random_uuid()::text,
  news_id varchar(36) not null references community_news(id) on delete cascade,
  user_id varchar(36) not null references users(id) on delete cascade,
  created_at timestamptz not null default now(),
  constraint uq_community_news_like unique(news_id, user_id)
);

alter table community_news enable row level security;
alter table community_news_dismissals enable row level security;
alter table community_news_likes enable row level security;
revoke all on community_news from anon, authenticated;
revoke all on community_news_dismissals from anon, authenticated;
revoke all on community_news_likes from anon, authenticated;
