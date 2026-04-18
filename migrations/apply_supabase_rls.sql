-- Apply these in Supabase SQL editor for production/Postgres.
-- Review table names and auth mapping before running.

create schema if not exists app;

create or replace function app.current_user_id()
returns uuid
language sql
stable
as $$
  select u.id::uuid
  from users u
  where u.auth_id::text = auth.uid()::text
  limit 1
$$;

alter table users enable row level security;
alter table user_profile enable row level security;
alter table outfits enable row level security;
alter table outfit_scores enable row level security;
alter table outfit_suggestions enable row level security;
alter table style_dna enable row level security;
alter table drip_score_history enable row level security;
alter table user_subscriptions enable row level security;
alter table billing_receipts enable row level security;

drop policy if exists users_select_own on users;
create policy users_select_own on users
for select using (id::uuid = app.current_user_id());

drop policy if exists users_update_own on users;
create policy users_update_own on users
for update using (id::uuid = app.current_user_id())
with check (id::uuid = app.current_user_id());

drop policy if exists user_profile_all_own on user_profile;
create policy user_profile_all_own on user_profile
for all using (user_id::uuid = app.current_user_id())
with check (user_id::uuid = app.current_user_id());

drop policy if exists outfits_all_own on outfits;
create policy outfits_all_own on outfits
for all using (user_id::uuid = app.current_user_id())
with check (user_id::uuid = app.current_user_id());

drop policy if exists style_dna_all_own on style_dna;
create policy style_dna_all_own on style_dna
for all using (user_id::uuid = app.current_user_id())
with check (user_id::uuid = app.current_user_id());

drop policy if exists drip_score_history_all_own on drip_score_history;
create policy drip_score_history_all_own on drip_score_history
for all using (user_id::uuid = app.current_user_id())
with check (user_id::uuid = app.current_user_id());

drop policy if exists user_subscriptions_all_own on user_subscriptions;
create policy user_subscriptions_all_own on user_subscriptions
for all using (user_id::uuid = app.current_user_id())
with check (user_id::uuid = app.current_user_id());

drop policy if exists billing_receipts_all_own on billing_receipts;
create policy billing_receipts_all_own on billing_receipts
for all using (user_id::uuid = app.current_user_id())
with check (user_id::uuid = app.current_user_id());
