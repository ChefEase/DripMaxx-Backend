-- Month 1 first-party growth analytics. Safe to run repeatedly in Supabase.
alter table event_log add column if not exists anonymous_id varchar(80);
create index if not exists idx_event_log_anonymous_time on event_log (anonymous_id, created_at);
create index if not exists idx_event_log_name_time on event_log (name, created_at);

-- One row per calendar day with unique-user funnel counts. New accounts and
-- completed stored ratings remain database truth; client events describe the
-- steps immediately before and after those durable records.
create or replace view growth_daily_funnel as
with dates as (
  select generate_series(
    least(
      coalesce((select min(created_at)::date from users), current_date),
      coalesce((select min(created_at)::date from event_log), current_date)
    ),
    current_date,
    interval '1 day'
  )::date as day
), user_days as (
  select created_at::date as day, count(*) as new_accounts
  from users group by 1
), event_days as (
  select
    created_at::date as day,
    count(distinct coalesce(user_id, anonymous_id)) as active_users,
    count(distinct user_id) filter (where name in ('onboard_completed', 'privacy_onboarding_completed')) as onboarded_users,
    count(distinct user_id) filter (where name = 'scan_started') as rating_started_users,
    count(distinct user_id) filter (where name = 'score_viewed') as rating_completed_users,
    count(distinct user_id) filter (where name = 'style_started') as styling_started_users,
    count(distinct user_id) filter (where name = 'style_completed') as styling_completed_users,
    count(distinct user_id) filter (where name = 'paywall_viewed') as paywall_viewed_users,
    count(distinct user_id) filter (where name = 'purchase_completed') as purchased_users,
    count(distinct user_id) filter (where name = 'score_shared') as sharing_users
  from event_log group by 1
)
select
  dates.day,
  coalesce(user_days.new_accounts, 0) as new_accounts,
  coalesce(event_days.active_users, 0) as active_users,
  coalesce(event_days.onboarded_users, 0) as onboarded_users,
  coalesce(event_days.rating_started_users, 0) as rating_started_users,
  coalesce(event_days.rating_completed_users, 0) as rating_completed_users,
  coalesce(event_days.styling_started_users, 0) as styling_started_users,
  coalesce(event_days.styling_completed_users, 0) as styling_completed_users,
  coalesce(event_days.paywall_viewed_users, 0) as paywall_viewed_users,
  coalesce(event_days.purchased_users, 0) as purchased_users,
  coalesce(event_days.sharing_users, 0) as sharing_users
from dates
left join user_days using (day)
left join event_days using (day)
order by dates.day desc;

-- First-touch campaign performance among identifiable DripMaxx users.
create or replace view growth_campaign_performance as
with attributed_users as (
  select distinct on (user_id)
    user_id,
    coalesce(payload #>> '{attribution,first_touch,source}', 'direct_or_unknown') as source,
    coalesce(payload #>> '{attribution,first_touch,medium}', 'unknown') as medium,
    coalesce(payload #>> '{attribution,first_touch,campaign}', 'unknown') as campaign,
    coalesce(payload #>> '{attribution,first_touch,content}', 'unknown') as content,
    coalesce(payload #>> '{attribution,first_touch,creator}', 'none') as creator
  from event_log
  where user_id is not null
    and payload #>> '{attribution,first_touch,captured_at}' is not null
  order by user_id, created_at
), outcomes as (
  select
    user_id,
    bool_or(name in ('score_viewed', 'style_completed')) as activated,
    bool_or(name = 'score_shared') as shared,
    bool_or(name = 'paywall_viewed') as saw_paywall,
    bool_or(name = 'purchase_completed') as purchased,
    count(distinct created_at::date) >= 2 as used_multiple_days
  from event_log where user_id is not null group by user_id
)
select
  source, medium, campaign, content, creator,
  count(*) as identified_users,
  count(*) filter (where outcomes.activated) as activated_users,
  count(*) filter (where outcomes.used_multiple_days) as second_use_users,
  count(*) filter (where outcomes.shared) as sharing_users,
  count(*) filter (where outcomes.saw_paywall) as paywall_users,
  count(*) filter (where outcomes.purchased) as purchased_users,
  round(100.0 * count(*) filter (where outcomes.activated) / nullif(count(*), 0), 1) as activation_rate_pct,
  round(100.0 * count(*) filter (where outcomes.used_multiple_days) / nullif(count(*) filter (where outcomes.activated), 0), 1) as second_use_rate_pct
from attributed_users
left join outcomes using (user_id)
group by source, medium, campaign, content, creator
order by activated_users desc, identified_users desc;

-- Signup cohorts with exact-day D1 and D7 return rates.
create or replace view growth_cohort_retention as
with cohorts as (
  select id as user_id, created_at::date as cohort_day from users
), activity as (
  select distinct user_id, created_at::date as active_day
  from event_log where user_id is not null
)
select
  cohorts.cohort_day,
  count(distinct cohorts.user_id) as new_users,
  count(distinct cohorts.user_id) filter (where d1.user_id is not null) as d1_returned,
  count(distinct cohorts.user_id) filter (where d7.user_id is not null) as d7_returned,
  round(100.0 * count(distinct cohorts.user_id) filter (where d1.user_id is not null) / nullif(count(distinct cohorts.user_id), 0), 1) as d1_retention_pct,
  round(100.0 * count(distinct cohorts.user_id) filter (where d7.user_id is not null) / nullif(count(distinct cohorts.user_id), 0), 1) as d7_retention_pct
from cohorts
left join activity d1 on d1.user_id = cohorts.user_id and d1.active_day = cohorts.cohort_day + 1
left join activity d7 on d7.user_id = cohorts.user_id and d7.active_day = cohorts.cohort_day + 7
group by cohorts.cohort_day
order by cohorts.cohort_day desc;

-- These are founder/operator views for the Supabase SQL editor, not public API data.
revoke all on growth_daily_funnel from anon, authenticated;
revoke all on growth_campaign_performance from anon, authenticated;
revoke all on growth_cohort_retention from anon, authenticated;
