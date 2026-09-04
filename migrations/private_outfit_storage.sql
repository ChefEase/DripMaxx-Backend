-- Privacy migration: run once in the Supabase SQL editor before deploying the
-- matching backend. The API uses service-role uploads and short-lived signed
-- reads, so clients never need a direct storage.objects policy.

update storage.buckets
set public = false,
    file_size_limit = 12582912,
    allowed_mime_types = array['image/jpeg', 'image/png', 'image/webp']::text[]
where id = 'outfits';

-- New accounts must affirmatively choose social display. Existing explicit
-- choices are preserved because historical true/false values may be intentional.
alter table user_profile alter column community_feed_choice set default 'undecided';
alter table user_profile alter column leaderboard_choice set default 'undecided';

-- Replace legacy durable Supabase URLs with private object paths. Keeping the
-- existing columns preserves API/database compatibility during rollout.
update outfits
set image_url = regexp_replace(
  image_url,
  '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/',
  ''
)
where image_url ~ '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/';

update outfit_evolution_sessions
set target_image_url = regexp_replace(
  target_image_url,
  '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/',
  ''
)
where target_image_url ~ '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/';

update community_news
set image_url = regexp_replace(
  image_url,
  '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/',
  ''
)
where image_url ~ '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/';

update community_news
set content = jsonb_set(
  content::jsonb,
  '{before_image_url}',
  to_jsonb(regexp_replace(content->>'before_image_url', '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/', '')),
  true
)
where content->>'before_image_url' ~ '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/';

update community_news
set content = jsonb_set(
  content::jsonb,
  '{after_image_url}',
  to_jsonb(regexp_replace(content->>'after_image_url', '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/', '')),
  true
)
where content->>'after_image_url' ~ '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/';

-- Challenge placements are a JSON array. Rewrite only their image_url fields.
update community_news news
set content = jsonb_set(
  news.content::jsonb,
  '{placements}',
  (
    select jsonb_agg(
      case when item->>'image_url' ~ '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/'
        then jsonb_set(item, '{image_url}', to_jsonb(regexp_replace(item->>'image_url', '^https?://[^/]+/storage/v1/object/(public|sign|authenticated)/outfits/', '')))
        else item
      end
    )
    from jsonb_array_elements(news.content::jsonb->'placements') item
  ),
  false
)
where jsonb_typeof(news.content::jsonb->'placements') = 'array';
