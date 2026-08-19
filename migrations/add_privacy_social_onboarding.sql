-- Existing accounts retain their current experience. New profiles are created
-- with onboarding_privacy_completed=false explicitly by the API.
alter table user_profile
  add column if not exists onboarding_privacy_completed boolean not null default true;
alter table user_profile
  add column if not exists profile_visibility_choice text;
alter table user_profile
  add column if not exists community_feed_choice text not null default 'true';
alter table user_profile
  add column if not exists leaderboard_choice text not null default 'true';

-- Preserve existing visibility instead of resetting private/friends-only
-- accounts to public. Friends-only maps to the new safer Private option.
update user_profile
set profile_visibility_choice = case
  when profile_visibility_mode in ('private', 'friends_only') then 'private'
  else 'public'
end
where profile_visibility_choice is null;

alter table user_profile alter column profile_visibility_choice set default 'public';
alter table user_profile alter column profile_visibility_choice set not null;

alter table user_profile drop constraint if exists ck_profile_visibility_choice;
alter table user_profile add constraint ck_profile_visibility_choice
  check (profile_visibility_choice in ('private', 'public', 'undecided'));
alter table user_profile drop constraint if exists ck_community_feed_choice;
alter table user_profile add constraint ck_community_feed_choice
  check (community_feed_choice in ('true', 'false', 'undecided'));
alter table user_profile drop constraint if exists ck_leaderboard_choice;
alter table user_profile add constraint ck_leaderboard_choice
  check (leaderboard_choice in ('true', 'false', 'undecided'));
