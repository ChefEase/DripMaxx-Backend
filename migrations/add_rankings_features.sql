-- Migration: Add style_tags to outfits, profile_visibility to user_profile
-- Run this if your DB was created before these changes.
-- PostgreSQL only.

-- Add style_tags to outfits (JSON array, default [])
ALTER TABLE outfits ADD COLUMN IF NOT EXISTS style_tags JSONB DEFAULT '[]';

-- Add profile_visibility to user_profile
ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS profile_visibility TEXT DEFAULT 'public';
