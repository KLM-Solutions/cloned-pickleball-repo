-- ============================================================================
-- StrikeSense Database Migration: Users Table with Foreign Key
-- ============================================================================
-- Run this migration in your Supabase SQL Editor:
-- https://app.supabase.com/project/YOUR_PROJECT/sql/new
-- ============================================================================

-- 1. CREATE USERS TABLE
-- Clerk user_id is the primary key (e.g., "user_2abc123...")
CREATE TABLE IF NOT EXISTS public.users (
    id TEXT PRIMARY KEY,  -- Clerk user_id
    email TEXT,
    first_name TEXT,
    last_name TEXT,
    profile_image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);

-- Enable Row Level Security
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role to do everything (for webhook)
CREATE POLICY "Service role has full access" ON public.users
    FOR ALL 
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- 2. ADD FOREIGN KEY TO ANALYSIS_JOBS
-- ============================================================================
-- NOTE: Before running this, ensure:
-- - The user_id column exists in analysis_jobs (should already be there)
-- - Any existing user_id values have matching entries in users table
--   OR those records should be deleted/nullified first

-- First, check if user_id column exists (should from previous migration)
-- If not, add it:
-- ALTER TABLE public.analysis_jobs ADD COLUMN IF NOT EXISTS user_id TEXT;

-- IMPORTANT: Handle orphan records before adding foreign key
-- Option A: Delete orphan analysis jobs (uncomment if needed)
-- DELETE FROM public.analysis_jobs 
-- WHERE user_id IS NOT NULL 
-- AND user_id NOT IN (SELECT id FROM public.users);

-- Option B: Set orphan user_ids to NULL (safer, keeps data)
UPDATE public.analysis_jobs 
SET user_id = NULL 
WHERE user_id IS NOT NULL 
AND user_id NOT IN (SELECT id FROM public.users);

-- Now add the foreign key constraint
ALTER TABLE public.analysis_jobs
DROP CONSTRAINT IF EXISTS fk_analysis_jobs_user;

ALTER TABLE public.analysis_jobs
ADD CONSTRAINT fk_analysis_jobs_user
FOREIGN KEY (user_id) REFERENCES public.users(id)
ON DELETE CASCADE;  -- When user is deleted, delete all their analyses too

-- ============================================================================
-- 3. VERIFICATION QUERIES
-- ============================================================================
-- Run these separately to verify the migration worked:

-- Check users table exists
-- SELECT * FROM public.users LIMIT 5;

-- Check foreign key constraint
-- SELECT conname, conrelid::regclass, confrelid::regclass
-- FROM pg_constraint 
-- WHERE conname = 'fk_analysis_jobs_user';

-- Check relationship
-- SELECT aj.id, aj.user_id, aj.stroke_type, u.email 
-- FROM public.analysis_jobs aj 
-- LEFT JOIN public.users u ON aj.user_id = u.id 
-- LIMIT 10;
