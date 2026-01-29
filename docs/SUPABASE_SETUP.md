# Supabase Setup Guide for StrikeSense

This guide walks you through setting up Supabase for video storage and analysis results.

## 1. Storage Buckets

Go to **Storage** in your Supabase dashboard and create these buckets:

### Create Buckets (via Dashboard)
1. Click **"New bucket"**
2. Create bucket: `videos` (for uploaded videos)
   - Public: **Yes**
3. Create bucket: `analysis-results` (for output videos + frames)
   - Public: **Yes**

### Or via SQL Editor:
```sql
-- Note: Buckets are typically created via Dashboard, but you can insert directly
INSERT INTO storage.buckets (id, name, public)
VALUES 
  ('videos', 'videos', true),
  ('analysis-results', 'analysis-results', true)
ON CONFLICT (id) DO NOTHING;
```

---

## 2. Storage Policies (IMPORTANT)

Go to **Storage → Policies** for each bucket and add these policies.

### For `videos` bucket:

```sql
-- Allow anyone to upload videos (for demo; restrict in production)
CREATE POLICY "Allow public uploads to videos"
ON storage.objects
FOR INSERT
TO public
WITH CHECK (bucket_id = 'videos');

-- Allow anyone to read videos
CREATE POLICY "Allow public read access to videos"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'videos');

-- Allow deletion (for cleanup)
CREATE POLICY "Allow public delete from videos"
ON storage.objects
FOR DELETE
TO public
USING (bucket_id = 'videos');
```

### For `analysis-results` bucket:

```sql
-- Allow service role to upload results (from RunPod)
CREATE POLICY "Allow public uploads to analysis-results"
ON storage.objects
FOR INSERT
TO public
WITH CHECK (bucket_id = 'analysis-results');

-- Allow anyone to read results
CREATE POLICY "Allow public read access to analysis-results"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'analysis-results');

-- Allow deletion
CREATE POLICY "Allow public delete from analysis-results"
ON storage.objects
FOR DELETE
TO public
USING (bucket_id = 'analysis-results');
```

---

## 3. Database Table for Analysis Jobs (Optional but Recommended)

Go to **SQL Editor** and run:

```sql
-- Table to track analysis jobs with FULL input and output storage
CREATE TABLE IF NOT EXISTS analysis_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- User Inputs (stored when job is created)
  input_video_url TEXT,                    -- Original video URL from Supabase
  stroke_type TEXT NOT NULL DEFAULT 'serve', -- serve, dink, groundstroke, overhead
  input_json JSONB,                        -- Full input: {video_url, stroke_type, crop_region, step, target_point}
  
  -- Legacy (kept for compatibility)
  video_url TEXT,
  crop_region TEXT,
  
  -- Status
  status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
  error_message TEXT,
  
  -- Output (populated after analysis)
  result_video_url TEXT,                   -- Annotated video URL
  result_json JSONB,                       -- Full output: {frames, summary, metrics, injury_risks}
  frames_folder TEXT,                      -- Path in storage (deprecated - frames not stored separately)
  llm_response TEXT,                       -- AI coaching feedback (generated once and cached)
  
  -- Metadata
  processing_time_sec FLOAT,
  total_frames INT
);

-- Index for faster queries
CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_created ON analysis_jobs(created_at DESC);

-- Enable Row Level Security (optional for public access)
ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;

-- Allow public access (for demo; restrict in production)
CREATE POLICY "Allow public access to analysis_jobs"
ON analysis_jobs
FOR ALL
TO public
USING (true)
WITH CHECK (true);

-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_analysis_jobs
BEFORE UPDATE ON analysis_jobs
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();
```

---

## 4. Get Your API Keys

Go to **Settings → API** and note:

| Key | Where to Use |
|-----|--------------|
| `Project URL` | Both Next.js and Python |
| `anon (public)` key | Next.js frontend |
| `service_role` key | Python backend (RunPod) - **KEEP SECRET** |

---

## 5. Environment Variables

### For Next.js (Vercel) - `.env.local`:
```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...your-anon-key
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_ENDPOINT_ID=your-endpoint-id
```

### For Python (RunPod) - Set in RunPod environment:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...your-service-role-key
```

---

## 6. Complete SQL Script (Copy-Paste All at Once)

```sql
-- =============================================
-- SUPABASE SETUP FOR STRIKESENSE
-- Run this in SQL Editor (all at once)
-- =============================================

-- 1. Create storage buckets (if not via UI)
INSERT INTO storage.buckets (id, name, public)
VALUES 
  ('videos', 'videos', true),
  ('analysis-results', 'analysis-results', true)
ON CONFLICT (id) DO NOTHING;

-- 2. Storage policies for 'videos' bucket
CREATE POLICY "videos_insert" ON storage.objects
FOR INSERT TO public
WITH CHECK (bucket_id = 'videos');

CREATE POLICY "videos_select" ON storage.objects
FOR SELECT TO public
USING (bucket_id = 'videos');

CREATE POLICY "videos_delete" ON storage.objects
FOR DELETE TO public
USING (bucket_id = 'videos');

-- 3. Storage policies for 'analysis-results' bucket
CREATE POLICY "results_insert" ON storage.objects
FOR INSERT TO public
WITH CHECK (bucket_id = 'analysis-results');

CREATE POLICY "results_select" ON storage.objects
FOR SELECT TO public
USING (bucket_id = 'analysis-results');

CREATE POLICY "results_delete" ON storage.objects
FOR DELETE TO public
USING (bucket_id = 'analysis-results');

-- 4. Analysis jobs table
CREATE TABLE IF NOT EXISTS analysis_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  video_url TEXT NOT NULL,
  stroke_type TEXT NOT NULL DEFAULT 'serve',
  crop_region TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  error_message TEXT,
  result_video_url TEXT,
  result_json JSONB,
  frames_folder TEXT,
  processing_time_sec FLOAT,
  total_frames INT,
  llm_response TEXT                  -- AI coaching feedback (generated once, cached)
);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created ON analysis_jobs(created_at DESC);

ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "analysis_jobs_all" ON analysis_jobs
FOR ALL TO public
USING (true) WITH CHECK (true);

-- 5. Auto-update trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_analysis_jobs ON analysis_jobs;
CREATE TRIGGER trigger_update_analysis_jobs
BEFORE UPDATE ON analysis_jobs
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Done!
SELECT 'Supabase setup complete!' as message;
```

---

## 7. Test Your Setup

After running the SQL, verify:

1. **Storage → Buckets**: You should see `videos` and `analysis-results`
2. **Table Editor → analysis_jobs**: Table should exist
3. **Try uploading** a test file to `videos` bucket via Dashboard

---

## Quick Reference: Storage URLs

After upload, files are accessible at:
```
https://your-project.supabase.co/storage/v1/object/public/videos/{filename}
https://your-project.supabase.co/storage/v1/object/public/analysis-results/{path}
```

