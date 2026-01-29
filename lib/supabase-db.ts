/**
 * Supabase Database Client for Analysis History
 * 
 * Handles queries to the analysis_jobs table for:
 * - Fetching analysis history
 * - Getting individual job details
 * - Storing LLM responses
 */

import { createClient } from "@supabase/supabase-js";

// Types for analysis jobs
export interface AnalysisJob {
  id: string;
  created_at: string;
  updated_at: string;
  video_url: string;
  stroke_type: string;
  crop_region: string | null;
  status: "pending" | "processing" | "completed" | "failed";
  error_message: string | null;
  result_video_url: string | null;
  result_json: any | null;
  input_json: any | null;
  processing_time_sec: number | null;
  total_frames: number | null;
  llm_response: string | null;
  user_id: string | null;
}

// Create Supabase client
function getSupabaseClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    throw new Error("Missing Supabase environment variables");
  }

  return createClient(supabaseUrl, supabaseKey);
}

/**
 * Fetch analysis history (most recent first)
 * @param limit - Maximum number of results to return
 * @param userId - Optional Clerk user ID to filter results
 */
export async function getAnalysisHistory(limit: number = 20, userId?: string): Promise<AnalysisJob[]> {
  try {
    const supabase = getSupabaseClient();

    let query = supabase
      .from("analysis_jobs")
      .select("*")
      .order("created_at", { ascending: false });

    if (userId) {
      query = query.eq("user_id", userId);
    }

    const { data, error } = await query.limit(limit);

    if (error) {
      console.error("Error fetching history:", error);
      return [];
    }

    return data || [];
  } catch (error) {
    console.error("Failed to fetch analysis history:", error);
    return [];
  }
}

/**
 * Fetch a single analysis job by ID
 * @param jobId - The job ID to fetch
 * @param userId - Optional Clerk user ID to verify ownership
 */
export async function getAnalysisJob(jobId: string, userId?: string): Promise<AnalysisJob | null> {
  try {
    const supabase = getSupabaseClient();

    let query = supabase
      .from("analysis_jobs")
      .select("*")
      .eq("id", jobId);

    if (userId) {
      query = query.eq("user_id", userId);
    }

    const { data, error } = await query.single();

    if (error) {
      console.error("Error fetching job:", error);
      return null;
    }

    return data;
  } catch (error) {
    console.error("Failed to fetch analysis job:", error);
    return null;
  }
}

/**
 * Fetch completed analyses only
 * @param limit - Maximum number of results to return
 * @param userId - Optional Clerk user ID to filter results
 */
export async function getCompletedAnalyses(limit: number = 20, userId?: string): Promise<AnalysisJob[]> {
  try {
    const supabase = getSupabaseClient();

    let query = supabase
      .from("analysis_jobs")
      .select("*")
      .eq("status", "completed")
      .order("created_at", { ascending: false });

    if (userId) {
      query = query.eq("user_id", userId);
    }

    const { data, error } = await query.limit(limit);

    if (error) {
      console.error("Error fetching completed analyses:", error);
      return [];
    }

    return data || [];
  } catch (error) {
    console.error("Failed to fetch completed analyses:", error);
    return [];
  }
}

/**
 * Fetch all analyses (including processing, pending, completed, failed)
 * @param limit - Maximum number of results to return
 * @param userId - Optional Clerk user ID to filter results
 */
export async function getAllAnalyses(limit: number = 50, userId?: string): Promise<AnalysisJob[]> {
  try {
    const supabase = getSupabaseClient();

    let query = supabase
      .from("analysis_jobs")
      .select("*")
      .order("created_at", { ascending: false });

    if (userId) {
      query = query.eq("user_id", userId);
    }

    const { data, error } = await query.limit(limit);

    if (error) {
      console.error("Error fetching all analyses:", error);
      return [];
    }

    return data || [];
  } catch (error) {
    console.error("Failed to fetch all analyses:", error);
    return [];
  }
}

/**
 * Get analysis count by status
 * @param userId - Optional Clerk user ID to filter results
 */
export async function getAnalysisCounts(userId?: string): Promise<Record<string, number>> {
  try {
    const supabase = getSupabaseClient();

    let query = supabase
      .from("analysis_jobs")
      .select("status");

    if (userId) {
      query = query.eq("user_id", userId);
    }

    const { data, error } = await query;

    if (error) {
      console.error("Error fetching counts:", error);
      return { completed: 0, pending: 0, failed: 0 };
    }

    const counts: Record<string, number> = {
      completed: 0,
      pending: 0,
      processing: 0,
      failed: 0,
    };

    (data || []).forEach((job: { status: string }) => {
      counts[job.status] = (counts[job.status] || 0) + 1;
    });

    return counts;
  } catch (error) {
    console.error("Failed to fetch counts:", error);
    return { completed: 0, pending: 0, failed: 0 };
  }
}

/**
 * Format date for display
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
  });
}

/**
 * Get stroke type display info
 */
export function getStrokeInfo(strokeType: string): { icon: string; color: string; label: string } {
  const strokeInfo: Record<string, { icon: string; color: string; label: string }> = {
    serve: { icon: "🎾", color: "emerald", label: "Serve" },
    dink: { icon: "🤏", color: "violet", label: "Dink" },
    groundstroke: { icon: "💪", color: "orange", label: "Drive" },
    overhead: { icon: "⚡", color: "blue", label: "Overhead" },
  };

  return strokeInfo[strokeType] || { icon: "🏓", color: "slate", label: strokeType };
}

// ============================================
// USER PROFILE FUNCTIONS
// ============================================

export interface UserProfile {
  user_id: string;
  name: string | null;
  skill_level: string | null;
  dominant_hand: "left" | "right" | null;
  injury_history: string | null;
  created_at: string;
  updated_at: string;
}

/**
 * Get user profile by Clerk user ID
 */
export async function getUserProfile(userId: string): Promise<UserProfile | null> {
  try {
    const supabase = getSupabaseClient();
    const { data, error } = await supabase
      .from("user_profiles")
      .select("*")
      .eq("user_id", userId)
      .single();

    if (error && error.code !== "PGRST116") {
      console.error("Error fetching profile:", error);
      return null;
    }

    return data;
  } catch (error) {
    console.error("Failed to fetch user profile:", error);
    return null;
  }
}

/**
 * Upsert user profile
 */
export async function upsertUserProfile(
  userId: string,
  profile: Partial<Omit<UserProfile, "user_id" | "created_at" | "updated_at">>
): Promise<UserProfile | null> {
  try {
    const supabase = getSupabaseClient();
    const { data, error } = await supabase
      .from("user_profiles")
      .upsert(
        { user_id: userId, ...profile, updated_at: new Date().toISOString() },
        { onConflict: "user_id" }
      )
      .select()
      .single();

    if (error) {
      console.error("Error upserting profile:", error);
      return null;
    }

    return data;
  } catch (error) {
    console.error("Failed to upsert user profile:", error);
    return null;
  }
}

// ============================================
// SESSION MANAGEMENT FUNCTIONS
// ============================================

/**
 * Delete an analysis job
 */
export async function deleteAnalysisJob(jobId: string, userId: string): Promise<boolean> {
  try {
    const supabase = getSupabaseClient();
    const { error } = await supabase
      .from("analysis_jobs")
      .delete()
      .eq("id", jobId)
      .eq("user_id", userId);

    if (error) {
      console.error("Error deleting job:", error);
      return false;
    }

    return true;
  } catch (error) {
    console.error("Failed to delete analysis job:", error);
    return false;
  }
}

/**
 * Rename an analysis job (update custom_name field)
 */
export async function renameAnalysisJob(
  jobId: string,
  userId: string,
  customName: string
): Promise<boolean> {
  try {
    const supabase = getSupabaseClient();
    const { error } = await supabase
      .from("analysis_jobs")
      .update({ custom_name: customName })
      .eq("id", jobId)
      .eq("user_id", userId);

    if (error) {
      console.error("Error renaming job:", error);
      return false;
    }

    return true;
  } catch (error) {
    console.error("Failed to rename analysis job:", error);
    return false;
  }
}
