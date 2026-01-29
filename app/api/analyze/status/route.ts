/**
 * Job Status API Route
 * 
 * Polls the status of an analysis job from Supabase.
 * Used by frontend to check when webhook-triggered jobs complete.
 */

import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function GET(request: Request): Promise<NextResponse> {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get("job_id");

    if (!jobId) {
      return NextResponse.json(
        { error: "Missing job_id parameter" },
        { status: 400 }
      );
    }

    // Get job from database
    const { data: job, error } = await supabase
      .from("analysis_jobs")
      .select("*")
      .eq("id", jobId)
      .single();

    if (error) {
      console.error("Failed to get job:", error);
      return NextResponse.json(
        { error: "Job not found", details: error.message },
        { status: 404 }
      );
    }

    // Return job status and data
    return NextResponse.json({
      job_id: job.id,
      status: job.status, // pending, processing, completed, failed
      stroke_type: job.stroke_type,
      created_at: job.created_at,
      updated_at: job.updated_at,
      error_message: job.error_message,
      processing_time_sec: job.processing_time_sec,
      total_frames: job.total_frames,
      // Include result only if completed
      ...(job.status === "completed" && {
        result: job.result_json,
        videoUrl: job.result_video_url,
        llm_response: job.llm_response,
      }),
    });

  } catch (error: any) {
    console.error("Status check error:", error);
    return NextResponse.json(
      { error: "Failed to check status", details: error.message },
      { status: 500 }
    );
  }
}

