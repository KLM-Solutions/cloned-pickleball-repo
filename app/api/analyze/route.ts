/**
 * Video Analysis API Route
 * 
 * Receives video URL and parameters, calls RunPod serverless worker with webhook,
 * returns immediately. Results are delivered via webhook to /api/webhook/runpod.
 * 
 * Flow:
 * 1. Create job in DB (with user_id from Clerk)
 * 2. Call RunPod (async with webhook or sync)
 * 3. For sync: Run TypeScript analysis on raw frames, store in DB
 * 4. For async: Webhook handles analysis and storage
 * 
 * Note: User must exist in users table (created via Clerk webhook on signup)
 */

import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import { startAnalysis, runAnalysisSync } from "@/lib/runpod";
import {
  createAnalysisJob,
  updateJobToProcessing,
  updateJobCompleted,
  updateJobFailed
} from "@/lib/supabase";
import { analyzeFrames, RawFrame } from "@/lib/analysis";

export const maxDuration = 300; // This function can run for a maximum of 5 minutes

type StrokeType =
  | "serve"
  | "groundstroke"
  | "dink"
  | "overhead"
  | "footwork"
  | "overall";

// Get the webhook URL based on environment
function getWebhookUrl(): string {
  // Use VERCEL_URL for production, or construct from request
  const baseUrl = process.env.NEXT_PUBLIC_APP_URL
    || (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : null)
    || 'http://localhost:3000';

  return `${baseUrl}/api/webhook/runpod`;
}

export async function POST(request: Request): Promise<NextResponse> {
  let jobId: string | undefined;

  try {
    // Get authenticated user ID from Clerk
    // User should already exist in users table (created via Clerk webhook on signup)
    const { userId } = await auth();

    const body = await request.json();

    const {
      videoUrl,
      strokeType,
      cropRegion,
      targetPoint,
      step = 1, // Analyze every frame by default
      useWebhook = true, // Use webhook by default (async)
    } = body;

    // Validate required fields
    if (!videoUrl) {
      return NextResponse.json(
        { error: "Missing required field: videoUrl" },
        { status: 400 }
      );
    }

    if (!strokeType) {
      return NextResponse.json(
        { error: "Missing required field: strokeType" },
        { status: 400 }
      );
    }

    // Validate video URL (must be from Supabase or allowed origins)
    const allowedOrigins = [
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      "https://", // Allow any HTTPS for flexibility
    ].filter(Boolean);

    const isValidUrl = allowedOrigins.some(origin => videoUrl.startsWith(origin!));
    if (!isValidUrl) {
      return NextResponse.json(
        { error: "Invalid video URL origin" },
        { status: 400 }
      );
    }

    // 1. CREATE JOB IN DATABASE (with user ID if authenticated)
    console.log("Creating analysis job in database...", userId ? `for user: ${userId}` : "(anonymous)");
    jobId = await createAnalysisJob({
      videoUrl,
      strokeType,
      cropRegion,
      targetPoint,
      step: Number(step),
      userId: userId || undefined,
    });
    console.log("Job created:", jobId);

    // 2. Update job status to processing
    await updateJobToProcessing(jobId);

    // Build RunPod input
    const runpodInput = {
      video_url: videoUrl,
      stroke_type: strokeType as StrokeType,
      crop_region: cropRegion || undefined,
      target_point: targetPoint || undefined,
      step: Number(step),
      job_id: jobId,
    };

    // 3. Start analysis with webhook (async) or sync
    if (useWebhook) {
      // ASYNC MODE: Start job and return immediately
      // RunPod will call our webhook when complete
      const webhookUrl = getWebhookUrl();

      console.log("Starting RunPod analysis with webhook:", {
        job_id: jobId,
        stroke_type: strokeType,
        step,
        webhook_url: webhookUrl,
      });

      const runpodJobId = await startAnalysis(runpodInput, webhookUrl);

      console.log("RunPod job started:", runpodJobId);

      return NextResponse.json({
        status: "processing",
        job_id: jobId,
        runpod_job_id: runpodJobId,
        message: "Analysis started. Poll job status or wait for completion.",
      });

    } else {
      // SYNC MODE: Wait for completion (legacy)
      console.log("Starting RunPod analysis (sync mode):", {
        job_id: jobId,
        stroke_type: strokeType,
        step,
      });

      const result = await runAnalysisSync(runpodInput, 600000);

      // Convert raw frames from Python output
      const rawFrames: RawFrame[] = (result.frames || []).map((frame: any) => ({
        frameIdx: frame.frameIdx ?? frame.frame_idx ?? 0,
        timestampSec: frame.timestampSec ?? frame.timestamp_sec ?? 0,
        bbox: frame.bbox || [],
        confidence: frame.confidence ?? 1,
        track_id: frame.track_id ?? 0,
        landmarks: frame.landmarks || null,
      }));

      console.log(`Processing ${rawFrames.length} frames with TypeScript analysis...`);

      // Run TypeScript biomechanics analysis
      const analysisResult = analyzeFrames(
        rawFrames,
        strokeType,
        jobId,
        result.video_url || ''
      );

      // Add processing time from RunPod
      analysisResult.processingTime = result.processing_time_sec;

      console.log(`Analysis complete. Overall risk: ${analysisResult.summary.overall_risk}`);

      // Store results in database
      await updateJobCompleted(jobId, {
        resultVideoUrl: result.video_url || undefined,
        resultJson: analysisResult,
        processingTimeSec: result.processing_time_sec,
        totalFrames: rawFrames.length,
      });

      return NextResponse.json(analysisResult);
    }

  } catch (error: any) {
    console.error("Analysis error:", error);

    // Update job status to failed if we have a job ID
    if (jobId) {
      await updateJobFailed(jobId, error.message || String(error));
    }

    // Handle timeout
    if (error.name === 'TimeoutError' || error.message?.includes('timeout')) {
      return NextResponse.json(
        { error: "Analysis timed out. Try a shorter video or increase step." },
        { status: 504 }
      );
    }

    return NextResponse.json(
      {
        error: "Failed to analyze video",
        details: error.message || String(error),
      },
      { status: 500 }
    );
  }
}
