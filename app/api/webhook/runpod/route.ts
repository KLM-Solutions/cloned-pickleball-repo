/**
 * RunPod Webhook Endpoint
 * 
 * This endpoint receives notifications from RunPod when a job completes.
 * RunPod sends a POST request with the job result to this URL.
 * 
 * Flow:
 * 1. Python sends raw frames (with landmarks)
 * 2. TypeScript analyzes frames: angles, risk, feedback
 * 3. Generate LLM coaching response
 * 4. Store complete analysis + LLM response in Supabase
 */

import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import { analyzeFrames, RawFrame } from "@/lib/analysis";
import OpenAI from "openai";

// Use service key for webhook (server-side only)
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Issue type labels
const ISSUE_LABELS: Record<string, string> = {
  shoulder_overuse: 'Shoulder Overuse Risk',
  shoulder_over_rotation: 'Shoulder Over-Rotation',
  shoulder_under_rotation: 'Shoulder Under-Rotation',
  poor_kinetic_chain: 'Poor Kinetic Chain',
  insufficient_hip_rotation: 'Insufficient Hip Rotation',
  knee_stress: 'Knee Stress',
  excessive_knee_bend: 'Excessive Knee Bend',
  insufficient_knee_bend: 'Insufficient Knee Bend',
  elbow_form: 'Elbow Position Issue',
  elbow_strain: 'Elbow Strain Risk',
};

// Stroke context
const STROKE_CONTEXT: Record<string, string> = {
  serve: 'The player is performing a pickleball serve. Key mechanics include: proper ball toss, hip rotation for power, shoulder position under 140°, and follow-through.',
  dink: 'The player is performing a dink shot at the kitchen line. Key mechanics include: soft hands, low paddle position, bent knees, and controlled touch.',
  groundstroke: 'The player is performing a drive/groundstroke. Key mechanics include: hip rotation (45°+), early preparation, contact point in front, and weight transfer.',
  overhead: 'The player is performing an overhead smash. Key mechanics include: positioning under the ball, full extension, and avoiding excessive shoulder abduction (>140°).',
  volley: 'The player is performing a volley. Key mechanics include: ready position, compact swing, firm wrist, and quick recovery.',
};

interface FrameIssue {
  timestampSec: number;
  issues: string[];
  severity: string;
  metrics?: {
    hip_rotation_deg?: number;
    right_shoulder_abduction?: number;
    right_knee_flexion?: number;
    right_elbow_flexion?: number;
  };
}

/**
 * Filter frames to get one representative frame per second with issues
 */
function filterFramesPerSecond(frames: any[]): FrameIssue[] {
  if (!frames || frames.length === 0) return [];

  const framesBySecond: Map<number, FrameIssue> = new Map();

  frames.forEach(frame => {
    const second = Math.floor(frame.timestampSec);
    const issues: string[] = [];

    if (frame.injury_risks && Array.isArray(frame.injury_risks)) {
      frame.injury_risks.forEach((risk: any) => {
        if (risk.type) {
          issues.push(ISSUE_LABELS[risk.type] || risk.type);
        }
      });
    }

    if (issues.length === 0) return;

    const existing = framesBySecond.get(second);
    const currentSeverity = frame.injury_risk === 'high' ? 3 : frame.injury_risk === 'medium' ? 2 : 1;
    const existingSeverity = existing?.severity === 'high' ? 3 : existing?.severity === 'medium' ? 2 : 1;

    if (!existing || currentSeverity > existingSeverity || issues.length > (existing.issues?.length || 0)) {
      framesBySecond.set(second, {
        timestampSec: frame.timestampSec,
        issues,
        severity: frame.injury_risk || 'low',
        metrics: {
          hip_rotation_deg: frame.metrics?.hip_rotation_deg,
          right_shoulder_abduction: frame.metrics?.right_shoulder_abduction,
          right_knee_flexion: frame.metrics?.right_knee_flexion,
          right_elbow_flexion: frame.metrics?.right_elbow_flexion,
        }
      });
    }
  });

  return Array.from(framesBySecond.values()).sort((a, b) => a.timestampSec - b.timestampSec);
}

/**
 * Build prompt for OpenAI
 */
function buildPrompt(strokeType: string, filteredIssues: FrameIssue[], summary: any): string {
  const context = STROKE_CONTEXT[strokeType] || STROKE_CONTEXT.groundstroke;

  const issuesList = filteredIssues.map(fi => {
    const metricsStr = fi.metrics ?
      `(Hip: ${fi.metrics.hip_rotation_deg?.toFixed(0) || '--'}°, Shoulder: ${fi.metrics.right_shoulder_abduction?.toFixed(0) || '--'}°, Knee: ${fi.metrics.right_knee_flexion?.toFixed(0) || '--'}°, Elbow: ${fi.metrics.right_elbow_flexion?.toFixed(0) || '--'}°)` : '';
    return `- At ${fi.timestampSec.toFixed(1)}s [${fi.severity.toUpperCase()}]: ${fi.issues.join(', ')} ${metricsStr}`;
  }).join('\n');

  const issueCount: Record<string, number> = {};
  filteredIssues.forEach(fi => {
    fi.issues.forEach(issue => {
      issueCount[issue] = (issueCount[issue] || 0) + 1;
    });
  });

  const issueSummary = Object.entries(issueCount)
    .sort((a, b) => b[1] - a[1])
    .map(([issue, count]) => `${issue}: ${count} occurrences`)
    .join('\n');

  return `You are an expert pickleball coach analyzing a player's ${strokeType} technique.

## Context
${context}

## Session Summary
- Total frames analyzed: ${summary?.total_frames || 'N/A'}
- Duration: ${summary?.duration_sec?.toFixed(1) || 'N/A'} seconds
- FPS: ${summary?.fps || 30}
- Overall risk level: ${summary?.overall_risk || 'N/A'}

## Detected Issues Summary
${issueSummary || 'No major issues detected'}

## Frame-by-Frame Issues (one per second)
${issuesList || 'No frame-specific issues detected'}

## Your Task
Based on this analysis, provide a comprehensive coaching report in markdown format. Include:

1. **Overall Assessment** (2-3 sentences summary)
2. **Key Strengths** (bullet points of what's working well)
3. **Priority Areas for Improvement** (ranked by importance, with specific angles/metrics where relevant)
4. **Drill Recommendations** (3 specific drills to address the issues)
5. **Injury Prevention Notes** (if any high-risk movements detected)

Be specific, actionable, and encouraging. Reference the actual timestamps and metrics when relevant.
Format your response in clean markdown with headers (##), bullet points, and emphasis where appropriate.`;
}

/**
 * Generate LLM response
 */
async function generateLLMResponse(strokeType: string, frames: any[], summary: any): Promise<string | null> {
  if (!process.env.OPENAI_API_KEY) {
    console.log("OPENAI_API_KEY not configured, skipping LLM generation");
    return null;
  }

  try {
    const filteredIssues = filterFramesPerSecond(frames);
    const prompt = buildPrompt(strokeType, filteredIssues, summary);

    console.log(`Generating LLM response for ${filteredIssues.length} filtered issues...`);

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "You are an expert pickleball coach with deep knowledge of biomechanics, injury prevention, and technique optimization. You provide actionable, encouraging, and specific feedback to help players improve."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.7,
      max_tokens: 1500,
    });

    const response = completion.choices[0]?.message?.content || null;
    console.log(`LLM response generated: ${response?.length || 0} characters`);
    return response;

  } catch (error: any) {
    console.error("LLM generation failed:", error.message);
    return null;
  }
}

export async function POST(request: Request): Promise<NextResponse> {
  try {
    const payload = await request.json();

    console.log("\n");
    console.log("╔══════════════════════════════════════════════════════════════════╗");
    console.log("║              RUNPOD WEBHOOK - FULL PAYLOAD RECEIVED              ║");
    console.log("╚══════════════════════════════════════════════════════════════════╝");
    console.log("Timestamp:", new Date().toISOString());
    console.log("");

    // Log the COMPLETE raw payload
    console.log("┌─────────────────────────────────────────────────────────────────┐");
    console.log("│  COMPLETE RAW PAYLOAD (JSON)                                    │");
    console.log("└─────────────────────────────────────────────────────────────────┘");
    console.log(JSON.stringify(payload, null, 2));
    console.log("");

    // Log structured breakdown
    console.log("┌─────────────────────────────────────────────────────────────────┐");
    console.log("│  PAYLOAD BREAKDOWN                                              │");
    console.log("└─────────────────────────────────────────────────────────────────┘");
    console.log("Status:", payload.status);
    console.log("RunPod Job ID:", payload.id);
    console.log("Delay Time:", payload.delayTime);
    console.log("Execution Time:", payload.executionTime);
    console.log("Error:", payload.error || "None");
    console.log("Top-level Keys:", Object.keys(payload).join(", "));
    console.log("");

    // Log input section
    if (payload.input) {
      console.log("┌─────────────────────────────────────────────────────────────────┐");
      console.log("│  INPUT SECTION                                                  │");
      console.log("└─────────────────────────────────────────────────────────────────┘");
      console.log(JSON.stringify(payload.input, null, 2));
      console.log("");
    }

    // Log output section
    if (payload.output) {
      console.log("┌─────────────────────────────────────────────────────────────────┐");
      console.log("│  OUTPUT SECTION                                                 │");
      console.log("└─────────────────────────────────────────────────────────────────┘");
      console.log("Output Keys:", Object.keys(payload.output).join(", "));
      console.log("Output Status:", payload.output.status);
      console.log("Output Job ID:", payload.output.job_id);
      console.log("Stroke Type:", payload.output.stroke_type);
      console.log("Video URL:", payload.output.video_url);
      console.log("Processing Time (sec):", payload.output.processing_time_sec);
      console.log("Error:", payload.output.error || "None");
      console.log("Frames Count:", payload.output.frames?.length || 0);

      // Log first few frames if available
      if (payload.output.frames && payload.output.frames.length > 0) {
        console.log("");
        console.log("First 3 frames sample:");
        console.log(JSON.stringify(payload.output.frames.slice(0, 3), null, 2));
      }
      console.log("");
    }

    console.log("═══════════════════════════════════════════════════════════════════");
    console.log("");

    const {
      id: runpodJobId,
      status,
      output,
      error: jobError,
    } = payload;

    const jobId = output?.job_id || payload.input?.job_id;
    const strokeType = output?.stroke_type || payload.input?.stroke_type || 'groundstroke';

    if (!jobId) {
      console.error("No job_id in webhook payload");
      console.log("Available in output:", output ? Object.keys(output) : 'no output');
      console.log("Available in input:", payload.input ? Object.keys(payload.input) : 'no input');
      return NextResponse.json({
        received: true,
        warning: "No job_id found"
      });
    }

    console.log("Internal Job ID:", jobId);
    console.log("Stroke Type:", strokeType);

    if (status === "COMPLETED" && output) {
      // Get raw frames from Python output
      const rawFrames: RawFrame[] = (output.frames || []).map((frame: any) => ({
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
        output.video_url || ''
      );

      analysisResult.processingTime = output.processing_time_sec;

      console.log(`Analysis complete. Overall risk: ${analysisResult.summary.overall_risk}`);

      // Generate LLM coaching response
      const llmResponse = await generateLLMResponse(
        strokeType,
        analysisResult.frames,
        analysisResult.summary
      );

      // Update job in database with full analysis + LLM response
      const { error: updateError } = await supabase
        .from("analysis_jobs")
        .update({
          status: "completed",
          result_video_url: output.video_url || null,
          result_json: analysisResult,
          llm_response: llmResponse,
          processing_time_sec: output.processing_time_sec,
          total_frames: rawFrames.length,
          updated_at: new Date().toISOString(),
        })
        .eq("id", jobId);

      if (updateError) {
        console.error("Failed to update job:", updateError);
        return NextResponse.json(
          { error: "Failed to update job", details: updateError.message },
          { status: 500 }
        );
      }

      console.log(`Job ${jobId} marked as COMPLETED (LLM: ${llmResponse ? 'generated' : 'skipped'})`);
      return NextResponse.json({
        received: true,
        status: "completed",
        job_id: jobId,
        llm_generated: !!llmResponse,
        summary: {
          total_frames: analysisResult.summary.total_frames,
          analyzed_frames: analysisResult.summary.analyzed_frames,
          overall_risk: analysisResult.summary.overall_risk,
        }
      });

    } else if (status === "FAILED") {
      const errorMessage = jobError || output?.error || "Job failed";

      const { error: updateError } = await supabase
        .from("analysis_jobs")
        .update({
          status: "failed",
          error_message: errorMessage,
          updated_at: new Date().toISOString(),
        })
        .eq("id", jobId);

      if (updateError) {
        console.error("Failed to update failed job:", updateError);
      }

      console.log(`Job ${jobId} marked as FAILED:`, errorMessage);
      return NextResponse.json({
        received: true,
        status: "failed",
        job_id: jobId,
        error: errorMessage
      });

    } else if (status === "IN_PROGRESS") {
      await supabase
        .from("analysis_jobs")
        .update({
          status: "processing",
          updated_at: new Date().toISOString(),
        })
        .eq("id", jobId);

      console.log(`Job ${jobId} is IN_PROGRESS`);
      return NextResponse.json({
        received: true,
        status: "processing",
        job_id: jobId
      });
    }

    console.log(`Job ${jobId} has status: ${status}`);
    return NextResponse.json({
      received: true,
      status: status,
      job_id: jobId
    });

  } catch (error: any) {
    console.error("Webhook error:", error);
    return NextResponse.json(
      { error: "Webhook processing failed", details: error.message },
      { status: 500 }
    );
  }
}

export async function GET(): Promise<NextResponse> {
  return NextResponse.json({
    status: "ok",
    endpoint: "RunPod Webhook",
    timestamp: new Date().toISOString()
  });
}
