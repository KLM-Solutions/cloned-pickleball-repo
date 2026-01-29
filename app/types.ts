export type StrokeType =
    | "serve"
    | "groundstroke"
    | "dink"
    | "overhead"
    | "footwork"
    | "overall";

export type ApiFrame = {
    frameIndex: number;
    frame_idx?: number; // Backend snake_case
    timestampSec: number;
    frameFilename?: string; // Backend output
    bbox: [number, number, number, number];
    confidence: number;
    ball?: {
        bbox: [number, number, number, number];
        confidence: number;
    } | null;
    metrics?: {
        right_elbow_flexion?: number;
        right_shoulder_abduction?: number;
        right_knee_flexion?: number;
        hip_rotation_deg?: number;
        wrist_to_body_distance_norm?: number;
        injury_risk?: "low" | "medium" | "high";
        feedback?: string[];
        wrist_velocity_y?: number;
    };
};

export type StrokeEvent = {
    type: "serve" | "groundstroke" | "dink" | "overhead" | "volley";
    startFrame: number;
    start_frame?: number; // Backend snake_case
    endFrame: number;
    end_frame?: number;   // Backend snake_case
    startSec: number;
    endSec: number;
    confidence: number;
    // New Peak Velocity Fields
    peak_frame_idx?: number;
    peak_timestamp?: number;
    peak_velocity?: number;
};

export type AnalyzeResponse = {
    stroke_type: StrokeType;
    frames: ApiFrame[];
    strokes?: StrokeEvent[];
    videoUrl?: string | null;
    ballStats?: {
        avgSpeedKmh?: number;
        maxSpeedKmh?: number;
        totalDistanceMeters?: number;
    } | null;
    playerStats?: {
        totalDistanceMeters: number;
        avgSpeedKmh: number;
        trackedDurationSec: number;
    };
};
