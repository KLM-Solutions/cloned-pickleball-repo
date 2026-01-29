/**
 * Types for biomechanics analysis
 */

// Landmark from pose detection
export interface Landmark {
  name: string;
  x: number;      // 0-1 normalized (left to right)
  y: number;      // 0-1 normalized (top to bottom)
  z: number;      // depth (negative = closer)
  visibility: number; // 0-1
}

// Raw frame from Python track.py
export interface RawFrame {
  frameIdx: number;
  timestampSec: number;
  bbox: number[];
  confidence: number;
  track_id: number;
  landmarks: Landmark[] | null;
}

// Calculated metrics for a frame
export interface FrameMetrics {
  // Joint angles
  right_elbow_flexion: number | null;
  left_elbow_flexion: number | null;
  right_knee_flexion: number | null;
  left_knee_flexion: number | null;
  right_shoulder_abduction: number | null;
  left_shoulder_abduction: number | null;
  hip_rotation_deg: number | null;
  
  // Position checks
  wrist_above_waist: boolean;
  wrist_above_head: boolean;
  
  // Raw positions for reference
  nose_y: number | null;
  right_wrist_y: number | null;
  right_hip_y: number | null;
}

// Risk level
export type RiskLevel = 'low' | 'medium' | 'high';

// Injury risk detail
export interface InjuryRisk {
  type: string;
  severity: RiskLevel;
  angle?: number;
  message: string;
  recommendation: string;
}

// Analyzed frame output
export interface AnalyzedFrame {
  frameIdx: number;
  timestampSec: number;
  bbox: number[];
  confidence: number;
  track_id: number;
  metrics: FrameMetrics;
  injury_risk: RiskLevel;
  injury_risks: InjuryRisk[];
  feedback: string[];
}

// Analysis summary
export interface AnalysisSummary {
  total_frames: number;
  analyzed_frames: number;
  fps: number;
  duration_sec: number;
  stroke_type: string;
  overall_risk: RiskLevel;
  risk_percentages: {
    shoulder_overuse: number;
    poor_kinetic_chain: number;
    knee_stress: number;
  };
}

// Complete analysis result
export interface AnalysisResult {
  job_id: string;
  stroke_type: string;
  videoUrl: string;
  frames: AnalyzedFrame[];
  summary: AnalysisSummary;
  processingTime?: number;
}

// Stroke type
export type StrokeType = 'serve' | 'dink' | 'groundstroke' | 'overhead' | 'volley';

