/**
 * Frame filtering and insights extraction based on stroke type
 * 
 * Filters frames to identify:
 * - Injury risk frames
 * - Frames needing improvement
 * - Key technique moments
 */

import { AnalyzedFrame, StrokeType, RiskLevel, InjuryRisk, FrameMetrics } from './types';

// Issue types
export type IssueType = 'injury_risk' | 'form_improvement' | 'technique_flaw' | 'good_form';

// Filtered frame with context
export interface FilteredFrame {
  frame: AnalyzedFrame;
  issues: FrameIssue[];
  category: IssueType;
  priority: number; // 1 = highest priority
}

// Detailed issue info
export interface FrameIssue {
  type: string;
  severity: RiskLevel;
  description: string;
  recommendation: string;
  metric?: string;
  value?: number;
  optimal?: { min: number; max: number };
}

// Filter summary
export interface FilterSummary {
  total_frames: number;
  injury_risk_frames: number;
  improvement_frames: number;
  good_form_frames: number;
  key_issues: KeyIssue[];
}

// Key issue summary
export interface KeyIssue {
  type: string;
  count: number;
  percentage: number;
  affected_frames: number[];
  recommendation: string;
}

// Stroke-specific thresholds
const STROKE_THRESHOLDS: Record<StrokeType, {
  shoulder_optimal: { min: number; max: number };
  elbow_optimal: { min: number; max: number };
  knee_optimal: { min: number; max: number };
  hip_rotation_min: number;
  key_positions: string[];
}> = {
  serve: {
    shoulder_optimal: { min: 90, max: 140 },
    elbow_optimal: { min: 90, max: 130 },
    knee_optimal: { min: 120, max: 160 },
    hip_rotation_min: 30,
    key_positions: ['trophy_position', 'contact_point', 'follow_through'],
  },
  groundstroke: {
    shoulder_optimal: { min: 60, max: 120 },
    elbow_optimal: { min: 120, max: 160 },
    knee_optimal: { min: 110, max: 150 },
    hip_rotation_min: 45,
    key_positions: ['backswing', 'contact_point', 'follow_through'],
  },
  dink: {
    shoulder_optimal: { min: 30, max: 80 },
    elbow_optimal: { min: 90, max: 120 },
    knee_optimal: { min: 100, max: 140 },
    hip_rotation_min: 10,
    key_positions: ['ready_position', 'contact_point', 'recovery'],
  },
  overhead: {
    shoulder_optimal: { min: 140, max: 170 },
    elbow_optimal: { min: 90, max: 170 },
    knee_optimal: { min: 130, max: 170 },
    hip_rotation_min: 30,
    key_positions: ['preparation', 'extension', 'contact'],
  },
  volley: {
    shoulder_optimal: { min: 40, max: 90 },
    elbow_optimal: { min: 90, max: 130 },
    knee_optimal: { min: 120, max: 160 },
    hip_rotation_min: 15,
    key_positions: ['ready_position', 'contact', 'recovery'],
  },
};

/**
 * Filter frames to find issues and improvements
 */
export function filterFramesForIssues(
  frames: AnalyzedFrame[],
  strokeType: StrokeType | string
): FilteredFrame[] {
  const stroke = (strokeType as StrokeType) || 'groundstroke';
  const thresholds = STROKE_THRESHOLDS[stroke] || STROKE_THRESHOLDS.groundstroke;
  const filteredFrames: FilteredFrame[] = [];

  frames.forEach(frame => {
    const issues: FrameIssue[] = [];
    const metrics = frame.metrics;

    // Skip frames without valid metrics
    if (!metrics.right_elbow_flexion && !metrics.right_shoulder_abduction && !metrics.right_knee_flexion) {
      return;
    }

    // 1. Check existing injury risks
    frame.injury_risks.forEach(risk => {
      issues.push({
        type: risk.type,
        severity: risk.severity,
        description: risk.message,
        recommendation: risk.recommendation,
        value: risk.angle,
      });
    });

    // 2. Check shoulder against stroke-specific optimal range
    const shoulderAngle = metrics.right_shoulder_abduction;
    if (shoulderAngle !== null) {
      if (shoulderAngle < thresholds.shoulder_optimal.min) {
        issues.push({
          type: 'shoulder_under_rotation',
          severity: 'medium',
          description: `Shoulder abduction (${Math.round(shoulderAngle)}°) below optimal for ${stroke}`,
          recommendation: `Increase shoulder abduction to ${thresholds.shoulder_optimal.min}-${thresholds.shoulder_optimal.max}°`,
          metric: 'shoulder_abduction',
          value: shoulderAngle,
          optimal: thresholds.shoulder_optimal,
        });
      } else if (shoulderAngle > thresholds.shoulder_optimal.max && stroke !== 'overhead') {
        issues.push({
          type: 'shoulder_over_rotation',
          severity: shoulderAngle > 160 ? 'high' : 'medium',
          description: `Shoulder abduction (${Math.round(shoulderAngle)}°) exceeds optimal for ${stroke}`,
          recommendation: `Reduce shoulder abduction to below ${thresholds.shoulder_optimal.max}°`,
          metric: 'shoulder_abduction',
          value: shoulderAngle,
          optimal: thresholds.shoulder_optimal,
        });
      }
    }

    // 3. Check elbow against stroke-specific optimal range
    const elbowAngle = metrics.right_elbow_flexion;
    if (elbowAngle !== null) {
      if (elbowAngle < thresholds.elbow_optimal.min || elbowAngle > thresholds.elbow_optimal.max) {
        issues.push({
          type: 'elbow_form',
          severity: 'low',
          description: `Elbow angle (${Math.round(elbowAngle)}°) outside optimal range for ${stroke}`,
          recommendation: `Adjust elbow angle to ${thresholds.elbow_optimal.min}-${thresholds.elbow_optimal.max}°`,
          metric: 'elbow_flexion',
          value: elbowAngle,
          optimal: thresholds.elbow_optimal,
        });
      }
    }

    // 4. Check knee flexion
    const kneeAngle = metrics.right_knee_flexion;
    if (kneeAngle !== null) {
      if (kneeAngle < thresholds.knee_optimal.min) {
        issues.push({
          type: 'excessive_knee_bend',
          severity: kneeAngle < 90 ? 'medium' : 'low',
          description: `Knee flexion (${Math.round(kneeAngle)}°) too deep for ${stroke}`,
          recommendation: `Maintain knee angle between ${thresholds.knee_optimal.min}-${thresholds.knee_optimal.max}°`,
          metric: 'knee_flexion',
          value: kneeAngle,
          optimal: thresholds.knee_optimal,
        });
      } else if (kneeAngle > thresholds.knee_optimal.max) {
        issues.push({
          type: 'insufficient_knee_bend',
          severity: 'low',
          description: `Knee too straight (${Math.round(kneeAngle)}°) for ${stroke}`,
          recommendation: `Bend knees more for better stability (${thresholds.knee_optimal.min}-${thresholds.knee_optimal.max}°)`,
          metric: 'knee_flexion',
          value: kneeAngle,
          optimal: thresholds.knee_optimal,
        });
      }
    }

    // 5. Check hip rotation for power strokes
    const hipRotation = metrics.hip_rotation_deg;
    if (hipRotation !== null && ['serve', 'groundstroke', 'overhead'].includes(stroke)) {
      if (hipRotation < thresholds.hip_rotation_min) {
        issues.push({
          type: 'insufficient_hip_rotation',
          severity: 'medium',
          description: `Hip rotation (${Math.round(hipRotation)}°) insufficient for power in ${stroke}`,
          recommendation: `Engage hips more. Target at least ${thresholds.hip_rotation_min}° rotation`,
          metric: 'hip_rotation',
          value: hipRotation,
          optimal: { min: thresholds.hip_rotation_min, max: 90 },
        });
      }
    }

    // Determine category and priority
    let category: IssueType = 'good_form';
    let priority = 4;

    if (issues.some(i => i.severity === 'high')) {
      category = 'injury_risk';
      priority = 1;
    } else if (issues.some(i => i.severity === 'medium')) {
      category = 'form_improvement';
      priority = 2;
    } else if (issues.some(i => i.severity === 'low')) {
      category = 'technique_flaw';
      priority = 3;
    }

    // Only add frames with issues or sample good frames
    if (issues.length > 0 || (frame.injury_risk === 'low' && Math.random() < 0.1)) {
      filteredFrames.push({
        frame,
        issues,
        category,
        priority,
      });
    }
  });

  // Sort by priority (most important first)
  return filteredFrames.sort((a, b) => a.priority - b.priority);
}

/**
 * Get key frames by issue type
 */
export function getKeyFramesByIssue(filteredFrames: FilteredFrame[]): Record<string, FilteredFrame[]> {
  const grouped: Record<string, FilteredFrame[]> = {};

  filteredFrames.forEach(ff => {
    ff.issues.forEach(issue => {
      if (!grouped[issue.type]) {
        grouped[issue.type] = [];
      }
      if (grouped[issue.type].length < 5) { // Limit to 5 frames per issue type
        grouped[issue.type].push(ff);
      }
    });
  });

  return grouped;
}

/**
 * Get summary of all issues
 */
export function getFilterSummary(
  frames: AnalyzedFrame[],
  filteredFrames: FilteredFrame[]
): FilterSummary {
  const issueCount: Record<string, { count: number; frames: number[] }> = {};

  filteredFrames.forEach(ff => {
    ff.issues.forEach(issue => {
      if (!issueCount[issue.type]) {
        issueCount[issue.type] = { count: 0, frames: [] };
      }
      issueCount[issue.type].count++;
      if (!issueCount[issue.type].frames.includes(ff.frame.frameIdx)) {
        issueCount[issue.type].frames.push(ff.frame.frameIdx);
      }
    });
  });

  const keyIssues: KeyIssue[] = Object.entries(issueCount)
    .map(([type, data]) => ({
      type,
      count: data.count,
      percentage: Math.round((data.frames.length / frames.length) * 100 * 10) / 10,
      affected_frames: data.frames.slice(0, 10),
      recommendation: getRecommendationForIssue(type),
    }))
    .sort((a, b) => b.count - a.count);

  return {
    total_frames: frames.length,
    injury_risk_frames: filteredFrames.filter(f => f.category === 'injury_risk').length,
    improvement_frames: filteredFrames.filter(f => f.category === 'form_improvement').length,
    good_form_frames: frames.length - filteredFrames.filter(f => f.category !== 'good_form').length,
    key_issues: keyIssues.slice(0, 5), // Top 5 issues
  };
}

/**
 * Get recommendation text for issue type
 * Uses enhanced coaching templates from recommendations.ts
 */
import { getQuickRecommendation, getEnhancedRecommendation, CoachingRecommendation } from './recommendations';

function getRecommendationForIssue(type: string): string {
  return getQuickRecommendation(type);
}

/**
 * Get full coaching recommendation with drill and benefit
 */
export function getFullRecommendation(
  issueType: string,
  strokeType?: StrokeType | string,
  severity?: 'high' | 'medium' | 'low'
): CoachingRecommendation {
  return getEnhancedRecommendation(issueType, strokeType, severity);
}

/**
 * Get top issues for quick display
 */
export function getTopIssues(
  filteredFrames: FilteredFrame[],
  limit: number = 3
): { issue: string; severity: RiskLevel; frameCount: number; firstFrame: AnalyzedFrame }[] {
  const issueMap: Record<string, { severity: RiskLevel; frames: AnalyzedFrame[] }> = {};

  filteredFrames.forEach(ff => {
    ff.issues.forEach(issue => {
      if (!issueMap[issue.type]) {
        issueMap[issue.type] = { severity: issue.severity, frames: [] };
      }
      if (issue.severity === 'high' ||
        (issue.severity === 'medium' && issueMap[issue.type].severity !== 'high')) {
        issueMap[issue.type].severity = issue.severity;
      }
      issueMap[issue.type].frames.push(ff.frame);
    });
  });

  return Object.entries(issueMap)
    .map(([issue, data]) => ({
      issue,
      severity: data.severity,
      frameCount: data.frames.length,
      firstFrame: data.frames[0],
    }))
    .sort((a, b) => {
      const severityOrder = { high: 0, medium: 1, low: 2 };
      return severityOrder[a.severity] - severityOrder[b.severity] || b.frameCount - a.frameCount;
    })
    .slice(0, limit);
}

