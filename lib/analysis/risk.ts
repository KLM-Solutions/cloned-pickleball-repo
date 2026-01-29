/**
 * Injury risk detection based on biomechanics thresholds
 * 
 * Research-based thresholds:
 * - Shoulder abduction >140° = high risk (except overhead)
 * - Poor kinetic chain (no hip rotation) = elbow/back strain
 * - Excessive knee flexion >90° = patellar stress
 */

import { FrameMetrics, InjuryRisk, RiskLevel, StrokeType } from './types';

// Thresholds based on biomechanics research
const THRESHOLDS = {
  shoulder: {
    safe: 120,
    caution: 140,
    high_risk: 160,
  },
  knee: {
    safe_max: 90,  // Above this = stress
    ready_min: 20,
    ready_max: 45,
  },
  hip_rotation: {
    power_minimum: 30,  // Below this for power strokes = poor kinetic chain
  },
  elbow: {
    serve: { min: 90, max: 130 },
    groundstroke: { min: 120, max: 160 },
    dink: { min: 90, max: 110 },
    overhead: { min: 90, max: 170 },
    volley: { min: 90, max: 120 },
  },
};

/**
 * Detect injury risks for a single frame
 */
export function detectRisks(
  metrics: FrameMetrics,
  strokeType: StrokeType
): { risks: InjuryRisk[]; level: RiskLevel } {
  const risks: InjuryRisk[] = [];
  
  // 1. SHOULDER OVERUSE CHECK
  const shoulderAngle = metrics.right_shoulder_abduction || 0;
  
  if (strokeType !== 'overhead' && shoulderAngle > THRESHOLDS.shoulder.caution) {
    const severity: RiskLevel = shoulderAngle > THRESHOLDS.shoulder.high_risk ? 'high' : 'medium';
    
    risks.push({
      type: 'shoulder_overuse',
      severity,
      angle: shoulderAngle,
      message: `Shoulder abduction ${shoulderAngle}° exceeds safe range`,
      recommendation: 'Reduce shoulder abduction to <140°. Focus on hip rotation for power.',
    });
  }
  
  // 2. POOR KINETIC CHAIN (No hip rotation for power strokes)
  const hipRotation = metrics.hip_rotation_deg || 0;
  const isPowerStroke = ['groundstroke', 'serve', 'overhead'].includes(strokeType);
  
  if (isPowerStroke && hipRotation < THRESHOLDS.hip_rotation.power_minimum) {
    risks.push({
      type: 'poor_kinetic_chain',
      severity: 'medium',
      angle: hipRotation,
      message: `Insufficient hip rotation (${hipRotation}°)`,
      recommendation: 'Engage hips and core for power. Reduce arm strain.',
    });
  }
  
  // 3. KNEE STRESS (Excessive flexion)
  const kneeAngle = metrics.right_knee_flexion || 180;
  
  // Note: Lower angle = more bent. Check if angle is too low (too bent)
  if (kneeAngle < THRESHOLDS.knee.safe_max) {
    risks.push({
      type: 'knee_stress',
      severity: 'medium',
      angle: kneeAngle,
      message: `Deep knee flexion (${kneeAngle}°) detected`,
      recommendation: 'Avoid excessive squatting. Maintain athletic stance (knee angle 120-160°).',
    });
  }
  
  // 4. ELBOW STRAIN CHECK
  const elbowAngle = metrics.right_elbow_flexion || 0;
  const elbowRange = THRESHOLDS.elbow[strokeType] || THRESHOLDS.elbow.groundstroke;
  
  if (elbowAngle < elbowRange.min || elbowAngle > elbowRange.max) {
    risks.push({
      type: 'elbow_strain',
      severity: 'low',
      angle: elbowAngle,
      message: `Elbow angle (${elbowAngle}°) outside optimal range for ${strokeType}`,
      recommendation: `Optimal elbow angle for ${strokeType}: ${elbowRange.min}-${elbowRange.max}°`,
    });
  }
  
  // Determine overall risk level
  let level: RiskLevel = 'low';
  
  if (risks.some(r => r.severity === 'high')) {
    level = 'high';
  } else if (risks.some(r => r.severity === 'medium')) {
    level = 'medium';
  }
  
  return { risks, level };
}

/**
 * Calculate risk percentages for session summary
 */
export function calculateRiskPercentages(
  frames: { injury_risks: InjuryRisk[] }[]
): { shoulder_overuse: number; poor_kinetic_chain: number; knee_stress: number } {
  const totalFrames = frames.length;
  
  if (totalFrames === 0) {
    return { shoulder_overuse: 0, poor_kinetic_chain: 0, knee_stress: 0 };
  }
  
  let shoulderCount = 0;
  let kineticChainCount = 0;
  let kneeCount = 0;
  
  frames.forEach(frame => {
    frame.injury_risks.forEach(risk => {
      if (risk.type === 'shoulder_overuse') shoulderCount++;
      if (risk.type === 'poor_kinetic_chain') kineticChainCount++;
      if (risk.type === 'knee_stress') kneeCount++;
    });
  });
  
  return {
    shoulder_overuse: Math.round((shoulderCount / totalFrames) * 100 * 10) / 10,
    poor_kinetic_chain: Math.round((kineticChainCount / totalFrames) * 100 * 10) / 10,
    knee_stress: Math.round((kneeCount / totalFrames) * 100 * 10) / 10,
  };
}

/**
 * Determine overall session risk level
 */
export function getOverallRisk(
  percentages: { shoulder_overuse: number; poor_kinetic_chain: number; knee_stress: number }
): RiskLevel {
  const { shoulder_overuse, poor_kinetic_chain, knee_stress } = percentages;
  
  // High if shoulder overuse > 10% or multiple issues > 20%
  if (shoulder_overuse > 10) return 'high';
  if (poor_kinetic_chain > 20 && knee_stress > 15) return 'high';
  
  // Medium if any significant issue
  if (shoulder_overuse > 5 || poor_kinetic_chain > 15 || knee_stress > 10) {
    return 'medium';
  }
  
  return 'low';
}

