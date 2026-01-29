/**
 * Joint angle calculations from pose landmarks
 * 
 * All angles are calculated using 3-point geometry:
 * - Point A, B, C form an angle at B
 * - Uses dot product formula: cos(θ) = (BA · BC) / (|BA| × |BC|)
 */

import { Landmark, FrameMetrics } from './types';

// Landmark name to index mapping (MediaPipe order)
const LANDMARK_INDEX: Record<string, number> = {
  'nose': 0,
  'left_eye_inner': 1,
  'left_eye': 2,
  'left_eye_outer': 3,
  'right_eye_inner': 4,
  'right_eye': 5,
  'right_eye_outer': 6,
  'left_ear': 7,
  'right_ear': 8,
  'mouth_left': 9,
  'mouth_right': 10,
  'left_shoulder': 11,
  'right_shoulder': 12,
  'left_elbow': 13,
  'right_elbow': 14,
  'left_wrist': 15,
  'right_wrist': 16,
  'left_pinky': 17,
  'right_pinky': 18,
  'left_index': 19,
  'right_index': 20,
  'left_thumb': 21,
  'right_thumb': 22,
  'left_hip': 23,
  'right_hip': 24,
  'left_knee': 25,
  'right_knee': 26,
  'left_ankle': 27,
  'right_ankle': 28,
  'left_heel': 29,
  'right_heel': 30,
  'left_foot_index': 31,
  'right_foot_index': 32,
};

/**
 * Get landmark by name from array
 */
function getLandmark(landmarks: Landmark[], name: string): Landmark | null {
  // Try to find by name first
  const byName = landmarks.find(lm => lm.name === name);
  if (byName) return byName;

  // Fallback to index
  const index = LANDMARK_INDEX[name];
  if (index !== undefined && landmarks[index]) {
    return landmarks[index];
  }

  return null;
}

/**
 * Calculate angle at point B formed by points A-B-C
 * Returns angle in degrees (0-180)
 */
function calculateAngle(
  a: Landmark,
  b: Landmark,
  c: Landmark
): number {
  // Vector BA
  const ba = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z,
  };

  // Vector BC
  const bc = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: c.z - b.z,
  };

  // Dot product
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;

  // Magnitudes
  const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
  const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);

  // Avoid division by zero
  if (magBA === 0 || magBC === 0) return 0;

  // Cosine of angle
  let cosAngle = dot / (magBA * magBC);

  // Clamp to [-1, 1] to avoid NaN from floating point errors
  cosAngle = Math.max(-1, Math.min(1, cosAngle));

  // Convert to degrees
  const angle = Math.acos(cosAngle) * (180 / Math.PI);

  return Math.round(angle * 10) / 10; // Round to 1 decimal
}

/**
 * Calculate hip rotation from hip positions using 3D depth
 * Uses X difference and Z difference (depth) to calculate rotation around Y-axis
 */
function calculateHipRotation(leftHip: Landmark, rightHip: Landmark): number {
  const dx = Math.abs(rightHip.x - leftHip.x);
  const dz = Math.abs(rightHip.z - leftHip.z); // Using Z depth

  if (dx === 0) return 90; // If hips are stacked vertically in X, it's 90 degrees rotation (side view)

  // atan(dz / dx) gives rotation relative to camera plane
  const rotation = Math.atan(dz / dx) * (180 / Math.PI);
  return Math.round(rotation * 10) / 10;
}

/**
 * Calculate all metrics from landmarks
 */
export function calculateMetrics(landmarks: Landmark[]): FrameMetrics {
  const metrics: FrameMetrics = {
    right_elbow_flexion: null,
    left_elbow_flexion: null,
    right_knee_flexion: null,
    left_knee_flexion: null,
    right_shoulder_abduction: null,
    left_shoulder_abduction: null,
    hip_rotation_deg: null,
    wrist_above_waist: false,
    wrist_above_head: false,
    nose_y: null,
    right_wrist_y: null,
    right_hip_y: null,
  };

  if (!landmarks || landmarks.length === 0) {
    return metrics;
  }

  // Get landmarks
  const nose = getLandmark(landmarks, 'nose');
  const leftShoulder = getLandmark(landmarks, 'left_shoulder');
  const rightShoulder = getLandmark(landmarks, 'right_shoulder');
  const leftElbow = getLandmark(landmarks, 'left_elbow');
  const rightElbow = getLandmark(landmarks, 'right_elbow');
  const leftWrist = getLandmark(landmarks, 'left_wrist');
  const rightWrist = getLandmark(landmarks, 'right_wrist');
  const leftHip = getLandmark(landmarks, 'left_hip');
  const rightHip = getLandmark(landmarks, 'right_hip');
  const leftKnee = getLandmark(landmarks, 'left_knee');
  const rightKnee = getLandmark(landmarks, 'right_knee');
  const leftAnkle = getLandmark(landmarks, 'left_ankle');
  const rightAnkle = getLandmark(landmarks, 'right_ankle');

  // Store raw positions
  if (nose) metrics.nose_y = Math.round(nose.y * 10000) / 10000;
  if (rightWrist) metrics.right_wrist_y = Math.round(rightWrist.y * 10000) / 10000;
  if (rightHip) metrics.right_hip_y = Math.round(rightHip.y * 10000) / 10000;

  // Right elbow flexion (shoulder-elbow-wrist)
  if (rightShoulder && rightElbow && rightWrist) {
    metrics.right_elbow_flexion = calculateAngle(rightShoulder, rightElbow, rightWrist);
  }

  // Left elbow flexion
  if (leftShoulder && leftElbow && leftWrist) {
    metrics.left_elbow_flexion = calculateAngle(leftShoulder, leftElbow, leftWrist);
  }

  // Right knee flexion (hip-knee-ankle)
  if (rightHip && rightKnee && rightAnkle) {
    metrics.right_knee_flexion = calculateAngle(rightHip, rightKnee, rightAnkle);
  }

  // Left knee flexion
  if (leftHip && leftKnee && leftAnkle) {
    metrics.left_knee_flexion = calculateAngle(leftHip, leftKnee, leftAnkle);
  }

  // Right shoulder abduction (hip-shoulder-elbow)
  if (rightHip && rightShoulder && rightElbow) {
    metrics.right_shoulder_abduction = calculateAngle(rightHip, rightShoulder, rightElbow);
  }

  // Left shoulder abduction
  if (leftHip && leftShoulder && leftElbow) {
    metrics.left_shoulder_abduction = calculateAngle(leftHip, leftShoulder, leftElbow);
  }

  // Hip rotation
  if (leftHip && rightHip) {
    metrics.hip_rotation_deg = calculateHipRotation(leftHip, rightHip);
  }

  // Position checks (Y increases downward, so < means ABOVE)
  if (rightWrist && rightHip && leftHip) {
    const midHipY = (rightHip.y + leftHip.y) / 2;
    metrics.wrist_above_waist = rightWrist.y < midHipY;
  }

  if (rightWrist && nose) {
    metrics.wrist_above_head = rightWrist.y < nose.y;
  }

  return metrics;
}

export { getLandmark, calculateAngle, calculateHipRotation };

