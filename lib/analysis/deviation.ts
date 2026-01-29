/**
 * Deviation Scoring System
 * 
 * Calculates deviation scores for biomechanical parameters
 * Compares user's metrics to optimal ranges and generates improvement insights
 */

import { StrokeType, AnalyzedFrame, FrameMetrics } from './types';

// Deviation parameter with score and analysis
export interface DeviationParameter {
    key: string;
    label: string;
    userValue: number | null;
    optimalRange: { min: number; max: number };
    score: number;           // 0-100, higher = closer to optimal
    deviation: number;       // Actual distance from optimal (0 if within range)
    status: 'optimal' | 'warning' | 'critical';
    performanceImpact: 'high' | 'medium' | 'low';
    recommendation: string;
    impactWeight: number;    // Weight for overall score calculation
}

// Full deviation report for a session
export interface DeviationReport {
    strokeType: string;
    overallScore: number;
    parameters: DeviationParameter[];
    topDeviations: DeviationParameter[];  // Top 3 worst
    prioritizedImprovements: string[];     // Ordered recommendations
    summary: string;
}

// Parameter definitions with optimal ranges per stroke type
interface ParameterConfig {
    key: string;
    label: string;
    metricKey: keyof FrameMetrics;
    performanceImpact: 'high' | 'medium' | 'low';
    impactWeight: number;
    maxDeviation: number; // Max degrees before score = 0
    getOptimalRange: (strokeType: StrokeType) => { min: number; max: number };
    getRecommendation: (value: number, optimal: { min: number; max: number }) => string;
}

// Optimal ranges by stroke type
const OPTIMAL_RANGES: Record<StrokeType, {
    hip_rotation: { min: number; max: number };
    shoulder_abduction: { min: number; max: number };
    elbow_flexion: { min: number; max: number };
    knee_flexion: { min: number; max: number };
}> = {
    serve: {
        hip_rotation: { min: 30, max: 70 },
        shoulder_abduction: { min: 90, max: 140 },
        elbow_flexion: { min: 90, max: 130 },
        knee_flexion: { min: 120, max: 160 },
    },
    groundstroke: {
        hip_rotation: { min: 45, max: 75 },
        shoulder_abduction: { min: 60, max: 120 },
        elbow_flexion: { min: 120, max: 160 },
        knee_flexion: { min: 110, max: 150 },
    },
    dink: {
        hip_rotation: { min: 10, max: 30 },
        shoulder_abduction: { min: 30, max: 80 },
        elbow_flexion: { min: 90, max: 120 },
        knee_flexion: { min: 100, max: 140 },
    },
    overhead: {
        hip_rotation: { min: 30, max: 60 },
        shoulder_abduction: { min: 140, max: 170 },
        elbow_flexion: { min: 90, max: 170 },
        knee_flexion: { min: 130, max: 170 },
    },
    volley: {
        hip_rotation: { min: 15, max: 40 },
        shoulder_abduction: { min: 40, max: 90 },
        elbow_flexion: { min: 90, max: 130 },
        knee_flexion: { min: 120, max: 160 },
    },
};

// Parameter configurations
const PARAMETER_CONFIGS: ParameterConfig[] = [
    {
        key: 'hip_rotation',
        label: 'Hip Rotation',
        metricKey: 'hip_rotation_deg',
        performanceImpact: 'high',
        impactWeight: 1.5,
        maxDeviation: 40,
        getOptimalRange: (stroke) => OPTIMAL_RANGES[stroke].hip_rotation,
        getRecommendation: (value, optimal) => {
            if (value < optimal.min) {
                return `Increase hip rotation by ${Math.round(optimal.min - value)}° to generate more power from your core`;
            }
            return `Reduce hip rotation by ${Math.round(value - optimal.max)}° to maintain control`;
        },
    },
    {
        key: 'shoulder_abduction',
        label: 'Shoulder Position',
        metricKey: 'right_shoulder_abduction',
        performanceImpact: 'high',
        impactWeight: 1.4,
        maxDeviation: 50,
        getOptimalRange: (stroke) => OPTIMAL_RANGES[stroke].shoulder_abduction,
        getRecommendation: (value, optimal) => {
            if (value < optimal.min) {
                return `Raise your shoulder ${Math.round(optimal.min - value)}° higher for better reach and power`;
            }
            return `Lower your shoulder ${Math.round(value - optimal.max)}° to reduce injury risk`;
        },
    },
    {
        key: 'elbow_flexion',
        label: 'Elbow Angle',
        metricKey: 'right_elbow_flexion',
        performanceImpact: 'medium',
        impactWeight: 1.2,
        maxDeviation: 45,
        getOptimalRange: (stroke) => OPTIMAL_RANGES[stroke].elbow_flexion,
        getRecommendation: (value, optimal) => {
            if (value < optimal.min) {
                return `Extend your elbow ${Math.round(optimal.min - value)}° more for better control`;
            }
            return `Bend your elbow ${Math.round(value - optimal.max)}° more for a compact swing`;
        },
    },
    {
        key: 'knee_flexion',
        label: 'Knee Bend',
        metricKey: 'right_knee_flexion',
        performanceImpact: 'medium',
        impactWeight: 1.1,
        maxDeviation: 40,
        getOptimalRange: (stroke) => OPTIMAL_RANGES[stroke].knee_flexion,
        getRecommendation: (value, optimal) => {
            if (value < optimal.min) {
                return `Your knees are too bent. Straighten ${Math.round(optimal.min - value)}° for better balance`;
            }
            return `Bend your knees ${Math.round(value - optimal.max)}° more for a stable athletic stance`;
        },
    },
];

/**
 * Calculate deviation score for a single value against optimal range
 * Returns 0-100 (100 = within optimal range)
 */
export function calculateDeviationScore(
    value: number | null,
    optimalRange: { min: number; max: number },
    maxDeviation: number
): { score: number; deviation: number; status: 'optimal' | 'warning' | 'critical' } {
    if (value === null) {
        return { score: 0, deviation: 0, status: 'critical' };
    }

    // Check if within optimal range
    if (value >= optimalRange.min && value <= optimalRange.max) {
        return { score: 100, deviation: 0, status: 'optimal' };
    }

    // Calculate distance from optimal range
    const deviation = value < optimalRange.min
        ? optimalRange.min - value
        : value - optimalRange.max;

    // Calculate score (linear decrease from 100 to 0)
    const scoreReduction = (deviation / maxDeviation) * 100;
    const score = Math.max(0, Math.round(100 - scoreReduction));

    // Determine status based on score
    let status: 'optimal' | 'warning' | 'critical';
    if (score >= 80) {
        status = 'optimal';
    } else if (score >= 50) {
        status = 'warning';
    } else {
        status = 'critical';
    }

    return { score, deviation: Math.round(deviation), status };
}

/**
 * Get average value for a metric across all frames
 */
function getAverageMetric(frames: AnalyzedFrame[], metricKey: keyof FrameMetrics): number | null {
    const values = frames
        .map(f => f.metrics[metricKey])
        .filter((v): v is number => typeof v === 'number' && !isNaN(v));

    if (values.length === 0) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Get peak/max value for a metric across all frames
 */
function getMaxMetric(frames: AnalyzedFrame[], metricKey: keyof FrameMetrics): number | null {
    const values = frames
        .map(f => f.metrics[metricKey])
        .filter((v): v is number => typeof v === 'number' && !isNaN(v));

    if (values.length === 0) return null;
    return Math.max(...values);
}

/**
 * Generate full deviation report for a session
 */
export function getDeviationReport(
    frames: AnalyzedFrame[],
    strokeType: StrokeType | string
): DeviationReport {
    const stroke = (strokeType as StrokeType) || 'groundstroke';
    const parameters: DeviationParameter[] = [];

    // Calculate deviation for each parameter
    for (const config of PARAMETER_CONFIGS) {
        const optimalRange = config.getOptimalRange(stroke);

        // Use average value for most metrics, max for shoulder (peak extension)
        const userValue = config.key === 'shoulder_abduction'
            ? getMaxMetric(frames, config.metricKey)
            : getAverageMetric(frames, config.metricKey);

        const { score, deviation, status } = calculateDeviationScore(
            userValue,
            optimalRange,
            config.maxDeviation
        );

        const recommendation = userValue !== null && status !== 'optimal'
            ? config.getRecommendation(userValue, optimalRange)
            : 'Great form! Keep it up.';

        parameters.push({
            key: config.key,
            label: config.label,
            userValue: userValue !== null ? Math.round(userValue) : null,
            optimalRange,
            score,
            deviation,
            status,
            performanceImpact: config.performanceImpact,
            impactWeight: config.impactWeight,
            recommendation,
        });
    }

    // Calculate weighted overall score
    const validParams = parameters.filter(p => p.userValue !== null);
    const totalWeight = validParams.reduce((sum, p) => sum + p.impactWeight, 0);
    const weightedSum = validParams.reduce((sum, p) => sum + (p.score * p.impactWeight), 0);
    const overallScore = totalWeight > 0 ? Math.round(weightedSum / totalWeight) : 0;

    // Get top 3 deviations (prioritized by performance impact and deviation amount)
    const topDeviations = [...parameters]
        .filter(p => p.status !== 'optimal' && p.userValue !== null)
        .sort((a, b) => {
            // Prioritize by impact weight first, then by deviation score (lower = worse)
            const impactDiff = b.impactWeight - a.impactWeight;
            if (Math.abs(impactDiff) > 0.1) return impactDiff;
            return a.score - b.score;
        })
        .slice(0, 3);

    // Generate prioritized improvement recommendations
    const prioritizedImprovements = topDeviations.map(p => p.recommendation);

    // Generate summary
    const summary = generateSummary(overallScore, topDeviations, stroke);

    return {
        strokeType: stroke,
        overallScore,
        parameters,
        topDeviations,
        prioritizedImprovements,
        summary,
    };
}

/**
 * Generate human-readable summary of deviations
 */
function generateSummary(
    overallScore: number,
    topDeviations: DeviationParameter[],
    strokeType: string
): string {
    if (overallScore >= 90) {
        return `Excellent ${strokeType} technique! Your biomechanics are well within optimal ranges.`;
    }

    if (overallScore >= 75) {
        const issues = topDeviations.map(d => d.label.toLowerCase()).join(' and ');
        return `Good ${strokeType} form with minor adjustments needed in ${issues}.`;
    }

    if (overallScore >= 50) {
        const criticalIssues = topDeviations
            .filter(d => d.performanceImpact === 'high')
            .map(d => d.label.toLowerCase());
        if (criticalIssues.length > 0) {
            return `Your ${strokeType} has room for improvement. Focus on ${criticalIssues.join(' and ')} for biggest gains.`;
        }
        return `Moderate ${strokeType} technique. Work on the highlighted areas to improve performance.`;
    }

    return `Your ${strokeType} technique needs significant work. Start with the top recommendations below.`;
}

/**
 * Quick function to get just the top deviations
 */
export function getTopDeviations(
    frames: AnalyzedFrame[],
    strokeType: StrokeType | string,
    limit: number = 3
): DeviationParameter[] {
    const report = getDeviationReport(frames, strokeType);
    return report.topDeviations.slice(0, limit);
}
