/**
 * Analytics Summary API
 * 
 * Returns comprehensive analytics data including:
 * - Aggregate statistics
 * - Trend data over time
 * - Stroke-specific breakdown
 * - Metrics with optimal ranges
 * - Personalized drill recommendations
 */

import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

// Optimal ranges for metrics
const OPTIMAL_RANGES = {
    hip_rotation: { min: 30, max: 90, unit: "°", label: "Hip Rotation" },
    shoulder_abduction: { min: 0, max: 140, unit: "°", label: "Shoulder Abduction" },
    knee_flexion: { min: 120, max: 160, unit: "°", label: "Knee Flexion" },
    elbow_flexion: { min: 90, max: 150, unit: "°", label: "Elbow Flexion" },
};

// Drill recommendations based on issues
const DRILL_RECOMMENDATIONS: Record<string, { name: string; description: string; duration: string }> = {
    poor_kinetic_chain: {
        name: "Hip Drive Drill",
        description: "Practice rotating hips before arm swing. Stand sideways, load on back foot, rotate hips toward target.",
        duration: "5 mins",
    },
    shoulder_overuse: {
        name: "Low Contact Drill",
        description: "Keep paddle below shoulder height. Practice compact swings with emphasis on wrist snap.",
        duration: "5 mins",
    },
    knee_stress: {
        name: "Athletic Stance Practice",
        description: "Maintain soft knees at 130-150° angle. Practice ready position without over-bending.",
        duration: "3 mins",
    },
    elbow_strain: {
        name: "Arm Extension Drill",
        description: "Focus on full extension at contact. Practice shadow swings with proper arm angle.",
        duration: "4 mins",
    },
};

interface TrendPoint {
    date: string;
    score: number;
    sessions: number;
}

interface StrokeBreakdown {
    stroke_type: string;
    sessions: number;
    avg_score: number;
    avg_risk: string;
    metrics: {
        hip_rotation: number | null;
        shoulder_abduction: number | null;
        knee_flexion: number | null;
    };
}

interface MetricCard {
    key: string;
    label: string;
    current: number | null;
    optimal_min: number;
    optimal_max: number;
    unit: string;
    status: "good" | "warning" | "critical";
}

interface DrillRecommendation {
    name: string;
    description: string;
    duration: string;
    priority: number;
    issue: string;
}

interface AnalyticsSummary {
    totalSessions: number;
    completedSessions: number;
    skillScore: number;
    averageRisk: string;
    favoriteStroke: string | null;
    riskBreakdown: {
        shoulder_overuse: number;
        poor_kinetic_chain: number;
        knee_stress: number;
    };
    recentSessions: Array<{
        id: string;
        stroke_type: string;
        created_at: string;
        overall_risk: string;
        score: number;
    }>;
    topIssues: Array<{
        type: string;
        count: number;
        percentage: number;
    }>;
    // New enhanced data
    trends: TrendPoint[];
    strokeBreakdown: StrokeBreakdown[];
    metricCards: MetricCard[];
    drills: DrillRecommendation[];
}

export async function GET(): Promise<NextResponse> {
    try {
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: "Authentication required" },
                { status: 401 }
            );
        }

        // Fetch all completed analyses for this user
        const { data: analyses, error } = await supabase
            .from("analysis_jobs")
            .select("*")
            .eq("user_id", userId)
            .eq("status", "completed")
            .order("created_at", { ascending: false });

        if (error) {
            console.error("Error fetching analyses:", error);
            return NextResponse.json(
                { error: "Failed to fetch analytics" },
                { status: 500 }
            );
        }

        const totalSessions = analyses?.length || 0;

        // Count stroke types and aggregate metrics
        const strokeCounts: Record<string, number> = {};
        const strokeMetrics: Record<string, { hip: number[]; shoulder: number[]; knee: number[]; scores: number[] }> = {};
        const riskCounts: Record<string, number> = { low: 0, medium: 0, high: 0 };
        const aggregatedRisks = { shoulder_overuse: 0, poor_kinetic_chain: 0, knee_stress: 0 };
        const allMetrics = { hip_rotation: [] as number[], shoulder_abduction: [] as number[], knee_flexion: [] as number[] };

        // For trends - group by date
        const dateScores: Record<string, { scores: number[]; count: number }> = {};

        analyses?.forEach((job) => {
            const stroke = job.stroke_type || "unknown";
            strokeCounts[stroke] = (strokeCounts[stroke] || 0) + 1;

            if (!strokeMetrics[stroke]) {
                strokeMetrics[stroke] = { hip: [], shoulder: [], knee: [], scores: [] };
            }

            if (job.result_json?.summary) {
                const summary = job.result_json.summary;
                const risk = summary.overall_risk || "low";
                riskCounts[risk] = (riskCounts[risk] || 0) + 1;

                if (summary.risk_percentages) {
                    aggregatedRisks.shoulder_overuse += summary.risk_percentages.shoulder_overuse || 0;
                    aggregatedRisks.poor_kinetic_chain += summary.risk_percentages.poor_kinetic_chain || 0;
                    aggregatedRisks.knee_stress += summary.risk_percentages.knee_stress || 0;
                }

                // Calculate score for this session
                const avgRiskPct = summary.risk_percentages
                    ? (summary.risk_percentages.shoulder_overuse + summary.risk_percentages.poor_kinetic_chain + summary.risk_percentages.knee_stress) / 3
                    : 0;
                const score = Math.round(Math.max(0, 100 - avgRiskPct * 2));
                strokeMetrics[stroke].scores.push(score);

                // Trend data - group by date
                const dateKey = new Date(job.created_at).toISOString().split("T")[0];
                if (!dateScores[dateKey]) {
                    dateScores[dateKey] = { scores: [], count: 0 };
                }
                dateScores[dateKey].scores.push(score);
                dateScores[dateKey].count++;
            }

            // Extract frame metrics
            if (job.result_json?.frames && Array.isArray(job.result_json.frames)) {
                job.result_json.frames.forEach((frame: any) => {
                    if (frame.metrics) {
                        if (frame.metrics.hip_rotation_deg != null) {
                            allMetrics.hip_rotation.push(frame.metrics.hip_rotation_deg);
                            strokeMetrics[stroke].hip.push(frame.metrics.hip_rotation_deg);
                        }
                        if (frame.metrics.right_shoulder_abduction != null) {
                            allMetrics.shoulder_abduction.push(frame.metrics.right_shoulder_abduction);
                            strokeMetrics[stroke].shoulder.push(frame.metrics.right_shoulder_abduction);
                        }
                        if (frame.metrics.right_knee_flexion != null) {
                            allMetrics.knee_flexion.push(frame.metrics.right_knee_flexion);
                            strokeMetrics[stroke].knee.push(frame.metrics.right_knee_flexion);
                        }
                    }
                });
            }
        });

        // Calculate averages
        if (totalSessions > 0) {
            aggregatedRisks.shoulder_overuse /= totalSessions;
            aggregatedRisks.poor_kinetic_chain /= totalSessions;
            aggregatedRisks.knee_stress /= totalSessions;
        }

        const favoriteStroke = Object.entries(strokeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

        let averageRisk = "low";
        if (riskCounts.high > totalSessions * 0.3) {
            averageRisk = "high";
        } else if (riskCounts.medium > totalSessions * 0.3 || riskCounts.high > 0) {
            averageRisk = "medium";
        }

        // Calculate overall skill score
        const avgRiskPct = (aggregatedRisks.shoulder_overuse + aggregatedRisks.poor_kinetic_chain + aggregatedRisks.knee_stress) / 3;
        const skillScore = Math.round(Math.max(0, 100 - avgRiskPct * 2));

        // Build trend data (last 7 days)
        const trends: TrendPoint[] = [];
        const today = new Date();
        for (let i = 6; i >= 0; i--) {
            const date = new Date(today);
            date.setDate(date.getDate() - i);
            const dateKey = date.toISOString().split("T")[0];
            const dayData = dateScores[dateKey];
            trends.push({
                date: date.toLocaleDateString("en-US", { weekday: "short" }),
                score: dayData ? Math.round(dayData.scores.reduce((a, b) => a + b, 0) / dayData.scores.length) : 0,
                sessions: dayData?.count || 0,
            });
        }

        // Build stroke breakdown
        const strokeBreakdown: StrokeBreakdown[] = Object.entries(strokeCounts).map(([stroke, count]) => {
            const metrics = strokeMetrics[stroke];
            const avgScore = metrics.scores.length > 0 ? Math.round(metrics.scores.reduce((a, b) => a + b, 0) / metrics.scores.length) : 0;
            return {
                stroke_type: stroke,
                sessions: count,
                avg_score: avgScore,
                avg_risk: avgScore >= 80 ? "low" : avgScore >= 60 ? "medium" : "high",
                metrics: {
                    hip_rotation: metrics.hip.length > 0 ? Math.round(metrics.hip.reduce((a, b) => a + b, 0) / metrics.hip.length) : null,
                    shoulder_abduction: metrics.shoulder.length > 0 ? Math.round(metrics.shoulder.reduce((a, b) => a + b, 0) / metrics.shoulder.length) : null,
                    knee_flexion: metrics.knee.length > 0 ? Math.round(metrics.knee.reduce((a, b) => a + b, 0) / metrics.knee.length) : null,
                },
            };
        });

        // Build metric cards
        const avgHip = allMetrics.hip_rotation.length > 0 ? Math.round(allMetrics.hip_rotation.reduce((a, b) => a + b, 0) / allMetrics.hip_rotation.length) : null;
        const avgShoulder = allMetrics.shoulder_abduction.length > 0 ? Math.round(allMetrics.shoulder_abduction.reduce((a, b) => a + b, 0) / allMetrics.shoulder_abduction.length) : null;
        const avgKnee = allMetrics.knee_flexion.length > 0 ? Math.round(allMetrics.knee_flexion.reduce((a, b) => a + b, 0) / allMetrics.knee_flexion.length) : null;

        const getStatus = (value: number | null, min: number, max: number): "good" | "warning" | "critical" => {
            if (value === null) return "good";
            if (value >= min && value <= max) return "good";
            const distance = value < min ? min - value : value - max;
            return distance > 20 ? "critical" : "warning";
        };

        const metricCards: MetricCard[] = [
            {
                key: "hip_rotation",
                label: "Hip Rotation",
                current: avgHip,
                optimal_min: 30,
                optimal_max: 90,
                unit: "°",
                status: getStatus(avgHip, 30, 90),
            },
            {
                key: "shoulder_abduction",
                label: "Shoulder Abduction",
                current: avgShoulder,
                optimal_min: 0,
                optimal_max: 140,
                unit: "°",
                status: getStatus(avgShoulder, 0, 140),
            },
            {
                key: "knee_flexion",
                label: "Knee Flexion",
                current: avgKnee,
                optimal_min: 120,
                optimal_max: 160,
                unit: "°",
                status: getStatus(avgKnee, 120, 160),
            },
        ];

        // Build drill recommendations based on top issues
        const drills: DrillRecommendation[] = [];
        const issueOrder = [
            { key: "poor_kinetic_chain", value: aggregatedRisks.poor_kinetic_chain },
            { key: "shoulder_overuse", value: aggregatedRisks.shoulder_overuse },
            { key: "knee_stress", value: aggregatedRisks.knee_stress },
        ].sort((a, b) => b.value - a.value);

        issueOrder.forEach((issue, idx) => {
            if (issue.value > 2 && DRILL_RECOMMENDATIONS[issue.key]) {
                drills.push({
                    ...DRILL_RECOMMENDATIONS[issue.key],
                    priority: idx + 1,
                    issue: issue.key,
                });
            }
        });

        // Recent sessions
        const recentSessions = (analyses || []).slice(0, 5).map((job) => {
            const summary = job.result_json?.summary;
            const avgRiskPct = summary?.risk_percentages
                ? (summary.risk_percentages.shoulder_overuse + summary.risk_percentages.poor_kinetic_chain + summary.risk_percentages.knee_stress) / 3
                : 0;
            const score = Math.round(Math.max(0, 100 - avgRiskPct * 2));
            return {
                id: job.id,
                stroke_type: job.stroke_type,
                created_at: job.created_at,
                overall_risk: summary?.overall_risk || "low",
                score,
            };
        });

        // Top issues
        const topIssues = [
            { type: "Shoulder Overuse", count: 0, percentage: aggregatedRisks.shoulder_overuse },
            { type: "Poor Kinetic Chain", count: 0, percentage: aggregatedRisks.poor_kinetic_chain },
            { type: "Knee Stress", count: 0, percentage: aggregatedRisks.knee_stress },
        ]
            .filter((issue) => issue.percentage > 0)
            .sort((a, b) => b.percentage - a.percentage);

        const summary: AnalyticsSummary = {
            totalSessions,
            completedSessions: totalSessions,
            skillScore,
            averageRisk,
            favoriteStroke,
            riskBreakdown: {
                shoulder_overuse: Math.round(aggregatedRisks.shoulder_overuse * 10) / 10,
                poor_kinetic_chain: Math.round(aggregatedRisks.poor_kinetic_chain * 10) / 10,
                knee_stress: Math.round(aggregatedRisks.knee_stress * 10) / 10,
            },
            recentSessions,
            topIssues,
            trends,
            strokeBreakdown,
            metricCards,
            drills,
        };

        return NextResponse.json(summary);

    } catch (error: any) {
        console.error("Analytics error:", error);
        return NextResponse.json(
            { error: "Failed to fetch analytics", details: error.message },
            { status: 500 }
        );
    }
}
