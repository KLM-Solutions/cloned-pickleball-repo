"use client";

import { useMemo } from "react";
import { Gauge } from "./Gauge";

interface BiomechanicalMetricsProps {
    currentFrameMetrics?: {
        hip_rotation_deg?: number;
        right_shoulder_abduction?: number;
        right_knee_flexion?: number;
    };
    aggregates: {
        hip: number;
        shoulder: number;
        knee: number;
    };
}

export function BiomechanicalMetrics({ currentFrameMetrics, aggregates }: BiomechanicalMetricsProps) {

    // FIXED: Fallback to aggregates with better validation and debugging
    const hipValue = currentFrameMetrics?.hip_rotation_deg ?? aggregates.hip ?? 0;
    const shoulderValue = currentFrameMetrics?.right_shoulder_abduction ?? aggregates.shoulder ?? 0;
    const kneeValue = currentFrameMetrics?.right_knee_flexion ?? aggregates.knee ?? 90;

    // Debug logging to ensure data is flowing
    console.log("BiomechanicalMetrics Update:", {
        currentFrame: currentFrameMetrics,
        aggregates,
        computed: { hipValue, shoulderValue, kneeValue }
    });

    return (
        <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-200">
            <h3 className="text-sm font-bold text-black mb-4 flex items-center gap-2">
                📐 Biomechanical Analysis
            </h3>

            <div className="grid grid-cols-3 gap-2">
                {/* Hip Rotation: 30-70° is optimal for power generation */}
                <Gauge
                    label="Hip Rotation"
                    value={hipValue}
                    max={120}
                    ranges={[
                        { min: 0, max: 20, color: '#ef4444' },    // Red - Too low (poor power)
                        { min: 21, max: 30, color: '#f59e0b' },   // Amber - Below optimal
                        { min: 31, max: 70, color: '#22c55e' },   // Green - Optimal range
                        { min: 71, max: 90, color: '#f59e0b' },   // Amber - Above optimal
                        { min: 91, max: 120, color: '#ef4444' },  // Red - Too high (strain risk)
                    ]}
                />

                {/* Shoulder Abduction: 45-110° is optimal for strokes */}
                <Gauge
                    label="Shldr Abd."
                    value={shoulderValue}
                    max={180}
                    ranges={[
                        { min: 0, max: 30, color: '#ef4444' },    // Red - Too low
                        { min: 31, max: 45, color: '#f59e0b' },   // Amber - Below optimal
                        { min: 46, max: 110, color: '#22c55e' },  // Green - Optimal range
                        { min: 111, max: 140, color: '#f59e0b' }, // Amber - Above optimal
                        { min: 141, max: 180, color: '#ef4444' }, // Red - Impingement risk
                    ]}
                />

                {/* Knee Flexion: 90-140° is optimal for stability */}
                <Gauge
                    label="Knee Flexion"
                    value={kneeValue}
                    max={180}
                    ranges={[
                        { min: 0, max: 70, color: '#ef4444' },    // Red - Too bent (strain)
                        { min: 71, max: 89, color: '#f59e0b' },   // Amber - Deep flexion
                        { min: 90, max: 140, color: '#22c55e' },  // Green - Optimal range
                        { min: 141, max: 160, color: '#f59e0b' }, // Amber - Too straight
                        { min: 161, max: 180, color: '#ef4444' }, // Red - Locked out (injury risk)
                    ]}
                />
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center justify-center gap-4 text-[10px] text-neutral-500">
                <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Optimal</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-amber-500"></div>
                    <span>Caution</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-red-500"></div>
                    <span>Risk</span>
                </div>
            </div>
        </div>
    );
}
