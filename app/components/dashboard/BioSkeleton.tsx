"use client";

import React, { useEffect, useState, useRef } from 'react';

interface BioSkeletonProps {
    risks?: {
        shoulder_overuse: number;
        poor_kinetic_chain: number;
        knee_stress: number;
    };
    className?: string; // e.g. "h-40 w-auto"
    mode?: 'analysis' | 'demo';
    drill?: string;
}

// 3D Point definition
type Point3D = { x: number; y: number; z: number };

// Skeleton structure (connections)
const BONES: [string, string][] = [
    // Torso
    ["head", "neck"],
    ["neck", "spine_mid"],
    ["spine_mid", "spine_base"],
    ["neck", "shoulder_center"], // Virtual center for shoulders
    ["shoulder_center", "left_shoulder"],
    ["shoulder_center", "right_shoulder"],
    ["spine_base", "hip_center"], // Virtual center for hips
    ["hip_center", "left_hip"],
    ["hip_center", "right_hip"],

    // Arms
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_hand"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_hand"],

    // Legs
    ["left_hip", "left_knee"],
    ["left_knee", "left_foot"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_foot"]
];

// Base Pose (T-Pose / Ready Pose)
const BASE_JOINTS: Record<string, Point3D> = {
    head: { x: 0, y: -50, z: 0 },
    neck: { x: 0, y: -40, z: 0 },
    spine_mid: { x: 0, y: -20, z: 5 }, // Slight curve
    spine_base: { x: 0, y: 0, z: 0 },

    shoulder_center: { x: 0, y: -38, z: 0 },
    left_shoulder: { x: -15, y: -38, z: 0 },
    right_shoulder: { x: 15, y: -38, z: 0 },

    left_elbow: { x: -25, y: -25, z: 10 }, // Arms forward slightly
    right_elbow: { x: 25, y: -25, z: 10 },
    left_hand: { x: -20, y: -10, z: 20 },
    right_hand: { x: 20, y: -10, z: 20 },

    hip_center: { x: 0, y: 0, z: 0 },
    left_hip: { x: -10, y: 5, z: 0 },
    right_hip: { x: 10, y: 5, z: 0 },

    left_knee: { x: -10, y: 30, z: 5 }, // Knees bent slightly
    right_knee: { x: 10, y: 30, z: 5 },
    left_foot: { x: -12, y: 55, z: 0 },
    right_foot: { x: 12, y: 55, z: 0 }
};


// Athletic Stance Animation Frames (Simple 3-frame sequence: Stand -> Squat -> Stand)
const ATHLETIC_STANCE_FRAMES: Record<string, Point3D>[] = [
    BASE_JOINTS,
    {
        ...BASE_JOINTS,
        spine_mid: { x: 0, y: -10, z: 10 },
        spine_base: { x: 0, y: 10, z: -5 },
        hip_center: { x: 0, y: 10, z: -5 },
        left_hip: { x: -10, y: 15, z: -5 },
        right_hip: { x: 10, y: 15, z: -5 },
        left_knee: { x: -12, y: 35, z: 20 },
        right_knee: { x: 12, y: 35, z: 20 },
        left_foot: { x: -12, y: 55, z: 0 },
        right_foot: { x: 12, y: 55, z: 0 },
        left_elbow: { x: -25, y: -15, z: 20 },
        right_elbow: { x: 25, y: -15, z: 20 },
        left_hand: { x: -20, y: -5, z: 30 },
        right_hand: { x: 20, y: -5, z: 30 },
    },
    BASE_JOINTS
];

// Hip Drive Frames (Rotation focus)
const HIP_DRIVE_FRAMES: Record<string, Point3D>[] = [
    BASE_JOINTS,
    {
        ...BASE_JOINTS,
        // Load phase (twist right)
        spine_mid: { x: 5, y: -20, z: -5 },
        left_shoulder: { x: -10, y: -38, z: 10 },
        right_shoulder: { x: 20, y: -38, z: -10 },
        left_hip: { x: -5, y: 5, z: 10 },
        right_hip: { x: 15, y: 5, z: -5 },
    },
    {
        ...BASE_JOINTS,
        // Drive phase (twist left)
        spine_mid: { x: -5, y: -20, z: 5 },
        left_shoulder: { x: -20, y: -38, z: -10 },
        right_shoulder: { x: 10, y: -38, z: 10 },
        left_hip: { x: -15, y: 5, z: -5 },
        right_hip: { x: 5, y: 5, z: 10 },
    },
    BASE_JOINTS
];

// Low Contact Frames (Stay low, compact)
const LOW_CONTACT_FRAMES: Record<string, Point3D>[] = [
    BASE_JOINTS,
    {
        ...BASE_JOINTS,
        // Low ready position
        spine_mid: { x: 0, y: -10, z: 5 },
        left_hand: { x: -10, y: 10, z: 25 }, // Hands lower
        right_hand: { x: 10, y: 10, z: 25 },
        left_knee: { x: -10, y: 35, z: 5 }, // Knees bent
        right_knee: { x: 10, y: 35, z: 5 },
    },
    {
        ...BASE_JOINTS,
        // Compact low swing
        right_hand: { x: 0, y: 10, z: 35 }, // Push forward low
        right_elbow: { x: 15, y: 0, z: 20 },
    },
    BASE_JOINTS
];

// Arm Extension Frames (Reach out)
const ARM_EXTENSION_FRAMES: Record<string, Point3D>[] = [
    BASE_JOINTS,
    {
        ...BASE_JOINTS,
        // Prepare
        right_hand: { x: 30, y: -20, z: 0 },
        right_elbow: { x: 35, y: -30, z: 5 },
    },
    {
        ...BASE_JOINTS,
        // Full Extension
        right_hand: { x: -10, y: -20, z: 40 }, // Reach far forward
        right_elbow: { x: 5, y: -25, z: 20 }, // Straight-ish arm
        spine_mid: { x: 0, y: -20, z: 10 }, // Leaning into it
    },
    BASE_JOINTS
];

// Linear Interpolation Helper
const lerp = (start: number, end: number, t: number) => start + (end - start) * t;

const lerpPoint = (p1: Point3D, p2: Point3D, t: number): Point3D => ({
    x: lerp(p1.x, p2.x, t),
    y: lerp(p1.y, p2.y, t),
    z: lerp(p1.z, p2.z, t),
});

export default function BioSkeleton({ risks, className = "", mode = "analysis", drill = "athletic_stance" }: BioSkeletonProps) {
    const [angle, setAngle] = useState(0);
    const [animationTime, setAnimationTime] = useState(0);
    const requestRef = useRef<number | null>(null);

    // Animation Loop
    useEffect(() => {
        const animate = (time: number) => {
            if (mode === 'analysis') {
                // Analysis Mode: Continuous slow rotation (Full rotation in ~6 seconds)
                setAngle((time / 6000) * Math.PI * 2);
            } else {
                // Demo Mode: 7s Static, 2s Rotation cycle (Total 9s)
                const CYCLE_DURATION = 9000;
                const STATIC_DURATION = 7000;
                const cycleTime = time % CYCLE_DURATION;

                if (cycleTime < STATIC_DURATION) {
                    setAngle(0); // Face front
                } else {
                    // Rotate 180 degrees (half rotation) during the last 2 seconds
                    const rotationProgress = (cycleTime - STATIC_DURATION) / (CYCLE_DURATION - STATIC_DURATION);
                    setAngle(rotationProgress * Math.PI);
                }

                // Animation Pulse: 0 to 1 every ~2 seconds for pose scrubbing
                // We use this to scrub through frames regardless of rotation
                setAnimationTime(time / 2000);
            }

            requestRef.current = requestAnimationFrame(animate);
        };
        requestRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(requestRef.current!);
    }, [mode]);

    // Calculate Current Pose based on Animation Time
    const getCurrentPose = () => {
        // If analysis mode, just return the base T-pose (will be rotated by project function)
        if (mode === 'analysis') {
            return BASE_JOINTS;
        }

        let frames = ATHLETIC_STANCE_FRAMES;
        if (drill === 'hip_drive') frames = HIP_DRIVE_FRAMES;
        if (drill === 'low_contact') frames = LOW_CONTACT_FRAMES;
        if (drill === 'arm_extension') frames = ARM_EXTENSION_FRAMES;

        const totalFrames = frames.length;
        const durationPerCycle = totalFrames - 1;

        // Map time to a position in the frame sequence (e.g. 0.5, 1.2, 2.0)
        // modulo durationPerCycle makes it loop
        const progress = animationTime % durationPerCycle;

        const currentFrameIndex = Math.floor(progress);
        const nextFrameIndex = (currentFrameIndex + 1) % totalFrames;
        const frameT = progress - currentFrameIndex; // 0.0 to 1.0 within current frame interval

        const frameA = frames[currentFrameIndex];
        const frameB = frames[nextFrameIndex];

        // Intepolate all joints
        const currentJoints: Record<string, Point3D> = {};
        for (const key in frameA) {
            // Use frameB[key] if available, otherwise fallback to frameA (or BASE) to prevent crash
            const targetPoint = frameB[key] || frameA[key];
            currentJoints[key] = lerpPoint(frameA[key], targetPoint, frameT);
        }
        return currentJoints;
    };

    const currentJoints = getCurrentPose();

    // 3D Projection Helper
    const project = (point: Point3D): { x: number; y: number; scale: number; opacity: number } => {
        // Rotate around Y axis
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        const x = point.x * cos - point.z * sin;
        const z = point.x * sin + point.z * cos;
        const y = point.y;

        // Perspective Project
        const FL = 300; // Focal length
        const scale = FL / (FL + z);

        return {
            x: x * scale + 50, // Center in 100x120 SVG
            y: y * scale + 60,
            scale,
            opacity: Math.max(0.3, (z + 50) / 100) // Simple depth cue
        };
    };

    // Color Logic
    const getColor = (risk: number) => {
        if (risk > 66) return "#ef4444"; // red
        if (risk > 33) return "#f59e0b"; // amber
        return "#10b981"; // emerald
    };

    const shoulderColor = getColor(risks?.shoulder_overuse || 0);
    const spineColor = getColor(risks?.poor_kinetic_chain || 0);
    const kneeColor = getColor(risks?.knee_stress || 0);
    const boneColor = "#d4d4d4";

    // Map joints to colors
    const getJointColor = (name: string) => {
        if (mode === 'demo') return "#10b981"; // emerald for perfect form

        if (name.includes("shoulder")) return shoulderColor;
        if (name.includes("spine") || name.includes("hip")) return spineColor;
        if (name.includes("knee")) return kneeColor;
        return boneColor;
    };

    const projectedJoints = Object.fromEntries(
        Object.entries(currentJoints).map(([name, point]) => [name, project(point)])
    );

    return (
        <svg viewBox="0 0 100 120" className={className} style={{ overflow: 'visible' }}>
            <defs>
                <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                    <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>

            {/* Bones */}
            {BONES.map(([startName, endName], i) => {
                const start = projectedJoints[startName];
                const end = projectedJoints[endName];

                // Safety check if joint is missing
                if (!start || !end) return null;

                // Color gradient for bone? Just average for now or transparent
                const isRiskArea = getJointColor(startName) !== boneColor || getJointColor(endName) !== boneColor;
                const strokeColor = isRiskArea ? (getJointColor(startName)) : boneColor;

                return (
                    <line
                        key={`bone-${i}`}
                        x1={start.x} y1={start.y}
                        x2={end.x} y2={end.y}
                        stroke={strokeColor}
                        strokeWidth={3 * ((start.scale + end.scale) / 2)}
                        strokeOpacity={Math.min(start.opacity, end.opacity)}
                        strokeLinecap="round"
                    />
                );
            })}

            {/* Joints */}
            {Object.entries(projectedJoints).map(([name, p]) => (
                <circle
                    key={name}
                    cx={p.x}
                    cy={p.y}
                    r={3 * p.scale}
                    fill={getJointColor(name)}
                    fillOpacity={p.opacity}
                    filter={getJointColor(name) !== boneColor ? "url(#glow)" : undefined}
                />
            ))}
        </svg>
    );
}
