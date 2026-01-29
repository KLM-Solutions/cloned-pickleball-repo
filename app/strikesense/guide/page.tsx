"use client";

import React, { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Camera, Ruler, MoveVertical, CheckCircle2, Smartphone, Sun, Clock, ChevronRight } from "lucide-react";

export const dynamic = 'force-dynamic';

const cameraGuides: Record<string, any> = {
    serve: {
        angle: '90° Side View',
        distance: '8-12 feet',
        height: 'Chest Level',
        icon: '🎾',
        title: 'Serve Analysis',
        subtitle: 'Capture your full service motion',
        image: '/images/serve-camera-guide.png',
        tips: [
            { icon: Smartphone, text: 'Place phone on tripod for stability' },
            { icon: Camera, text: 'Record from your non-dominant side' },
            { icon: Sun, text: 'Face the light source for clarity' },
            { icon: Clock, text: 'Record 3-5 serves for best results' },
        ]
    },
    groundstroke: {
        angle: '45° Rear View',
        distance: '10-15 feet',
        height: 'Waist Level',
        icon: '💪',
        title: 'Drive Analysis',
        subtitle: 'Analyze your forehand & backhand',
        image: '/images/groundstroke-camera-guide.png',
        tips: [
            { icon: Smartphone, text: 'Position behind and to the side' },
            { icon: Camera, text: 'Capture full body and racket swing' },
            { icon: Sun, text: 'Ensure good lighting conditions' },
            { icon: Clock, text: 'Record multiple rallies' },
        ]
    },
    dink: {
        angle: 'Front or Side',
        distance: '8-10 feet',
        height: 'Net Height',
        icon: '🤏',
        title: 'Dink Analysis',
        subtitle: 'Perfect your soft game technique',
        image: '/images/dink-camera-guide.png',
        tips: [
            { icon: Smartphone, text: 'Film at kitchen line level' },
            { icon: Camera, text: 'Focus on paddle and wrist action' },
            { icon: Sun, text: 'Avoid shadows on the court' },
            { icon: Clock, text: 'Record several dink exchanges' },
        ]
    },
    overhead: {
        angle: '90° Side View',
        distance: '8-12 feet',
        height: 'Chest Level',
        icon: '⚡',
        title: 'Overhead Analysis',
        subtitle: 'Optimize your smash technique',
        image: '/images/overhead-camera-guide.png',
        tips: [
            { icon: Smartphone, text: 'Capture full vertical extension' },
            { icon: Camera, text: 'Film from 90° to your hitting arm' },
            { icon: Sun, text: 'Bright lighting for fast motion' },
            { icon: Clock, text: 'Record 5-10 overhead shots' },
        ]
    },
    footwork: {
        angle: 'Rear View',
        distance: '12-15 feet',
        height: 'Waist Level',
        icon: '👟',
        title: 'Footwork Analysis',
        subtitle: 'Analyze movement and positioning',
        image: '/images/footwork-camera-guide.png',
        tips: [
            { icon: Smartphone, text: 'Position behind the baseline' },
            { icon: Camera, text: 'Capture full court movement' },
            { icon: Sun, text: 'Good lighting for quick movements' },
            { icon: Clock, text: 'Record several rally exchanges' },
        ]
    },
    overall: {
        angle: 'Rear or Side',
        distance: '10-15 feet',
        height: 'Chest Level',
        icon: '🎯',
        title: 'Overall Analysis',
        subtitle: 'Full game assessment',
        image: '/images/overall-camera-guide.png',
        tips: [
            { icon: Smartphone, text: 'Capture variety of shots' },
            { icon: Camera, text: 'Film from consistent angle' },
            { icon: Sun, text: 'Ensure uniform lighting' },
            { icon: Clock, text: 'Record 2-3 minutes of play' },
        ]
    }
};

function CameraGuideContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const strokeType = searchParams.get('stroke') || 'serve';
    const guide = cameraGuides[strokeType] || cameraGuides.serve;

    return (
        <div className="min-h-screen bg-white text-neutral-900">
            {/* Header */}
            <header className="relative z-10 px-4 py-3 md:py-4">
                <div className="max-w-lg mx-auto flex items-center justify-between">
                    <button
                        onClick={() => router.push('/')}
                        className="flex items-center gap-1.5 md:gap-2 text-neutral-500 hover:text-black transition p-1"
                    >
                        <ArrowLeft className="w-4 h-4 md:w-5 md:h-5" />
                        <span className="text-xs md:text-sm font-medium">Home</span>
                    </button>
                    <div className="text-[10px] md:text-xs font-bold tracking-widest uppercase text-neutral-400">
                        Step 1 of 3
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 px-4 pb-6 md:pb-8">
                <div className="max-w-lg mx-auto">

                    {/* Hero */}
                    <div className="text-center pt-2 md:pt-4 pb-6 md:pb-8">
                        <div className="inline-flex w-20 h-20 md:w-24 md:h-24 rounded-2xl md:rounded-3xl bg-black items-center justify-center text-4xl md:text-5xl mb-4 md:mb-6">
                            {guide.icon}
                        </div>
                        <h1 className="text-2xl md:text-3xl font-bold mb-1.5 md:mb-2">{guide.title}</h1>
                        <p className="text-neutral-500 text-sm md:text-lg">{guide.subtitle}</p>
                    </div>

                    {/* Visual Camera Guide Image */}
                    {guide.image && (
                        <div className="mb-4 md:mb-6">
                            <div className="relative rounded-2xl md:rounded-3xl overflow-hidden border-2 border-neutral-200 shadow-lg">
                                <img
                                    src={guide.image}
                                    alt={`Camera position guide for ${guide.title}`}
                                    className="w-full h-auto"
                                />
                                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                                    <div className="flex items-center gap-2 text-white">
                                        <Camera className="w-4 h-4" />
                                        <span className="text-sm font-semibold">Camera Setup Guide</span>
                                    </div>
                                </div>
                            </div>
                            <p className="text-center text-xs text-neutral-400 mt-2">
                                Position your camera as shown above for best results
                            </p>
                        </div>
                    )}

                    {/* Camera Position Card */}
                    <div className="relative bg-black rounded-2xl md:rounded-3xl p-5 md:p-6 mb-4 md:mb-6 overflow-hidden">
                        <div className="absolute top-0 right-0 w-24 md:w-32 h-24 md:h-32 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                        <div className="absolute bottom-0 left-0 w-20 md:w-24 h-20 md:h-24 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />

                        <div className="relative z-10">
                            <h2 className="text-sm md:text-lg font-bold mb-3 md:mb-4 flex items-center gap-2 text-white">
                                <Camera className="w-4 h-4 md:w-5 md:h-5" />
                                Camera Position
                            </h2>

                            <div className="grid grid-cols-3 gap-2 md:gap-3">
                                <div className="bg-white/10 rounded-xl md:rounded-2xl p-3 md:p-4 text-center">
                                    <Camera className="w-5 h-5 md:w-6 md:h-6 mx-auto mb-1.5 md:mb-2 text-white" />
                                    <div className="text-[10px] md:text-xs text-white/60 mb-0.5 md:mb-1">Angle</div>
                                    <div className="text-xs md:text-sm font-bold text-white">{guide.angle}</div>
                                </div>
                                <div className="bg-white/10 rounded-xl md:rounded-2xl p-3 md:p-4 text-center">
                                    <Ruler className="w-5 h-5 md:w-6 md:h-6 mx-auto mb-1.5 md:mb-2 text-white" />
                                    <div className="text-[10px] md:text-xs text-white/60 mb-0.5 md:mb-1">Distance</div>
                                    <div className="text-xs md:text-sm font-bold text-white">{guide.distance}</div>
                                </div>
                                <div className="bg-white/10 rounded-xl md:rounded-2xl p-3 md:p-4 text-center">
                                    <MoveVertical className="w-5 h-5 md:w-6 md:h-6 mx-auto mb-1.5 md:mb-2 text-white" />
                                    <div className="text-[10px] md:text-xs text-white/60 mb-0.5 md:mb-1">Height</div>
                                    <div className="text-xs md:text-sm font-bold text-white">{guide.height}</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Tips */}
                    <div className="bg-neutral-50 border border-neutral-200 rounded-xl md:rounded-2xl p-4 md:p-5 mb-6 md:mb-8">
                        <h3 className="text-[10px] md:text-sm font-bold uppercase tracking-wider text-neutral-400 mb-3 md:mb-4">
                            Recording Tips
                        </h3>
                        <div className="space-y-3 md:space-y-4">
                            {guide.tips.map((tip: any, i: number) => (
                                <div key={i} className="flex items-start gap-3 md:gap-4">
                                    <div className="w-9 h-9 md:w-10 md:h-10 rounded-lg md:rounded-xl bg-black flex items-center justify-center flex-shrink-0">
                                        <tip.icon className="w-4 h-4 md:w-5 md:h-5 text-white" />
                                    </div>
                                    <div className="flex-1 pt-2 md:pt-2">
                                        <span className="text-neutral-600 text-xs md:text-sm">{tip.text}</span>
                                    </div>
                                    <CheckCircle2 className="w-4 h-4 md:w-5 md:h-5 text-black mt-2 md:mt-2.5 flex-shrink-0" />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* CTA */}
                    <button
                        onClick={() => router.push(`/strikesense/upload?stroke=${strokeType}`)}
                        className="w-full bg-black hover:bg-neutral-800 text-white py-4 md:py-5 rounded-xl md:rounded-2xl font-bold text-sm md:text-base transition-all active:scale-[0.98] flex items-center justify-center gap-2 md:gap-3"
                    >
                        <span>I'm Ready to Record</span>
                        <ChevronRight className="w-4 h-4 md:w-5 md:h-5" />
                    </button>

                    <p className="text-center text-neutral-400 text-xs md:text-sm mt-3 md:mt-4">
                        Already have a video? Tap to continue
                    </p>
                </div>
            </main>
        </div>
    );
}

export default function CameraGuidePage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="text-black font-bold animate-pulse">Loading...</div>
            </div>
        }>
            <CameraGuideContent />
        </Suspense>
    );
}
