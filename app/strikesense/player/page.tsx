"use client";

import { useState, useMemo, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Award, ChevronRight, BarChart3, AlertTriangle, TrendingUp, Play, Lightbulb, Sparkles } from "lucide-react";
import VideoPanel from "../../components/VideoPanel";
import { BiomechanicalMetrics } from "../../components/dashboard/BiomechanicalMetrics";
import { filterFramesForIssues, getTopIssues, getFilterSummary, getFullRecommendation, FilteredFrame, FilterSummary, CoachingRecommendation } from "@/lib/analysis";

// Demo job UUID for fetching pre-analyzed session from database
const DEMO_JOB_ID = "762644e0-0192-49d9-bf85-a127c3b56449";

export const dynamic = 'force-dynamic';

function PlayerContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const strokeType = searchParams.get('stroke') || 'serve';
    const isDemo = searchParams.get('demo') === 'true';

    const [currentTime, setCurrentTime] = useState(0);
    const [showSidebar, setShowSidebar] = useState(false);
    const [analysisData, setAnalysisData] = useState<any>(null);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [showMetrics, setShowMetrics] = useState(true);
    const [activeTab, setActiveTab] = useState<'metrics' | 'issues'>('metrics');
    const [selectedIssue, setSelectedIssue] = useState<string | null>(null);
    const [demoLoading, setDemoLoading] = useState(false);

    useEffect(() => {
        if (typeof window !== 'undefined' && window.innerWidth >= 1024) {
            setShowSidebar(true);
        }
    }, []);

    useEffect(() => {
        // If demo mode, fetch data from database using the demo job UUID
        if (isDemo) {
            const fetchDemoData = async () => {
                setDemoLoading(true);
                try {
                    const response = await fetch(`/api/analyze/status?job_id=${DEMO_JOB_ID}`);
                    if (!response.ok) throw new Error("Failed to fetch demo data");

                    const jobData = await response.json();

                    if (jobData.status === "completed" && jobData.result) {
                        const result = typeof jobData.result === 'string'
                            ? JSON.parse(jobData.result)
                            : jobData.result;

                        const analysisResult = {
                            ...result,
                            videoUrl: jobData.videoUrl || result.videoUrl,
                            stroke_type: jobData.stroke_type || "serve",
                            llm_response: jobData.llm_response,
                            isDemo: true,
                        };

                        setAnalysisData(analysisResult);
                        // Also store in sessionStorage so analysis page can access it
                        sessionStorage.setItem('analysisResult', JSON.stringify(analysisResult));
                    }
                } catch (error) {
                    console.error("Failed to fetch demo data:", error);
                } finally {
                    setDemoLoading(false);
                }
            };

            fetchDemoData();
            return;
        }

        const storedResult = sessionStorage.getItem('analysisResult');
        if (storedResult) {
            const result = JSON.parse(storedResult);
            setAnalysisData(result);
        }
    }, [isDemo]);


    const currentFrame = useMemo(() => {
        if (!analysisData?.frames || analysisData.frames.length === 0) return null;
        let bestIdx = 0;
        let bestDiff = Infinity;
        analysisData.frames.forEach((frame: any, idx: number) => {
            const timestamp = frame.timestampSec ?? frame.timestamp_sec ?? 0;
            const diff = Math.abs(timestamp - currentTime);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = idx;
            }
        });
        const frame = analysisData.frames[bestIdx];
        return {
            ...frame,
            _calculatedIdx: bestIdx,
            _calculatedTime: frame.timestampSec ?? frame.timestamp_sec ?? 0
        };
    }, [currentTime, analysisData]);

    const aggregates = useMemo(() => {
        const frames = analysisData?.frames || [];
        if (frames.length === 0) return { hip: 0, shoulder: 0, knee: 0 };

        const validHip = frames.filter((f: any) => f.metrics?.hip_rotation_deg != null);
        const validShoulder = frames.filter((f: any) => f.metrics?.right_shoulder_abduction != null);
        const validKnee = frames.filter((f: any) => f.metrics?.right_knee_flexion != null);

        return {
            hip: validHip.length > 0 ? Math.round(validHip.reduce((acc: number, f: any) => acc + f.metrics.hip_rotation_deg, 0) / validHip.length) : 0,
            shoulder: validShoulder.length > 0 ? Math.round(Math.max(...validShoulder.map((f: any) => f.metrics.right_shoulder_abduction))) : 0,
            knee: validKnee.length > 0 ? Math.round(Math.min(...validKnee.map((f: any) => f.metrics.right_knee_flexion))) : 0,
        };
    }, [analysisData]);

    // Filter frames for issues based on stroke type
    const { filteredFrames, filterSummary, topIssues } = useMemo(() => {
        if (!analysisData?.frames || analysisData.frames.length === 0) {
            return { filteredFrames: [], filterSummary: null, topIssues: [] };
        }

        const filtered = filterFramesForIssues(analysisData.frames, strokeType);
        const summary = getFilterSummary(analysisData.frames, filtered);
        const top = getTopIssues(filtered, 5);

        return { filteredFrames: filtered, filterSummary: summary, topIssues: top };
    }, [analysisData, strokeType]);

    // Construct effect to set default selected issue
    useEffect(() => {
        if (topIssues.length > 0 && !selectedIssue) {
            setSelectedIssue(topIssues[0].issue);
        }
    }, [topIssues, selectedIssue]);

    // Jump to frame
    const jumpToFrame = (frame: any) => {
        const time = frame.timestampSec ?? frame.timestamp_sec ?? 0;
        setCurrentTime(time);
    };

    // Get issue label
    const getIssueLabel = (type: string) => {
        const labels: Record<string, string> = {
            shoulder_overuse: 'Shoulder Overuse',
            shoulder_over_rotation: 'Over-Rotation',
            shoulder_under_rotation: 'Under-Rotation',
            poor_kinetic_chain: 'Poor Kinetic Chain',
            insufficient_hip_rotation: 'Low Hip Power',
            knee_stress: 'Knee Stress',
            excessive_knee_bend: 'Deep Knee Bend',
            insufficient_knee_bend: 'Straight Knees',
            elbow_form: 'Elbow Position',
            elbow_strain: 'Elbow Strain',
        };
        return labels[type] || type.replace(/_/g, ' ');
    };

    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="relative z-20 h-14 bg-white border-b border-neutral-200 px-4 flex items-center justify-between sticky top-0">
                <div className="max-w-7xl mx-auto w-full flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => router.push('/')}
                            className="flex items-center gap-2 text-neutral-500 hover:text-black transition"
                        >
                            <ArrowLeft className="w-4 h-4" />
                            <span className="text-sm font-medium hidden sm:inline">Back</span>
                        </button>
                        <div className="h-4 w-px bg-neutral-200 hidden sm:block" />
                        <div className="flex items-center gap-2">
                            <span className="text-lg">🎾</span>
                            <h1 className="text-sm font-semibold text-black capitalize">{strokeType} Analysis</h1>
                            {isDemo && (
                                <span className="px-2 py-0.5 bg-gradient-to-r from-violet-500 to-purple-600 text-white text-[10px] font-bold rounded-full flex items-center gap-1">
                                    <Sparkles className="w-3 h-3" />
                                    DEMO
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowMetrics(!showMetrics)}
                            className={`lg:hidden p-2 rounded-lg transition border ${showMetrics
                                ? 'bg-black text-white border-black'
                                : 'bg-neutral-100 text-neutral-600 border-neutral-200 hover:text-black'
                                }`}
                        >
                            <BarChart3 className="w-4 h-4" />
                        </button>

                        <button
                            onClick={() => router.push(`/strikesense/analysis?stroke=${strokeType}`)}
                            className="px-3 py-2 text-sm font-semibold bg-black text-white rounded-lg transition hover:bg-neutral-800 flex items-center gap-1.5"
                        >
                            <Award className="w-4 h-4" />
                            <span className="hidden sm:inline">Full Report</span>
                            <ChevronRight className="w-4 h-4 sm:hidden" />
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className={`relative z-10 max-w-7xl mx-auto px-3 md:px-4 py-4 md:py-6 ${isDemo ? 'pb-28' : ''}`}>
                <div className="grid lg:grid-cols-12 gap-4 md:gap-6">

                    {/* Video Player */}
                    <div className="lg:col-span-8">
                        <div className="bg-neutral-100 border border-neutral-200 rounded-xl overflow-hidden">
                            {isDemo && demoLoading ? (
                                <div className="aspect-video flex flex-col items-center justify-center bg-neutral-50">
                                    <div className="w-12 h-12 border-4 border-neutral-200 border-t-black rounded-full animate-spin mb-4"></div>
                                    <p className="text-neutral-600 font-medium">Loading demo analysis...</p>
                                    <p className="text-neutral-400 text-sm mt-1">Please wait</p>
                                </div>
                            ) : (
                                <VideoPanel
                                    videoFile={null}
                                    videoUrl={analysisData?.videoUrl || null}
                                    onVideoUpload={() => { }}
                                    analysisData={analysisData}
                                    isProcessing={false}
                                    currentTime={currentTime}
                                    onTimeUpdate={setCurrentTime}
                                    sideBySide={false}
                                    showOverlay={true}
                                    playbackSpeed={playbackSpeed}
                                    onSpeedChange={setPlaybackSpeed}
                                />
                            )}
                        </div>
                    </div>

                    {/* Metrics Sidebar */}
                    <div className={`lg:col-span-4 space-y-4 ${showMetrics ? 'block' : 'hidden lg:block'}`}>

                        {/* Tab Switcher */}
                        <div className="flex gap-1 p-1 bg-neutral-100 rounded-lg border border-neutral-200">
                            <button
                                onClick={() => setActiveTab('metrics')}
                                className={`flex-1 px-3 py-2 text-xs font-semibold rounded-md transition flex items-center justify-center gap-1.5 ${activeTab === 'metrics'
                                    ? 'bg-white text-black shadow-sm border border-neutral-200'
                                    : 'text-neutral-500 hover:text-black'
                                    }`}
                            >
                                <BarChart3 className="w-3.5 h-3.5" />
                                Metrics
                            </button>
                            <button
                                onClick={() => setActiveTab('issues')}
                                className={`flex-1 px-3 py-2 text-xs font-semibold rounded-md transition flex items-center justify-center gap-1.5 ${activeTab === 'issues'
                                    ? 'bg-white text-black shadow-sm border border-neutral-200'
                                    : 'text-neutral-500 hover:text-black'
                                    }`}
                            >
                                <Lightbulb className="w-3.5 h-3.5" />
                                Issues
                                {filterSummary && filterSummary.injury_risk_frames > 0 && (
                                    <span className="ml-1 px-1.5 py-0.5 bg-red-100 text-red-600 rounded-full text-[10px]">
                                        {filterSummary.injury_risk_frames}
                                    </span>
                                )}
                            </button>
                        </div>

                        {/* Metrics Tab */}
                        {activeTab === 'metrics' && (
                            <>
                                {/* Real-time Metrics */}
                                <BiomechanicalMetrics
                                    currentFrameMetrics={currentFrame?.metrics}
                                    aggregates={aggregates}
                                />

                                {/* Current Frame Info */}
                                <div className="bg-neutral-50 border border-neutral-200 p-4 rounded-xl">
                                    <h3 className="text-xs font-semibold text-neutral-500 mb-3 uppercase tracking-wide">
                                        Current Frame
                                    </h3>

                                    <div className="grid grid-cols-2 gap-2 mb-3">
                                        <div className="bg-white rounded-lg p-3 border border-neutral-200">
                                            <div className="text-[10px] text-neutral-400 uppercase mb-0.5">Frame</div>
                                            <div className="text-lg font-bold text-black">
                                                {currentFrame ? (currentFrame.frameIdx ?? currentFrame._calculatedIdx ?? 0) : '--'}
                                            </div>
                                        </div>
                                        <div className="bg-white rounded-lg p-3 border border-neutral-200">
                                            <div className="text-[10px] text-neutral-400 uppercase mb-0.5">Time</div>
                                            <div className="text-lg font-bold text-black">
                                                {currentFrame ? `${(currentFrame.timestampSec ?? currentFrame._calculatedTime ?? 0).toFixed(2)}s` : '--'}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Risk Status */}
                                    <div className={`rounded-lg p-3 border min-h-[72px] ${!currentFrame
                                        ? 'bg-neutral-100 border-neutral-200'
                                        : currentFrame.injury_risk === 'high'
                                            ? 'bg-red-50 border-red-200'
                                            : currentFrame.injury_risk === 'medium'
                                                ? 'bg-amber-50 border-amber-200'
                                                : 'bg-green-50 border-green-200'
                                        }`}>
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs text-neutral-500">Form Status</span>
                                            <span className={`text-sm font-bold uppercase ${!currentFrame
                                                ? 'text-neutral-400'
                                                : currentFrame.injury_risk === 'high'
                                                    ? 'text-red-600'
                                                    : currentFrame.injury_risk === 'medium'
                                                        ? 'text-amber-600'
                                                        : 'text-green-600'
                                                }`}>
                                                {!currentFrame
                                                    ? '-- Waiting'
                                                    : currentFrame.injury_risk === 'high'
                                                        ? '⚠️ High Risk'
                                                        : currentFrame.injury_risk === 'medium'
                                                            ? '⚡ Medium'
                                                            : '✓ Good'}
                                            </span>
                                        </div>
                                        <p className={`text-xs mt-2 leading-relaxed min-h-[16px] ${!currentFrame
                                            ? 'text-neutral-400'
                                            : currentFrame.injury_risk === 'high'
                                                ? 'text-red-600'
                                                : currentFrame.injury_risk === 'medium'
                                                    ? 'text-amber-600'
                                                    : 'text-green-600'
                                            }`}>
                                            {!currentFrame
                                                ? 'Play video to see form analysis'
                                                : currentFrame.feedback && currentFrame.feedback.length > 0
                                                    ? currentFrame.feedback[0]
                                                    : 'No feedback for this frame'}
                                        </p>
                                    </div>
                                </div>

                                {/* Session Summary */}
                                <div className="bg-neutral-50 border border-neutral-200 p-4 rounded-xl">
                                    <h3 className="text-xs font-semibold text-neutral-500 mb-3 uppercase tracking-wide">
                                        Session Summary
                                    </h3>
                                    <div className="grid grid-cols-2 gap-2">
                                        <div className="bg-white rounded-lg p-3 border border-neutral-200">
                                            <div className="text-[10px] text-neutral-400 uppercase mb-0.5">Frames</div>
                                            <div className="text-lg font-bold text-black">
                                                {analysisData?.summary?.total_frames || analysisData?.frames?.length || '--'}
                                            </div>
                                        </div>
                                        <div className="bg-white rounded-lg p-3 border border-neutral-200">
                                            <div className="text-[10px] text-neutral-400 uppercase mb-0.5">Duration</div>
                                            <div className="text-lg font-bold text-black">
                                                {analysisData?.summary?.duration_sec?.toFixed(1) || '--'}s
                                            </div>
                                        </div>
                                        <div className="bg-white rounded-lg p-3 border border-neutral-200">
                                            <div className="text-[10px] text-neutral-400 uppercase mb-0.5">FPS</div>
                                            <div className="text-lg font-bold text-black">
                                                {analysisData?.summary?.fps || '--'}
                                            </div>
                                        </div>
                                        <div className="bg-white rounded-lg p-3 border border-neutral-200">
                                            <div className="text-[10px] text-neutral-400 uppercase mb-0.5">Stroke</div>
                                            <div className="text-lg font-bold text-black capitalize">
                                                {strokeType}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {/* Issues Tab */}
                        {activeTab === 'issues' && (
                            <>
                                {/* Issues Summary */}
                                {filterSummary && (
                                    <div className="bg-neutral-50 border border-neutral-200 p-4 rounded-xl">
                                        <h3 className="text-xs font-semibold text-neutral-500 mb-3 uppercase tracking-wide">
                                            Issues Found for {strokeType}
                                        </h3>
                                        <div className="grid grid-cols-3 gap-2">
                                            <div className="bg-red-50 rounded-lg p-3 border border-red-200 text-center">
                                                <div className="text-lg font-bold text-red-600">{filterSummary.injury_risk_frames}</div>
                                                <div className="text-[10px] text-red-500 uppercase">High Risk</div>
                                            </div>
                                            <div className="bg-amber-50 rounded-lg p-3 border border-amber-200 text-center">
                                                <div className="text-lg font-bold text-amber-600">{filterSummary.improvement_frames}</div>
                                                <div className="text-[10px] text-amber-500 uppercase">Improve</div>
                                            </div>
                                            <div className="bg-green-50 rounded-lg p-3 border border-green-200 text-center">
                                                <div className="text-lg font-bold text-green-600">{filterSummary.good_form_frames}</div>
                                                <div className="text-[10px] text-green-500 uppercase">Good</div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Top Issues */}
                                <div className="bg-neutral-50 border border-neutral-200 p-4 rounded-xl">
                                    <h3 className="text-xs font-semibold text-neutral-500 mb-3 uppercase tracking-wide flex items-center gap-2">
                                        <TrendingUp className="w-3.5 h-3.5" />
                                        Key Issues to Address
                                    </h3>

                                    {topIssues.length > 0 ? (
                                        <div className="space-y-2">
                                            {topIssues.map((issue, idx) => {
                                                const coaching = getFullRecommendation(issue.issue, strokeType, issue.severity);
                                                const isExpanded = selectedIssue === issue.issue;

                                                return (
                                                    <div key={idx} className="space-y-2">
                                                        <button
                                                            onClick={() => setSelectedIssue(isExpanded ? null : issue.issue)}
                                                            className={`w-full text-left p-3 rounded-lg border transition hover:scale-[1.01] active:scale-[0.99] ${issue.severity === 'high'
                                                                ? 'bg-red-50 border-red-200 hover:bg-red-100'
                                                                : issue.severity === 'medium'
                                                                    ? 'bg-amber-50 border-amber-200 hover:bg-amber-100'
                                                                    : 'bg-white border-neutral-200 hover:bg-neutral-100'
                                                                }`}
                                                        >
                                                            <div className="flex items-start justify-between gap-2">
                                                                <div className="flex items-start gap-2">
                                                                    {issue.severity === 'high' ? (
                                                                        <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                                                                    ) : issue.severity === 'medium' ? (
                                                                        <TrendingUp className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                                                                    ) : (
                                                                        <Lightbulb className="w-4 h-4 text-neutral-400 mt-0.5 flex-shrink-0" />
                                                                    )}
                                                                    <div>
                                                                        <div className={`text-xs font-semibold ${issue.severity === 'high' ? 'text-red-700'
                                                                            : issue.severity === 'medium' ? 'text-amber-700'
                                                                                : 'text-neutral-700'
                                                                            }`}>
                                                                            {coaching.title}
                                                                        </div>
                                                                        <div className="text-[10px] text-neutral-500 mt-0.5">
                                                                            {issue.frameCount} frame{issue.frameCount !== 1 ? 's' : ''} • Tap for tips
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                                <ChevronRight className={`w-4 h-4 text-neutral-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                                                            </div>
                                                        </button>

                                                        {/* Expanded Coaching Tips */}
                                                        {isExpanded && (
                                                            <div className="ml-2 p-3 bg-blue-50 border border-blue-200 rounded-lg space-y-3">
                                                                {/* Action */}
                                                                <div>
                                                                    <div className="text-[10px] font-semibold text-blue-600 uppercase mb-1">What to Change</div>
                                                                    <p className="text-xs text-neutral-700 leading-relaxed">{coaching.message}</p>
                                                                </div>

                                                                {/* Drill */}
                                                                <div className="bg-white rounded-lg p-2.5 border border-blue-100">
                                                                    <div className="text-[10px] font-semibold text-green-600 uppercase mb-1">🏋️ Try This Drill</div>
                                                                    <p className="text-xs text-neutral-700 leading-relaxed">{coaching.drill}</p>
                                                                </div>

                                                                {/* Benefit */}
                                                                <div>
                                                                    <div className="text-[10px] font-semibold text-purple-600 uppercase mb-1">Why It Matters</div>
                                                                    <p className="text-xs text-neutral-600 leading-relaxed">{coaching.benefit}</p>
                                                                </div>

                                                                {/* Frame Preview Thumbnail */}
                                                                {analysisData?.videoUrl && (
                                                                    <div>
                                                                        <div className="text-[10px] font-semibold text-neutral-500 uppercase mb-1.5">📹 Frame Preview</div>
                                                                        <div
                                                                            className="relative rounded-lg overflow-hidden border border-neutral-300 cursor-pointer group"
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                jumpToFrame(issue.firstFrame);
                                                                            }}
                                                                        >
                                                                            <video
                                                                                src={analysisData.videoUrl}
                                                                                className="w-full h-auto max-h-32 object-cover bg-neutral-200"
                                                                                muted
                                                                                playsInline
                                                                                preload="metadata"
                                                                                onLoadedMetadata={(e) => {
                                                                                    const video = e.currentTarget;
                                                                                    video.currentTime = issue.firstFrame?.timestampSec || 0;
                                                                                }}
                                                                            />
                                                                            <div className="absolute inset-0 bg-black/30 group-hover:bg-black/10 transition flex items-center justify-center">
                                                                                <div className="w-8 h-8 rounded-full bg-white/90 flex items-center justify-center shadow-lg">
                                                                                    <Play className="w-4 h-4 text-black ml-0.5" />
                                                                                </div>
                                                                            </div>
                                                                            <div className="absolute bottom-1 right-1 bg-black/70 text-white text-[10px] px-1.5 py-0.5 rounded font-mono">
                                                                                {(issue.firstFrame?.timestampSec || 0).toFixed(2)}s
                                                                            </div>
                                                                            <div className="absolute bottom-1 left-1 bg-red-600 text-white text-[10px] px-1.5 py-0.5 rounded font-semibold">
                                                                                Issue Frame
                                                                            </div>
                                                                        </div>
                                                                        <p className="text-[10px] text-neutral-400 mt-1 text-center">Tap to view in player</p>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="text-center py-6">
                                            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                                                <span className="text-2xl">✓</span>
                                            </div>
                                            <p className="text-sm font-medium text-green-700">Great Form!</p>
                                            <p className="text-xs text-neutral-500 mt-1">No significant issues detected</p>
                                        </div>
                                    )}
                                </div>

                                {/* Problem Frames List */}
                                {filteredFrames.length > 0 && (
                                    <div className="bg-neutral-50 border border-neutral-200 p-4 rounded-xl max-h-[300px] overflow-y-auto">
                                        <h3 className="text-xs font-semibold text-neutral-500 mb-3 uppercase tracking-wide sticky top-0 bg-neutral-50 py-1">
                                            Problem Frames ({filteredFrames.filter(f => f.category !== 'good_form').length})
                                        </h3>
                                        <div className="space-y-1.5">
                                            {filteredFrames
                                                .filter(f => f.category !== 'good_form')
                                                .slice(0, 20)
                                                .map((ff, idx) => (
                                                    <button
                                                        key={idx}
                                                        onClick={() => jumpToFrame(ff.frame)}
                                                        className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition ${ff.category === 'injury_risk'
                                                            ? 'bg-red-50 hover:bg-red-100 border border-red-200'
                                                            : ff.category === 'form_improvement'
                                                                ? 'bg-amber-50 hover:bg-amber-100 border border-amber-200'
                                                                : 'bg-white hover:bg-neutral-100 border border-neutral-200'
                                                            }`}
                                                    >
                                                        <div className="flex items-center gap-2">
                                                            <span className={`text-xs font-mono font-bold ${ff.category === 'injury_risk' ? 'text-red-600'
                                                                : ff.category === 'form_improvement' ? 'text-amber-600'
                                                                    : 'text-neutral-600'
                                                                }`}>
                                                                #{ff.frame.frameIdx}
                                                            </span>
                                                            <span className="text-[10px] text-neutral-500">
                                                                {ff.frame.timestampSec.toFixed(2)}s
                                                            </span>
                                                        </div>
                                                        <div className="text-[10px] text-neutral-500">
                                                            {ff.issues.length} issue{ff.issues.length !== 1 ? 's' : ''}
                                                        </div>
                                                    </button>
                                                ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>

                {/* Mobile FAB */}
                <div className="fixed bottom-4 right-4 lg:hidden z-30">
                    <button
                        onClick={() => router.push(`/strikesense/analysis?stroke=${strokeType}`)}
                        className="w-14 h-14 bg-black rounded-full shadow-lg flex items-center justify-center active:scale-95 transition-transform"
                    >
                        <Award className="w-6 h-6 text-white" />
                    </button>
                </div>
            </main>

            {/* Demo CTA Banner */}
            {isDemo && (
                <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-r from-violet-600 to-purple-700 text-white py-4 px-4 z-40 shadow-lg">
                    <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
                        <div className="text-center sm:text-left">
                            <p className="font-bold text-sm sm:text-base">Ready to analyze your own game?</p>
                            <p className="text-xs sm:text-sm text-white/80">Get personalized insights for your pickleball strokes</p>
                        </div>
                        <button
                            onClick={() => router.push('/')}
                            className="px-6 py-2.5 bg-white text-purple-700 font-bold text-sm rounded-xl hover:bg-neutral-100 transition whitespace-nowrap"
                        >
                            Try Your Own Video →
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default function PlayerPage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="text-black font-bold animate-pulse">Loading...</div>
            </div>
        }>
            <PlayerContent />
        </Suspense>
    );
}
