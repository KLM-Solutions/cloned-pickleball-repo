import { useState, useMemo } from "react";
import { AnalyzeResponse, ApiFrame } from "../../types";
import VideoPanel from "../VideoPanel";
import { Activity, AlertTriangle, TrendingUp, User, Layout, List, CheckCircle, ArrowLeft, Zap, Code, Copy, Check, X } from "lucide-react";
import { StatCard } from "./StatCard";
import { BiomechanicalMetrics } from "./BiomechanicalMetrics";

interface ResultsDashboardProps {
    result: AnalyzeResponse;
    videoFile: File | null;
    onReset: () => void;
}

export default function ResultsDashboard({ result, videoFile, onReset }: ResultsDashboardProps) {
    const [currentTime, setCurrentTime] = useState(0);
    const [viewMode, setViewMode] = useState<'analysis' | 'compare'>('analysis');
    const [showJson, setShowJson] = useState(false);
    const [copied, setCopied] = useState(false);

    const copyJson = () => {
        navigator.clipboard.writeText(JSON.stringify(result, null, 2));
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    // Find current frame data based on currentTime
    const currentFrame = useMemo(() => {
        if (!result.frames || result.frames.length === 0) return null;
        // Find closest frame
        return result.frames.reduce((prev, curr) => {
            return Math.abs(curr.timestampSec - currentTime) < Math.abs(prev.timestampSec - currentTime) ? curr : prev;
        });
    }, [result.frames, currentTime]);

    // Calculate aggregates (memoized)
    const aggregates = useMemo(() => {
        const frames = result.frames || [];
        if (frames.length === 0) return { hip: 0, shoulder: 0, knee: 0 };
        return {
            hip: Math.round(frames.reduce((acc, f) => acc + (f.metrics?.hip_rotation_deg || 0), 0) / frames.length),
            shoulder: Math.round(Math.max(...frames.map(f => f.metrics?.right_shoulder_abduction || 0))),
            knee: Math.round(Math.min(...frames.map(f => f.metrics?.right_knee_flexion || 180))),
        };
    }, [result.frames]);

    // Get stroke color
    const getStrokeColor = (type: string) => {
        switch (type) {
            case 'serve': return 'from-emerald-500 to-teal-600';
            case 'dink': return 'from-violet-500 to-purple-600';
            case 'groundstroke': return 'from-orange-500 to-red-500';
            case 'overhead': return 'from-blue-500 to-indigo-600';
            default: return 'from-slate-500 to-slate-600';
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
            {/* Animated background */}
            <div className="fixed inset-0 opacity-20 pointer-events-none">
                <div className="absolute top-20 left-10 w-72 h-72 bg-emerald-500 rounded-full filter blur-[128px]" />
                <div className="absolute bottom-20 right-10 w-96 h-96 bg-violet-500 rounded-full filter blur-[128px]" />
            </div>

            {/* JSON Modal */}
            {showJson && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
                    <div className="bg-slate-900 border border-white/10 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden shadow-2xl">
                        {/* Modal Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/5">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                                    <Code className="w-4 h-4 text-cyan-400" />
                                </div>
                                <div>
                                    <h3 className="font-bold text-white">Analysis JSON Output</h3>
                                    <p className="text-xs text-slate-500">
                                        {result.frames?.length || 0} frames • {(JSON.stringify(result).length / 1024).toFixed(1)}KB
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={copyJson}
                                    className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm font-medium transition border border-cyan-500/30"
                                >
                                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                                    {copied ? 'Copied!' : 'Copy'}
                                </button>
                                <button
                                    onClick={() => setShowJson(false)}
                                    className="p-2 hover:bg-white/10 rounded-lg transition text-slate-400 hover:text-white"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                        </div>

                        {/* JSON Content */}
                        <div className="flex-1 overflow-auto p-4 bg-slate-950">
                            <pre className="text-xs font-mono text-slate-300 leading-relaxed whitespace-pre-wrap">
                                <code>{JSON.stringify(result, null, 2)}</code>
                            </pre>
                        </div>

                        {/* Modal Footer */}
                        <div className="px-6 py-3 border-t border-white/10 bg-white/5 flex items-center justify-between">
                            <div className="flex items-center gap-4 text-xs text-slate-500">
                                <span>Frames: <strong className="text-cyan-400">{result.frames?.length || 0}</strong></span>
                                <span>Strokes: <strong className="text-cyan-400">{result.strokes?.length || 0}</strong></span>
                                <span>Type: <strong className="text-cyan-400 capitalize">{result.stroke_type || 'N/A'}</strong></span>
                            </div>
                            <button
                                onClick={() => setShowJson(false)}
                                className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg text-sm font-medium transition border border-white/10"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Header */}
            <header className="relative z-10 border-b border-white/10 backdrop-blur-sm">
                <div className="max-w-7xl mx-auto px-4 md:px-6 py-4">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div className="flex items-center gap-4">
                            {/* Back Button */}
                            <button
                                onClick={onReset}
                                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300 group"
                            >
                                <ArrowLeft className="w-4 h-4 text-slate-400 group-hover:text-white transition-colors" />
                                <span className="text-sm font-medium text-slate-400 group-hover:text-white transition-colors">Back</span>
                            </button>

                            <div className="flex items-center gap-3">
                                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${getStrokeColor(result.stroke_type)} flex items-center justify-center shadow-lg`}>
                                    <Zap className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h1 className="text-xl font-bold text-white">My Results</h1>
                                    <p className="text-xs text-slate-400 capitalize">
                                        {result.stroke_type} • {result.frames.length} frames • {result.strokes?.length || 0} strokes
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => setShowJson(!showJson)}
                                className={`px-3 py-2 text-sm font-bold rounded-lg transition-all flex items-center gap-2 ${showJson
                                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                                    : 'bg-white/5 text-slate-400 border border-white/10 hover:text-white hover:bg-white/10'
                                    }`}
                            >
                                <Code className="w-4 h-4" />
                                <span className="hidden sm:inline">JSON</span>
                            </button>
                            <div className="bg-white/5 p-1 rounded-lg border border-white/10 flex">
                                <button
                                    onClick={() => setViewMode('analysis')}
                                    className={`px-3 py-1.5 text-xs md:text-sm font-medium rounded-md transition flex items-center gap-2 ${viewMode === 'analysis' ? 'bg-white/10 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
                                >
                                    <Layout className="w-4 h-4" /> Analysis
                                </button>
                                <button
                                    onClick={() => setViewMode('compare')}
                                    className={`px-3 py-1.5 text-xs md:text-sm font-medium rounded-md transition flex items-center gap-2 ${viewMode === 'compare' ? 'bg-white/10 text-white shadow-sm' : 'text-slate-400 hover:text-white'}`}
                                >
                                    <List className="w-4 h-4" /> Split View
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 max-w-7xl mx-auto px-4 md:px-6 py-6">

                <div className="grid lg:grid-cols-12 gap-6">

                    {/* LEFT COLUMN: VIDEO PLAYER (8 cols) */}
                    <div className="lg:col-span-8 space-y-4">
                        <div className="bg-slate-800/50 p-1 rounded-xl border border-white/10 overflow-hidden h-auto md:h-[500px] shadow-xl backdrop-blur-sm">
                            <VideoPanel
                                videoFile={videoFile}
                                videoUrl={result.videoUrl || null}
                                onVideoUpload={() => { }}
                                analysisData={result}
                                isProcessing={false}
                                showOverlay={true}
                                currentTime={currentTime}
                                onTimeUpdate={setCurrentTime}
                                sideBySide={viewMode === 'compare'}
                            />
                        </div>

                        {/* Ball Stats Row */}
                        {result.ballStats && (
                            <div className="grid grid-cols-3 gap-4">
                                <StatCard metric="Avg Speed" value={result.ballStats.avgSpeedKmh || 0} target={80} unit=" km/h" />
                                <StatCard metric="Max Speed" value={result.ballStats.maxSpeedKmh || 0} target={100} unit=" km/h" />
                                <StatCard metric="Distance" value={result.ballStats.totalDistanceMeters || 0} target={20} unit=" m" />
                            </div>
                        )}

                        {/* KEY MOMENTS STRIP (Dynamic) */}
                        <div className="bg-slate-800/50 p-4 rounded-xl border border-white/10 backdrop-blur-sm shadow-lg">
                            <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                                <Activity className="w-3 h-3 text-cyan-400" /> Key Moments
                            </h3>
                            <div className="grid grid-cols-5 gap-2">
                                {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
                                    // 1. Determine active stroke context
                                    const activeStroke = result.strokes?.find(s => currentTime >= s.startSec && currentTime <= ((s.end_frame || 0) / 30));
                                    const frames = result.frames || [];
                                    if (frames.length === 0) return null;

                                    // 2. Calculate target frame index
                                    let targetFrameIdx = 0;
                                    let label = "";

                                    if (activeStroke) {
                                        // If inside a stroke, show ITS phases
                                        const start = activeStroke.start_frame || 0;
                                        const end = activeStroke.end_frame || 0;
                                        targetFrameIdx = Math.floor(start + (end - start) * ratio);

                                        if (i === 2) {
                                            targetFrameIdx = activeStroke.peak_frame_idx || targetFrameIdx;
                                            label = "PEAK";
                                        } else if (i === 0) label = "START";
                                        else if (i === 4) label = "FINISH";
                                    } else {
                                        // Default: Whole video distribution
                                        targetFrameIdx = Math.floor(frames.length * ratio);
                                        if (targetFrameIdx >= frames.length) targetFrameIdx = frames.length - 1;
                                    }

                                    const frameData = frames.find(f => f.frame_idx === targetFrameIdx);

                                    return (
                                        <button
                                            key={`thumb-${i}`}
                                            onClick={() => frameData && setCurrentTime(frameData.timestampSec)}
                                            className={`relative aspect-video rounded-lg overflow-hidden border transition-all group ${Math.abs(currentTime - (frameData?.timestampSec || 0)) < 0.1 ? 'border-cyan-400 ring-2 ring-cyan-400/20' : 'border-white/10 hover:border-white/30'}`}
                                        >
                                            {frameData && frameData.frameFilename ? (
                                                <img
                                                    src={frameData.frameFilename.startsWith('http') ? frameData.frameFilename : `/frames/${frameData.frameFilename}`}
                                                    alt={`Frame ${targetFrameIdx}`}
                                                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
                                                />
                                            ) : (
                                                <div className="w-full h-full bg-slate-900 flex items-center justify-center">
                                                    <span className="text-[10px] text-slate-600">No Img</span>
                                                </div>
                                            )}
                                            {label && (
                                                <div className={`absolute bottom-1 right-1 text-[8px] font-bold px-1.5 py-0.5 rounded-sm ${label === 'PEAK' ? 'bg-emerald-500 text-black' : 'bg-black/60 text-white'}`}>
                                                    {label}
                                                </div>
                                            )}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    </div>

                    {/* RIGHT COLUMN: INSIGHTS & METRICS (4 cols) */}
                    <div className="lg:col-span-4 space-y-4 flex flex-col h-auto md:h-[500px]">

                        {/* Real-time Metrics */}
                        <BiomechanicalMetrics
                            currentFrameMetrics={currentFrame?.metrics}
                            aggregates={aggregates}
                        />

                        {/* REAL-TIME ENHANCED ANALYSIS */}
                        <div className="flex-none bg-slate-800/50 p-4 rounded-xl border border-white/10 backdrop-blur-sm">
                            <h3 className="text-xs font-bold text-slate-400 mb-3 flex items-center gap-2 uppercase tracking-wider">
                                <TrendingUp className="h-3 w-3 text-emerald-400" /> Live Form Check
                            </h3>

                            <div className="space-y-3">
                                {/* SAFE STATUS */}
                                <div className={`p-3 rounded-lg border flex flex-col gap-1 transition-all duration-300 ${!currentFrame?.metrics?.injury_risk || currentFrame?.metrics?.injury_risk === 'low' ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-slate-700/50 border-white/5 opacity-50'}`}>
                                    <div className="flex items-center gap-2">
                                        <span className={`text-sm font-bold ${!currentFrame?.metrics?.injury_risk || currentFrame?.metrics?.injury_risk === 'low' ? 'text-emerald-400' : 'text-slate-500'}`}>
                                            Shoulder Risk: SAFE
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <CheckCircle className={`h-4 w-4 ${!currentFrame?.metrics?.injury_risk || currentFrame?.metrics?.injury_risk === 'low' ? 'text-emerald-500' : 'text-slate-600'}`} />
                                        <span className={`text-sm ${!currentFrame?.metrics?.injury_risk || currentFrame?.metrics?.injury_risk === 'low' ? 'text-emerald-300' : 'text-slate-500'}`}>
                                            Safe form detected
                                        </span>
                                    </div>
                                </div>

                                {/* RISK ALERT (Persistent Box) */}
                                <div className={`p-3 border rounded-lg flex flex-col gap-2 transition-all duration-300 ${currentFrame?.metrics?.injury_risk && currentFrame.metrics.injury_risk !== 'low' ? 'bg-red-500/10 border-red-500/30' : 'bg-slate-700/50 border-white/5'}`}>
                                    <div className="flex items-center gap-2">
                                        {currentFrame?.metrics?.injury_risk && currentFrame.metrics.injury_risk !== 'low' ? (
                                            <AlertTriangle className="h-4 w-4 text-red-400" />
                                        ) : (
                                            <Activity className="h-4 w-4 text-slate-500" />
                                        )}
                                        <span className={`text-xs font-bold uppercase tracking-wider ${currentFrame?.metrics?.injury_risk && currentFrame.metrics.injury_risk !== 'low' ? 'text-red-400' : 'text-slate-500'}`}>
                                            {currentFrame?.metrics?.injury_risk && currentFrame.metrics.injury_risk !== 'low' ? 'RISK ALERT' : 'LIVE STATUS'}
                                        </span>
                                    </div>
                                    <p className={`text-sm font-medium ${currentFrame?.metrics?.injury_risk && currentFrame.metrics.injury_risk !== 'low' ? 'text-red-300' : 'text-slate-400'}`}>
                                        {currentFrame?.metrics?.injury_risk && currentFrame.metrics.injury_risk !== 'low'
                                            ? `Focus Area: ${currentFrame.metrics.feedback?.[0] || 'Form deviation detected'}`
                                            : 'Monitoring technique...'}
                                    </p>
                                </div>

                                {/* Fallback for no video/frame loaded */}
                                {!currentFrame && (
                                    <div className="text-center py-4 opacity-50 text-xs text-slate-400">
                                        Play video to see real-time analysis
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* STROKE PERFORMANCE (Enhanced) */}
                        <div className="flex-1 min-h-0 overflow-y-auto bg-slate-800/50 p-4 rounded-xl border border-white/10 backdrop-blur-sm">
                            <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-wider sticky top-0 bg-slate-800/90 z-10 py-2 -mt-2 -mx-1 px-1 backdrop-blur-sm flex items-center justify-between">
                                <span>My Strokes</span>
                                <span className="text-[10px] bg-white/10 px-2 py-0.5 rounded-full text-slate-300">
                                    {result.strokes?.filter(s => s.type === result.stroke_type).length || 0} Detected
                                </span>
                            </h3>
                            <div className="space-y-3 pb-2">
                                {result.strokes?.filter(s => s.type === result.stroke_type).map((s, i) => (
                                    <button
                                        key={`stroke-log-${i}`}
                                        onClick={() => setCurrentTime(s.startSec)}
                                        className="w-full text-left p-3 rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 hover:border-cyan-500/50 hover:from-cyan-500/10 hover:to-cyan-900/10 transition-all duration-300 group relative overflow-hidden"
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2">
                                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br ${getStrokeColor(s.type)} shadow-lg`}>
                                                    <Zap className="w-4 h-4 text-white" />
                                                </div>
                                                <div>
                                                    <span className="text-sm font-bold text-white capitalize block">{s.type}</span>
                                                    <span className="text-[10px] text-slate-400 font-mono">
                                                        {s.startSec.toFixed(2)}s - {((s.end_frame || 0) / 30).toFixed(2)}s
                                                    </span>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-xs font-bold text-cyan-400 bg-cyan-500/10 px-2 py-1 rounded-md border border-cyan-500/20">
                                                    Peak: {s.peak_timestamp?.toFixed(2)}s
                                                </div>
                                            </div>
                                        </div>

                                        {/* Metrics Grid */}
                                        <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-white/5">
                                            <div className="flex flex-col">
                                                <span className="text-[10px] text-slate-500 uppercase">Max Speed</span>
                                                <span className="text-xs font-bold text-white">{s.peak_velocity?.toFixed(2) || 'N/A'}</span>
                                            </div>
                                            <div className="flex flex-col items-end">
                                                <span className="text-[10px] text-slate-500 uppercase">Confidence</span>
                                                <span className="text-xs font-bold text-emerald-400">{(s.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                        </div>
                                    </button>
                                ))}

                                {(!result.strokes || result.strokes.filter(s => s.type === result.stroke_type).length === 0) && (
                                    <div className="text-center py-10 opacity-50 flex flex-col items-center">
                                        <Activity className="w-10 h-10 text-slate-700 mb-3" />
                                        <p className="text-sm font-medium text-slate-500">No {result.stroke_type}s detected.</p>
                                        <p className="text-xs text-slate-600 mt-1">Try another analysis or upload a clearer video.</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
