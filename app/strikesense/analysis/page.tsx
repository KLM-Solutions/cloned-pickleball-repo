"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
    ArrowLeft, Activity, TrendingUp, Award,
    Zap, Target, MessageSquare, Clock, Home,
    ChevronDown, ChevronUp, AlertCircle, CheckCircle2, Shield,
    Download, Loader2
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { getDeviationReport, DeviationReport, DeviationParameter } from "@/lib/analysis";

export const dynamic = 'force-dynamic';

function AnalysisContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const strokeType = searchParams.get('stroke') || 'serve';
    const [analysisData, setAnalysisData] = useState<any>(null);
    const [expandedCoach, setExpandedCoach] = useState(true);
    const [isDemo, setIsDemo] = useState(false);

    const [llmResponse, setLlmResponse] = useState<string | null>(null);
    const [pdfLoading, setPdfLoading] = useState(false);

    useEffect(() => {
        const storedResult = sessionStorage.getItem('analysisResult');
        if (storedResult) {
            try {
                const parsed = JSON.parse(storedResult);
                setAnalysisData(parsed);

                // Check if this is demo mode
                if (parsed.isDemo) {
                    setIsDemo(true);
                }

                // Use cached LLM response from database
                if (parsed.llm_response) {
                    setLlmResponse(parsed.llm_response);
                }
            } catch (e) {
                router.push('/strikesense/upload');
            }
        } else {
            setTimeout(() => router.push('/strikesense/upload'), 1000);
        }
    }, [router]);

    if (!analysisData) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center p-4">
                <div className="text-center">
                    <div className="text-black mb-2 animate-pulse">Pulling up your results...</div>
                    <button
                        onClick={() => router.push('/strikesense/upload')}
                        className="text-sm text-neutral-500 hover:text-black underline"
                    >
                        Start over
                    </button>
                </div>
            </div>
        );
    }

    const frames = analysisData.frames || [];
    const summary = analysisData.summary || {};

    const getAverage = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
    const getMax = (arr: number[]) => arr.length > 0 ? Math.max(...arr) : null;
    const getMin = (arr: number[]) => arr.length > 0 ? Math.min(...arr) : null;

    const shoulderAngles = frames.map((f: any) => f.metrics?.right_shoulder_abduction).filter((v: any) => v != null);
    const hipRotations = frames.map((f: any) => f.metrics?.hip_rotation_deg).filter((v: any) => v != null);
    const kneeFlexions = frames.map((f: any) => f.metrics?.right_knee_flexion).filter((v: any) => v != null);
    const elbowAngles = frames.map((f: any) => f.metrics?.right_elbow_flexion).filter((v: any) => v != null);

    const riskCounts = {
        high: frames.filter((f: any) => f.injury_risk === 'high').length,
        medium: frames.filter((f: any) => f.injury_risk === 'medium').length,
        low: frames.filter((f: any) => f.injury_risk === 'low').length
    };
    const totalRiskFrames = riskCounts.high + riskCounts.medium + riskCounts.low;
    const riskScore = totalRiskFrames > 0
        ? Math.round(((riskCounts.low * 100 + riskCounts.medium * 50 + riskCounts.high * 0) / totalRiskFrames))
        : 100;

    const calculateScore = (input: number[] | number, idealMin: number, idealMax: number) => {
        let value = 0;
        if (Array.isArray(input)) {
            if (input.length === 0) return 80;
            value = getAverage(input) || 0;
        } else {
            value = input;
        }

        if (value >= idealMin && value <= idealMax) return 100;
        const distance = value < idealMin ? idealMin - value : value - idealMax;
        return Math.max(0, 100 - distance * 2);
    };

    const shoulderScore = calculateScore(getMax(shoulderAngles) || 0, 60, 120);
    const hipScore = hipRotations.length > 0 ? Math.min(100, (getAverage(hipRotations) || 0) * 3) : 70;
    const kneeScore = calculateScore(kneeFlexions, 120, 170);
    const overallScore = Math.round((shoulderScore + hipScore + kneeScore + riskScore) / 4);

    const getGrade = (score: number) => {
        if (score >= 90) return 'A';
        if (score >= 80) return 'B+';
        if (score >= 70) return 'B';
        if (score >= 60) return 'C+';
        return 'C';
    };

    const grade = getGrade(overallScore);

    // Calculate deviation report
    const deviationReport: DeviationReport = getDeviationReport(frames, strokeType);

    const handleDownloadPDF = async () => {
        setPdfLoading(true);
        try {
            const response = await fetch('/api/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    strokeType,
                    grade,
                    overallScore,
                    frames: frames.length,
                    duration: summary.duration_sec || 0,
                    riskScore,
                    riskCounts,
                    shoulderScore: Math.round(shoulderScore),
                    hipScore: Math.round(hipScore),
                    kneeScore: Math.round(kneeScore),
                    elbowScore: elbowAngles.length > 0 ? Math.round(calculateScore(elbowAngles, 90, 150)) : 80,
                    avgShoulder: getMax(shoulderAngles), // Use Max for shoulder to match UI
                    minShoulder: getMin(shoulderAngles),
                    maxShoulder: getMax(shoulderAngles),
                    avgHip: getAverage(hipRotations),
                    avgKnee: getAverage(kneeFlexions),
                    avgElbow: getAverage(elbowAngles),
                    llmResponse,
                    deviationReport, // Pass deviation report to PDF
                    generatedAt: new Date().toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                    }),
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to generate PDF');
            }

            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `strikesense-report-${strokeType}-${Date.now()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('PDF download error:', error);
            alert('Couldn\'t create the PDF right now. Please try again.');
        } finally {
            setPdfLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-white">
            <div className="relative z-10 max-w-4xl mx-auto px-4 py-6 md:py-10">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <button
                        onClick={() => router.push(isDemo ? `/strikesense/player?stroke=${strokeType}&demo=true` : `/strikesense/player?stroke=${strokeType}`)}
                        className="text-neutral-500 hover:text-black flex items-center gap-2 transition font-medium group"
                    >
                        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                        <span className="text-sm">Back to Video</span>
                    </button>
                    {!isDemo && (
                        <div className="flex items-center gap-2">
                            <button
                                onClick={handleDownloadPDF}
                                disabled={pdfLoading}
                                className="p-2.5 bg-black hover:bg-neutral-800 disabled:bg-neutral-300 rounded-lg text-white transition flex items-center gap-1.5"
                                title="Download PDF Report"
                            >
                                {pdfLoading ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Download className="w-4 h-4" />
                                )}
                            </button>
                            <button
                                onClick={() => router.push('/strikesense/history')}
                                className="p-2.5 bg-neutral-100 border border-neutral-200 hover:bg-neutral-200 rounded-lg text-neutral-600 hover:text-black transition"
                            >
                                <Clock className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => router.push('/')}
                                className="p-2.5 bg-neutral-100 border border-neutral-200 hover:bg-neutral-200 rounded-lg text-neutral-600 hover:text-black transition"
                            >
                                <Home className="w-4 h-4" />
                            </button>
                        </div>
                    )}
                </div>

                {/* Hero Section */}
                <div className="text-center mb-10">
                    <div className="inline-flex items-center px-3 py-1 rounded-full bg-neutral-100 text-neutral-700 text-xs font-semibold tracking-wide uppercase mb-4 border border-neutral-200">
                        <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                        Analysis Complete
                    </div>
                    <h1 className="text-3xl md:text-4xl font-bold text-black mb-2">
                        Performance Report
                    </h1>
                    <p className="text-neutral-500 text-sm">
                        {strokeType.charAt(0).toUpperCase() + strokeType.slice(1)} Analysis • {new Date().toLocaleDateString()}
                    </p>
                </div>

                {/* Grade Card */}
                <div className="bg-neutral-50 rounded-2xl p-6 md:p-8 border border-neutral-200 mb-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-neutral-500 text-sm font-medium mb-1">Overall Score</p>
                            <div className="flex items-baseline gap-3">
                                <span className={`text-5xl md:text-6xl font-black ${overallScore >= 80 ? 'text-emerald-600' :
                                    overallScore >= 50 ? 'text-amber-600' : 'text-red-600'
                                    }`}>
                                    {grade}
                                </span>
                                <span className={`text-2xl md:text-3xl font-bold ${overallScore >= 80 ? 'text-emerald-500' :
                                    overallScore >= 50 ? 'text-amber-500' : 'text-red-500'
                                    }`}>{overallScore}%</span>
                            </div>
                        </div>
                        <div className="w-20 h-20 md:w-24 md:h-24 relative">
                            <svg className="w-full h-full transform -rotate-90">
                                <circle cx="50%" cy="50%" r="45%" fill="none" stroke="currentColor" strokeWidth="8" className="text-neutral-200" />
                                <circle
                                    cx="50%" cy="50%" r="45%" fill="none" strokeWidth="8"
                                    strokeLinecap="round"
                                    strokeDasharray={`${overallScore * 2.83} 283`}
                                    className={
                                        overallScore >= 80 ? 'stroke-emerald-500' :
                                            overallScore >= 50 ? 'stroke-amber-500' : 'stroke-red-500'
                                    }
                                />
                            </svg>
                            <div className="absolute inset-0 flex items-center justify-center">
                                <Award className={`w-8 h-8 md:w-10 md:h-10 ${overallScore >= 80 ? 'text-emerald-600' :
                                    overallScore >= 50 ? 'text-amber-600' : 'text-red-600'
                                    }`} />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                    <MetricCard
                        icon={<Target className="w-4 h-4" />}
                        label="Frames Analyzed"
                        value={frames.length.toString()}
                        subtext={`${summary.duration_sec?.toFixed(1) || '0'}s duration`}
                        valueColor="default"
                    />
                    <MetricCard
                        icon={<Shield className="w-4 h-4" />}
                        label="Form Safety"
                        value={`${riskScore}%`}
                        subtext={riskCounts.high > 0 ? `${riskCounts.high} caution` : 'Looking good'}
                        valueColor={riskScore >= 80 ? 'green' : riskScore >= 50 ? 'amber' : 'red'}
                        optimal="100%"
                    />
                    <MetricCard
                        icon={<Activity className="w-4 h-4" />}
                        label="Avg Hip Rotation"
                        value={getAverage(hipRotations)?.toFixed(0) || '--'}
                        subtext="degrees"
                        valueColor={
                            getAverage(hipRotations) !== null
                                ? (getAverage(hipRotations)! >= 30 ? 'green' : getAverage(hipRotations)! >= 15 ? 'amber' : 'red')
                                : 'default'
                        }
                        optimal=">30°"
                    />
                    <MetricCard
                        icon={<Zap className="w-4 h-4" />}
                        label="Shoulder Range"
                        value={`${getMin(shoulderAngles)?.toFixed(0) || '--'}-${getMax(shoulderAngles)?.toFixed(0) || '--'}`}
                        subtext="degrees"
                        valueColor={
                            getMax(shoulderAngles) !== null
                                ? (getMax(shoulderAngles)! >= 90 ? 'green' : getMax(shoulderAngles)! >= 60 ? 'amber' : 'red')
                                : 'default'
                        }
                        optimal="<120°"
                    />
                </div>



                {/* What to Improve Section */}
                {deviationReport.topDeviations.length > 0 && (
                    <div className="bg-neutral-50 rounded-2xl border border-neutral-200 p-5 md:p-6 mb-6">
                        <h2 className="text-lg font-semibold text-black mb-2 flex items-center gap-2">
                            <Target className="w-5 h-5 text-neutral-600" />
                            What to Improve
                        </h2>
                        <p className="text-sm text-neutral-600 mb-5">{deviationReport.summary}</p>

                        <div className="space-y-4">
                            {deviationReport.topDeviations.map((param, idx) => (
                                <div
                                    key={param.key}
                                    className={`bg-white rounded-xl p-4 border-l-4 ${param.status === 'critical' ? 'border-red-500' : 'border-amber-500'
                                        }`}
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <div className="flex items-center gap-2">
                                            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white ${param.performanceImpact === 'high' ? 'bg-red-500' : 'bg-amber-500'
                                                }`}>
                                                {idx + 1}
                                            </span>
                                            <span className="font-semibold text-black">{param.label}</span>
                                            {param.performanceImpact === 'high' && (
                                                <span className="text-[10px] font-bold text-red-600 bg-red-50 px-2 py-0.5 rounded-full">
                                                    HIGH IMPACT
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-right">
                                            <span className={`text-lg font-bold ${param.score >= 80 ? 'text-emerald-600' :
                                                param.score >= 50 ? 'text-amber-600' : 'text-red-600'
                                                }`}>
                                                {param.score}/100
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-4 mb-3 text-sm">
                                        <div className="flex items-center gap-1">
                                            <span className="text-neutral-500">Your value:</span>
                                            <span className="font-semibold text-black">{param.userValue}°</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <span className="text-neutral-500">Optimal:</span>
                                            <span className="font-medium text-emerald-600">
                                                {param.optimalRange.min}-{param.optimalRange.max}°
                                            </span>
                                        </div>
                                    </div>

                                    <p className="text-sm text-neutral-700 bg-neutral-50 p-3 rounded-lg">
                                        💡 {param.recommendation}
                                    </p>
                                </div>
                            ))}
                        </div>

                        {/* All Parameters Summary */}
                        <div className="mt-5 pt-5 border-t border-neutral-200">
                            <p className="text-xs text-neutral-500 uppercase font-semibold mb-3">All Parameters</p>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                {deviationReport.parameters.map(param => (
                                    <div
                                        key={param.key}
                                        className={`p-2 rounded-lg text-center ${param.status === 'optimal' ? 'bg-emerald-50 border border-emerald-200' :
                                            param.status === 'warning' ? 'bg-amber-50 border border-amber-200' :
                                                'bg-red-50 border border-red-200'
                                            }`}
                                    >
                                        <p className="text-[10px] text-neutral-500 uppercase">{param.label}</p>
                                        <p className={`text-lg font-bold ${param.status === 'optimal' ? 'text-emerald-600' :
                                            param.status === 'warning' ? 'text-amber-600' : 'text-red-600'
                                            }`}>
                                            {param.score}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* AI Coach Feedback */}
                <div className="bg-neutral-50 rounded-2xl border border-neutral-200 mb-6 overflow-hidden">
                    <button
                        onClick={() => setExpandedCoach(!expandedCoach)}
                        className="w-full p-5 md:p-6 flex items-center justify-between hover:bg-neutral-100 transition"
                    >
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-black rounded-xl flex items-center justify-center">
                                <MessageSquare className="w-5 h-5 text-white" />
                            </div>
                            <div className="text-left">
                                <h3 className="text-base font-semibold text-black">AI Coach Analysis</h3>
                                <p className="text-xs text-neutral-500">Personalized feedback & recommendations</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            {expandedCoach ? <ChevronUp className="w-5 h-5 text-neutral-500" /> : <ChevronDown className="w-5 h-5 text-neutral-500" />}
                        </div>
                    </button>

                    {expandedCoach && (
                        <div className="px-5 md:px-6 pb-5 md:pb-6">
                            {llmResponse ? (
                                <div className="bg-white rounded-xl p-5 border border-neutral-200">
                                    <ReactMarkdown
                                        components={{
                                            h2: ({ children }) => (
                                                <h2 className="text-lg font-bold text-black mt-5 mb-3 first:mt-0">{children}</h2>
                                            ),
                                            h3: ({ children }) => (
                                                <h3 className="text-base font-semibold text-neutral-700 mt-4 mb-2">{children}</h3>
                                            ),
                                            p: ({ children }) => (
                                                <p className="text-neutral-600 mb-3 leading-relaxed text-sm">{children}</p>
                                            ),
                                            ul: ({ children }) => (
                                                <ul className="list-disc pl-5 text-neutral-600 mb-4 space-y-1.5 text-sm marker:text-neutral-400">{children}</ul>
                                            ),
                                            ol: ({ children }) => (
                                                <ol className="list-decimal pl-5 text-neutral-600 mb-4 space-y-1.5 text-sm marker:text-neutral-500">{children}</ol>
                                            ),
                                            li: ({ children }) => (
                                                <li className="pl-1">{children}</li>
                                            ),
                                            strong: ({ children }) => (
                                                <strong className="text-black font-semibold">{children}</strong>
                                            ),
                                            hr: () => <hr className="border-neutral-200 my-5" />,
                                        }}
                                    >
                                        {llmResponse}
                                    </ReactMarkdown>
                                </div>
                            ) : (
                                <div className="bg-neutral-100 border border-neutral-200 rounded-xl p-4 text-center">
                                    <AlertCircle className="w-6 h-6 text-neutral-400 mx-auto mb-2" />
                                    <p className="text-neutral-500 text-sm">
                                        AI coaching feedback will be available once the analysis is complete.
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Actions */}
                <div className={`flex flex-col sm:flex-row gap-3 ${isDemo ? 'pb-24' : ''}`}>
                    <button
                        onClick={() => router.push(isDemo ? `/strikesense/player?stroke=${strokeType}&demo=true` : `/strikesense/player?stroke=${strokeType}`)}
                        className="flex-1 bg-neutral-100 border border-neutral-200 hover:bg-neutral-200 text-black py-3.5 rounded-xl font-semibold transition text-sm"
                    >
                        ← Review Video
                    </button>
                    {!isDemo && (
                        <button
                            onClick={() => router.push('/')}
                            className="flex-1 bg-black hover:bg-neutral-800 text-white py-3.5 rounded-xl font-semibold transition text-sm"
                        >
                            New Analysis →
                        </button>
                    )}
                </div>

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
        </div>
    );
}

function MetricCard({ icon, label, value, subtext, valueColor, optimal }: {
    icon: React.ReactNode;
    label: string;
    value: string;
    subtext: string;
    valueColor?: 'green' | 'amber' | 'red' | 'default';
    optimal?: string;
}) {
    const getColorClass = () => {
        switch (valueColor) {
            case 'green': return 'text-emerald-600';
            case 'amber': return 'text-amber-600';
            case 'red': return 'text-red-600';
            default: return 'text-black';
        }
    };

    return (
        <div className="bg-neutral-50 rounded-xl p-4 border border-neutral-200">
            <div className="flex items-center gap-1.5 text-neutral-500 mb-2">
                {icon}
                <span className="text-xs font-medium">{label}</span>
            </div>
            <div className={`text-2xl font-bold ${getColorClass()}`}>{value}</div>
            <div className="text-xs text-neutral-400 mt-0.5">{subtext}</div>
            {optimal && (
                <div className="text-[10px] text-neutral-400 mt-1.5 pt-1.5 border-t border-neutral-200">
                    Target: <span className="font-medium text-neutral-600">{optimal}</span>
                </div>
            )}
        </div>
    );
}

function MetricBar({ label, value, detail, optimal }: { label: string; value: number; detail: string; optimal?: string }) {
    const getColor = (val: number) => {
        if (val >= 80) return 'text-emerald-600';
        if (val >= 50) return 'text-amber-600';
        return 'text-red-600';
    };
    const getBarColor = (val: number) => {
        if (val >= 80) return 'bg-emerald-500';
        if (val >= 50) return 'bg-amber-500';
        return 'bg-red-500';
    };

    return (
        <div>
            <div className="flex justify-between items-baseline mb-2">
                <span className="text-sm font-medium text-neutral-600">{label}</span>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-neutral-400">{detail}</span>
                    {optimal && <span className="text-[10px] text-neutral-400 bg-neutral-100 px-1.5 py-0.5 rounded border border-neutral-200">Target: {optimal}</span>}
                    <span className={`text-sm font-bold ${getColor(value)}`}>{value}%</span>
                </div>
            </div>
            <div className="h-2 bg-neutral-200 rounded-full overflow-hidden">
                <div
                    className={`h-full ${getBarColor(value)} transition-all duration-700`}
                    style={{ width: `${value}%` }}
                />
            </div>
        </div>
    );
}

export default function AnalysisPage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="text-black font-bold animate-pulse">Loading...</div>
            </div>
        }>
            <AnalysisContent />
        </Suspense>
    );
}
