"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import Header from "@/app/components/Header";
import AICoachModal from "@/app/components/dashboard/AICoachModal";
import BioSkeleton from "@/app/components/dashboard/BioSkeleton";
import { EmptyState } from "@/app/components/ui/EmptyState";
import {
    ArrowLeft,
    BarChart3,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    Activity,
    Target,
    Clock,
    ChevronRight,
    Loader2,
    Zap,
    CheckCircle,
    XCircle,
    Minus,
    Dumbbell,
    Sparkles,
} from "lucide-react";

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
    trend: number;
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
    trends: TrendPoint[];
    strokeBreakdown: StrokeBreakdown[];
    metricCards: MetricCard[];
    drills: DrillRecommendation[];
}

const STROKE_ICONS: Record<string, string> = {
    serve: "🎾",
    dink: "🤏",
    groundstroke: "💪",
    overhead: "⚡",
    volley: "🏓",
};

const STROKE_LABELS: Record<string, string> = {
    serve: "Serve",
    dink: "Dink",
    groundstroke: "Drive",
    overhead: "Overhead",
    volley: "Volley",
};

const RISK_BG_COLORS: Record<string, string> = {
    low: "bg-emerald-100 text-emerald-700",
    medium: "bg-amber-100 text-amber-700",
    high: "bg-red-100 text-red-700",
};

const STATUS_ICONS: Record<string, React.ReactNode> = {
    good: <CheckCircle className="w-4 h-4 text-emerald-500" />,
    warning: <Minus className="w-4 h-4 text-amber-500" />,
    critical: <XCircle className="w-4 h-4 text-red-500" />,
};

const STATUS_COLORS: Record<string, string> = {
    good: "border-emerald-200 bg-emerald-50",
    warning: "border-amber-200 bg-amber-50",
    critical: "border-red-200 bg-red-50",
};

export default function AnalyticsPage() {
    const router = useRouter();
    const { isSignedIn, isLoaded } = useAuth();
    const [data, setData] = useState<AnalyticsSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isAIModalOpen, setIsAIModalOpen] = useState(false);

    useEffect(() => {
        if (!isLoaded) return;
        if (!isSignedIn) {
            setLoading(false);
            return;
        }
        fetchAnalytics();
    }, [isLoaded, isSignedIn]);

    const fetchAnalytics = async () => {
        try {
            const response = await fetch("/api/analytics/summary");
            if (!response.ok) throw new Error("Failed to fetch analytics");
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffDays = Math.floor(diffMs / 86400000);
        if (diffDays === 0) return "Today";
        if (diffDays === 1) return "Yesterday";
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    };

    // Fetch full job data and navigate to player - mimics history page behavior
    const handleViewSession = async (jobId: string, strokeType: string) => {
        try {
            // Fetch the full job data from the API
            const response = await fetch(`/api/analyze/status?job_id=${jobId}`);
            if (!response.ok) throw new Error("Failed to fetch job data");
            const jobData = await response.json();

            if (jobData.result) {
                // Store the full analysis result in sessionStorage (same as history page)
                const analysisResult = {
                    ...jobData.result,
                    videoUrl: jobData.result_video_url || jobData.result?.videoUrl,
                    stroke_type: strokeType,
                    llm_response: jobData.llm_response,
                };
                sessionStorage.setItem("analysisResult", JSON.stringify(analysisResult));
                router.push(`/strikesense/player?stroke=${strokeType}&job_id=${jobId}`);
            } else {
                // Fallback - just navigate and let player page handle it
                router.push(`/strikesense/player?stroke=${strokeType}&job_id=${jobId}`);
            }
        } catch (err) {
            console.error("Error fetching job data:", err);
            // Fallback navigation
            router.push(`/strikesense/player?stroke=${strokeType}&job_id=${jobId}`);
        }
    };

    // Auth required
    if (isLoaded && !isSignedIn) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center px-4">
                <div className="text-center">
                    <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-neutral-100 to-neutral-200 rounded-2xl flex items-center justify-center shadow-lg">
                        <BarChart3 className="w-10 h-10 text-neutral-400" />
                    </div>
                    <h2 className="text-2xl font-bold mb-2 text-black">Sign In Required</h2>
                    <p className="text-neutral-500 text-sm mb-6">Join us to see your personalized stats!</p>
                    <button onClick={() => router.push("/")} className="px-8 py-3 bg-black text-white rounded-xl font-medium hover:bg-neutral-800 transition">
                        Go Home
                    </button>
                </div>
            </div>
        );
    }

    // Loading
    if (loading) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                        <div className="w-16 h-16 rounded-full border-4 border-neutral-200"></div>
                        <Loader2 className="w-16 h-16 animate-spin text-black absolute top-0 left-0" style={{ strokeDasharray: "40", strokeDashoffset: "30" }} />
                    </div>
                    <p className="text-neutral-500 font-medium">Crunching the numbers...</p>
                </div>
            </div>
        );
    }

    // Error
    if (error) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center px-4">
                <div className="text-center">
                    <div className="w-20 h-20 mx-auto mb-6 bg-red-100 rounded-2xl flex items-center justify-center">
                        <AlertTriangle className="w-10 h-10 text-red-500" />
                    </div>
                    <h2 className="text-xl font-bold mb-2 text-black">Couldn't Load Analytics</h2>
                    <p className="text-neutral-500 text-sm mb-6">{error}</p>
                    <button onClick={fetchAnalytics} className="px-8 py-3 bg-black text-white rounded-xl font-medium">
                        Try Again
                    </button>
                </div>
            </div>
        );
    }

    // Empty
    if (!data || data.totalSessions === 0) {
        return (
            <div className="min-h-screen bg-white">
                <header className="sticky top-0 z-20 bg-white border-b border-neutral-200">
                    <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-4">
                        <button onClick={() => router.push("/")} className="p-2 hover:bg-neutral-100 rounded-xl transition">
                            <ArrowLeft className="w-5 h-5 text-black" />
                        </button>
                        <h1 className="text-lg font-bold text-black">Analytics</h1>
                    </div>
                </header>
                <div className="max-w-7xl mx-auto px-4 py-8">
                    <EmptyState
                        title="Unlock Your Progress"
                        description="Complete 2 sessions to see your improvement trends over time. Discover what's holding back your game."
                        actionLabel="Upload First Video"
                        onAction={() => router.push('/strikesense/upload')}
                        variant="analytics"
                    />
                </div>
            </div>
        );
    }

    // Calculate trend direction
    const recentTrends = data.trends.filter(t => t.sessions > 0);
    const trendDirection = recentTrends.length >= 2
        ? recentTrends[recentTrends.length - 1].score - recentTrends[0].score
        : 0;

    return (
        <div className="min-h-screen bg-white pb-8">
            {/* Header */}
            <header className="sticky top-0 z-20 bg-white border-b border-neutral-200">
                <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button onClick={() => router.push("/")} className="p-2 hover:bg-neutral-100 rounded-xl transition">
                            <ArrowLeft className="w-5 h-5 text-black" />
                        </button>
                        <h1 className="text-lg font-bold text-black">Analytics Dashboard</h1>
                    </div>
                    <span className="text-xs text-neutral-400">{data.totalSessions} sessions</span>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-4 py-6 space-y-8">

                {/* Hero Score Card */}
                <section className="bg-gradient-to-br from-indigo-600 to-violet-700 rounded-3xl p-6 md:p-8 text-white relative overflow-hidden shadow-xl">
                    <div className="absolute top-0 right-0 w-40 h-40 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2"></div>
                    <div className="absolute bottom-0 left-0 w-32 h-32 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2"></div>

                    <div className="relative z-10">
                        <div className="flex flex-col gap-6">
                            <div>
                                <p className="text-white font-medium text-sm mb-2 opacity-90">Your Skill Score</p>
                                <div className="flex items-end gap-4 mb-4">
                                    <span className="text-6xl md:text-7xl font-bold text-white drop-shadow-sm">{data.skillScore}</span>
                                    <span className="text-2xl text-indigo-100 mb-2">/ 100</span>
                                    {trendDirection !== 0 && (
                                        <div className={`flex items-center gap-1 mb-3 ${trendDirection > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {trendDirection > 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                                            <span className="text-sm font-medium">{Math.abs(trendDirection)}%</span>
                                        </div>
                                    )}
                                </div>
                                <div className="flex flex-wrap gap-4 text-sm mt-4">
                                    <div className="bg-white/10 px-4 py-2 rounded-full border border-white/10 backdrop-blur-sm">
                                        <span className="text-indigo-100">Sessions: </span>
                                        <span className="font-bold text-white">{data.totalSessions}</span>
                                    </div>
                                    <div className="bg-white/10 px-4 py-2 rounded-full border border-white/10 backdrop-blur-sm">
                                        <span className="text-indigo-100">Top Stroke: </span>
                                        <span className="font-bold text-white">{STROKE_ICONS[data.favoriteStroke || "serve"]} {STROKE_LABELS[data.favoriteStroke || "serve"] || data.favoriteStroke}</span>
                                    </div>
                                    <div className={`px-4 py-2 rounded-full ${data.averageRisk === 'low' ? 'bg-emerald-500/20 text-emerald-300' : data.averageRisk === 'medium' ? 'bg-amber-500/20 text-amber-300' : 'bg-red-500/20 text-red-300'}`}>
                                        <span className="font-semibold capitalize">{data.averageRisk} Risk</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="mt-6 pt-6 border-t border-white/10 flex justify-center sm:justify-end">
                            <button
                                onClick={() => setIsAIModalOpen(true)}
                                className="flex items-center gap-2 text-indigo-900 hover:text-indigo-700 transition font-bold group bg-white hover:bg-indigo-50 px-5 py-2.5 rounded-xl shadow-lg hover:shadow-xl hover:-translate-y-0.5"
                            >
                                <Sparkles className="w-4 h-4 group-hover:scale-110 transition-transform text-indigo-600" />
                                Get AI Coach Insights
                            </button>
                        </div>
                    </div>
                </section>

                {/* Kinematic Chain Status Section */}
                <section className="bg-white border border-neutral-200 rounded-3xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50 rounded-full -translate-y-1/2 translate-x-1/2 z-0"></div>
                    <div className="relative z-10 flex flex-col sm:flex-row items-center justify-between gap-6">
                        <div>
                            <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                                <Dumbbell className="w-4 h-4" /> Kinematic Chain Status
                            </h2>
                            <p className="text-neutral-600 text-sm max-w-sm">
                                Visual representation of your biomechanical stress points. <span className="text-emerald-600 font-medium">Green</span> indicates optimal form, while <span className="text-red-500 font-medium">Red</span> highlights areas needing attention.
                            </p>
                        </div>
                        <div className="flex justify-center p-4 bg-neutral-50 rounded-2xl border border-neutral-100 shadow-sm w-full sm:w-auto min-w-[200px]">
                            <BioSkeleton risks={data.riskBreakdown} className="h-40 w-auto" />
                        </div>
                    </div>
                </section>

                {/* Weekly Trend Chart */}
                <section>
                    <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" /> Weekly Progress
                    </h2>
                    <div className="bg-white border border-neutral-200 rounded-2xl p-5">
                        <div className="flex items-end justify-between gap-2 h-32">
                            {data.trends.map((point, idx) => (
                                <div key={idx} className="flex-1 flex flex-col items-center gap-2">
                                    <div className="w-full flex flex-col items-center justify-end h-24">
                                        {point.sessions > 0 ? (
                                            <div
                                                className="w-full max-w-[40px] bg-gradient-to-t from-black to-neutral-600 rounded-t-lg transition-all hover:from-neutral-800"
                                                style={{ height: `${Math.max(point.score, 10)}%` }}
                                            >
                                                <div className="text-[10px] text-white font-bold text-center pt-1">
                                                    {point.score}
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="w-full max-w-[40px] h-2 bg-neutral-100 rounded-lg"></div>
                                        )}
                                    </div>
                                    <span className="text-[10px] text-neutral-400 font-medium">{point.date}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Metrics Cards */}
                {data.metricCards.length > 0 && (
                    <section>
                        <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Activity className="w-4 h-4" /> Your Technique Metrics
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            {data.metricCards.map((metric) => (
                                <div key={metric.key} className={`border-2 rounded-2xl p-4 ${STATUS_COLORS[metric.status]}`}>
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-medium text-neutral-700">{metric.label}</span>
                                        {STATUS_ICONS[metric.status]}
                                    </div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <h3 className="text-sm font-medium text-neutral-600">Trend</h3>
                                        {metric.trend >= 5 && (
                                            <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full animate-pulse border border-emerald-100 flex items-center gap-1">
                                                <TrendingUp className="w-3 h-3" /> Improving
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-baseline gap-1 mb-2">
                                        <span className="text-3xl font-bold text-black">
                                            {metric.current !== null ? metric.current : "--"}
                                        </span>
                                        <span className="text-lg text-neutral-500">{metric.unit}</span>
                                    </div>
                                    <div className="text-xs text-neutral-500">
                                        Optimal: {metric.optimal_min}-{metric.optimal_max}{metric.unit}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* Stroke Breakdown */}
                {data.strokeBreakdown.length > 0 && (
                    <section>
                        <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Target className="w-4 h-4" /> Performance by Stroke
                        </h2>
                        <div className="bg-white border border-neutral-200 rounded-2xl divide-y divide-neutral-100">
                            {data.strokeBreakdown.map((stroke) => (
                                <div key={stroke.stroke_type} className="p-4 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 rounded-xl bg-black flex items-center justify-center text-2xl">
                                            {STROKE_ICONS[stroke.stroke_type] || "🏓"}
                                        </div>
                                        <div>
                                            <p className="font-semibold text-black capitalize">
                                                {STROKE_LABELS[stroke.stroke_type] || stroke.stroke_type}
                                            </p>
                                            <p className="text-xs text-neutral-500">{stroke.sessions} session{stroke.sessions !== 1 ? "s" : ""}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="text-right">
                                            <p className="text-lg font-bold text-black">{stroke.avg_score}%</p>
                                            <span className={`text-[10px] px-2 py-0.5 rounded-full ${RISK_BG_COLORS[stroke.avg_risk]}`}>
                                                {stroke.avg_risk}
                                            </span>
                                        </div>
                                        <ChevronRight className="w-4 h-4 text-neutral-300" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* Risk Breakdown */}
                <section>
                    <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" /> Risk Analysis
                    </h2>
                    <div className="bg-white border border-neutral-200 rounded-2xl p-5 space-y-5">
                        {[
                            { label: "Shoulder Overuse", value: data.riskBreakdown.shoulder_overuse, threshold: 10 },
                            { label: "Poor Kinetic Chain", value: data.riskBreakdown.poor_kinetic_chain, threshold: 15 },
                            { label: "Knee Stress", value: data.riskBreakdown.knee_stress, threshold: 10 },
                        ].map((risk) => (
                            <div key={risk.label}>
                                <div className="flex justify-between text-sm mb-2">
                                    <span className="font-medium text-neutral-700">{risk.label}</span>
                                    <span className={`font-bold ${risk.value > risk.threshold ? 'text-red-500' : risk.value > risk.threshold / 2 ? 'text-amber-500' : 'text-emerald-500'}`}>
                                        {risk.value.toFixed(1)}%
                                    </span>
                                </div>
                                <div className="h-3 bg-neutral-100 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all ${risk.value > risk.threshold ? "bg-gradient-to-r from-red-400 to-red-500" :
                                            risk.value > risk.threshold / 2 ? "bg-gradient-to-r from-amber-400 to-amber-500" :
                                                "bg-gradient-to-r from-emerald-400 to-emerald-500"
                                            }`}
                                        style={{ width: `${Math.min(risk.value * 3, 100)}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Drill Recommendations */}
                {data.drills.length > 0 && (
                    <section>
                        <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Dumbbell className="w-4 h-4" /> Recommended Drills
                        </h2>
                        <div className="space-y-3">
                            {data.drills.map((drill, idx) => (
                                <div key={idx} className="bg-gradient-to-r from-neutral-50 to-white border border-neutral-200 rounded-2xl p-5">
                                    <div className="flex items-start justify-between mb-2">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-full bg-black text-white flex items-center justify-center text-sm font-bold">
                                                {drill.priority}
                                            </div>
                                            <h3 className="font-bold text-black">{drill.name}</h3>
                                        </div>
                                        <span className="text-xs bg-neutral-100 px-3 py-1 rounded-full text-neutral-600">
                                            {drill.duration}
                                        </span>
                                    </div>
                                    <p className="text-sm text-neutral-600 ml-11">{drill.description}</p>

                                    {/* Drill Animation Preview */}
                                    <div className="mt-4 ml-11 p-3 bg-black/5 rounded-xl border border-black/5 flex items-center gap-4">
                                        <div className="h-24 w-16 shrink-0 relative">
                                            {/* Derive reasonable drill key from name or issue */}
                                            <BioSkeleton
                                                mode="demo"
                                                drill={
                                                    drill.issue === 'poor_kinetic_chain' ? 'hip_drive' :
                                                        drill.issue === 'shoulder_overuse' ? 'low_contact' :
                                                            drill.issue === 'elbow_strain' ? 'arm_extension' :
                                                                'athletic_stance' // default/knee_stress
                                                }
                                                className="h-full w-full"
                                            />
                                        </div>
                                        <div>
                                            <p className="text-xs font-bold text-black mb-1">Visual Guide</p>
                                            <p className="text-[10px] text-neutral-500 leading-relaxed">
                                                Watch the skeleton to understand the correct form.
                                                Focus on keeping your spine straight and knees aligned.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* Recent Sessions */}
                <section>
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-sm font-semibold text-neutral-500 uppercase tracking-wider flex items-center gap-2">
                            <Clock className="w-4 h-4" /> Recent Analyses
                        </h2>
                        <button onClick={() => router.push("/strikesense/history")} className="text-xs text-neutral-500 hover:text-black flex items-center gap-1 transition">
                            View All <ChevronRight className="w-3 h-3" />
                        </button>
                    </div>
                    <div className="bg-white border border-neutral-200 rounded-2xl divide-y divide-neutral-100">
                        {data.recentSessions.map((session) => (
                            <button
                                key={session.id}
                                onClick={() => handleViewSession(session.id, session.stroke_type)}
                                className="w-full p-4 flex items-center justify-between hover:bg-neutral-50 transition"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="w-11 h-11 rounded-xl bg-black flex items-center justify-center text-xl">
                                        {STROKE_ICONS[session.stroke_type] || "🏓"}
                                    </div>
                                    <div className="text-left">
                                        <p className="font-medium text-black capitalize">{STROKE_LABELS[session.stroke_type] || session.stroke_type}</p>
                                        <p className="text-xs text-neutral-500">{formatDate(session.created_at)}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="text-right">
                                        <p className="font-bold text-black">{session.score}%</p>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-neutral-300" />
                                </div>
                            </button>
                        ))}
                    </div>
                </section>

            </main>
            {data && <AICoachModal isOpen={isAIModalOpen} onClose={() => setIsAIModalOpen(false)} analyticsData={data} />}
        </div>
    );
}
