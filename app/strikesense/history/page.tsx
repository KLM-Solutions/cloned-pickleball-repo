"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useAuth, SignInButton } from "@clerk/nextjs";
import {
  ArrowLeft,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Play,
  Calendar,
  Timer,
  BarChart3,
  RefreshCw,
  Plus,
  LogIn,
  ChevronDown,
  Trash2,
  MoreVertical,
  Filter,
  ArrowUpDown,
} from "lucide-react";
import { EmptyState } from "@/app/components/ui/EmptyState";
import { getAllAnalyses, formatDate, getStrokeInfo, AnalysisJob, deleteAnalysisJob } from "@/lib/supabase-db";

type SortOption = "newest" | "oldest" | "stroke";
type FilterOption = "all" | "serve" | "dink" | "groundstroke" | "overhead";

export default function HistoryPage() {
  const router = useRouter();
  const { userId, isLoaded, isSignedIn } = useAuth();
  const [analyses, setAnalyses] = useState<AnalysisJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Sort and Filter State
  const [sortBy, setSortBy] = useState<SortOption>("newest");
  const [filterBy, setFilterBy] = useState<FilterOption>("all");
  const [showFilters, setShowFilters] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  const fetchHistory = async () => {
    if (!userId) {
      setLoading(false);
      setAnalyses([]);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await getAllAnalyses(50, userId);
      setAnalyses(data);
    } catch (err: any) {
      setError(err.message || "Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isLoaded) {
      fetchHistory();
    }
  }, [isLoaded, userId]);

  // Sorted and Filtered Analyses
  const filteredAnalyses = useMemo(() => {
    let result = [...analyses];

    // Apply filter
    if (filterBy !== "all") {
      result = result.filter((job) => job.stroke_type === filterBy);
    }

    // Apply sort
    result.sort((a, b) => {
      if (sortBy === "oldest") {
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      } else if (sortBy === "stroke") {
        return a.stroke_type.localeCompare(b.stroke_type);
      }
      // Default: newest
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });

    return result;
  }, [analyses, sortBy, filterBy]);

  const handleViewAnalysis = (job: AnalysisJob) => {
    if (job.result_json) {
      const analysisResult = {
        ...job.result_json,
        videoUrl: job.result_video_url,
        stroke_type: job.stroke_type,
        llm_response: job.llm_response,
      };
      sessionStorage.setItem("analysisResult", JSON.stringify(analysisResult));
      router.push(`/strikesense/player?stroke=${job.stroke_type}&job_id=${job.id}`);
    }
  };

  const handleDelete = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    if (!userId) return;

    if (!confirm("Are you sure you want to delete this analysis?")) return;

    setDeletingId(jobId);
    try {
      const success = await deleteAnalysisJob(jobId, userId);
      if (success) {
        setAnalyses((prev) => prev.filter((job) => job.id !== jobId));
      } else {
        alert("Failed to delete. Please try again.");
      }
    } catch (err) {
      alert("An error occurred. Please try again.");
    } finally {
      setDeletingId(null);
      setMenuOpenId(null);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-black text-white rounded-full text-[10px] md:text-xs font-bold">
            <CheckCircle className="w-2.5 h-2.5 md:w-3 md:h-3" />
            <span className="hidden sm:inline">Completed</span>
            <span className="sm:hidden">Done</span>
          </span>
        );
      case "processing":
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-neutral-200 text-neutral-700 rounded-full text-[10px] md:text-xs font-bold">
            <Loader2 className="w-2.5 h-2.5 md:w-3 md:h-3 animate-spin" /> Processing
          </span>
        );
      case "failed":
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-neutral-200 text-neutral-500 rounded-full text-[10px] md:text-xs font-bold">
            <XCircle className="w-2.5 h-2.5 md:w-3 md:h-3" /> Failed
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 px-2 py-1 bg-neutral-200 text-neutral-500 rounded-full text-[10px] md:text-xs font-bold">
            <Clock className="w-2.5 h-2.5 md:w-3 md:h-3" /> Pending
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-white text-neutral-900">
      {/* Header */}
      <header className="relative z-20 border-b border-neutral-200 sticky top-0 bg-white">
        <div className="max-w-5xl mx-auto px-4 py-3 md:py-4 flex items-center justify-between">
          <div className="flex items-center gap-3 md:gap-4">
            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-1.5 md:gap-2 text-neutral-500 hover:text-black transition p-1"
            >
              <ArrowLeft className="w-4 h-4 md:w-5 md:h-5" />
              <span className="text-xs md:text-sm font-medium hidden sm:inline">Home</span>
            </button>
            <div className="flex items-center gap-2 md:gap-3">
              <div className="w-8 h-8 md:w-10 md:h-10 rounded-xl bg-black flex items-center justify-center">
                <Clock className="w-4 h-4 md:w-5 md:h-5 text-white" />
              </div>
              <div>
                <h1 className="text-base md:text-xl font-bold">History</h1>
                <p className="text-[10px] md:text-xs text-neutral-500">{filteredAnalyses.length} analyses</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-1.5 px-2.5 md:px-3 py-2 border rounded-lg text-xs md:text-sm transition ${showFilters ? "bg-black text-white border-black" : "bg-neutral-100 border-neutral-200 hover:bg-neutral-200"
                }`}
            >
              <Filter className="w-3.5 h-3.5 md:w-4 md:h-4" />
              <span className="hidden sm:inline">Filter</span>
            </button>
            <button
              onClick={fetchHistory}
              disabled={loading}
              className="flex items-center gap-1.5 px-2.5 md:px-3 py-2 bg-neutral-100 border border-neutral-200 rounded-lg hover:bg-neutral-200 transition text-xs md:text-sm"
            >
              <RefreshCw className={`w-3.5 h-3.5 md:w-4 md:h-4 ${loading ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-1.5 px-2.5 md:px-3 py-2 bg-black text-white rounded-lg text-xs md:text-sm font-bold"
            >
              <Plus className="w-3.5 h-3.5 md:w-4 md:h-4" />
              <span className="hidden sm:inline">New</span>
            </button>
          </div>
        </div>

        {/* Filter/Sort Bar */}
        {showFilters && (
          <div className="border-t border-neutral-200 px-4 py-3 bg-neutral-50">
            <div className="max-w-5xl mx-auto flex flex-wrap gap-3 items-center">
              {/* Sort */}
              <div className="flex items-center gap-2">
                <ArrowUpDown className="w-4 h-4 text-neutral-500" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortOption)}
                  className="px-3 py-1.5 text-sm border border-neutral-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-black"
                >
                  <option value="newest">Newest First</option>
                  <option value="oldest">Oldest First</option>
                  <option value="stroke">By Stroke Type</option>
                </select>
              </div>

              {/* Filter by stroke */}
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-neutral-500" />
                <select
                  value={filterBy}
                  onChange={(e) => setFilterBy(e.target.value as FilterOption)}
                  className="px-3 py-1.5 text-sm border border-neutral-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-black"
                >
                  <option value="all">All Strokes</option>
                  <option value="serve">Serve</option>
                  <option value="dink">Dink</option>
                  <option value="groundstroke">Groundstroke</option>
                  <option value="overhead">Overhead</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-5xl mx-auto px-4 py-6 md:py-8">
        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-16 md:py-20">
            <Loader2 className="w-10 h-10 md:w-12 md:h-12 text-black animate-spin mb-4" />
            <p className="text-neutral-500 text-sm md:text-base">Loading your history...</p>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="text-center py-16 md:py-20">
            <div className="w-14 h-14 md:w-16 md:h-16 mx-auto mb-4 bg-neutral-100 rounded-full flex items-center justify-center">
              <XCircle className="w-7 h-7 md:w-8 md:h-8 text-neutral-500" />
            </div>
            <h2 className="text-lg md:text-xl font-bold mb-2">Couldn't Load History</h2>
            <p className="text-neutral-500 mb-6 text-sm md:text-base px-4">{error}</p>
            <button
              onClick={fetchHistory}
              className="px-5 md:px-6 py-2.5 md:py-3 bg-black text-white rounded-xl font-bold text-sm md:text-base"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Not Signed In State */}
        {!loading && !error && !isSignedIn && isLoaded && (
          <div className="text-center py-16 md:py-20">
            <div className="w-16 h-16 md:w-20 md:h-20 mx-auto mb-5 md:mb-6 bg-neutral-100 rounded-full flex items-center justify-center border border-neutral-200">
              <LogIn className="w-8 h-8 md:w-10 md:h-10 text-neutral-400" />
            </div>
            <h2 className="text-xl md:text-2xl font-bold mb-2">Sign In Required</h2>
            <p className="text-neutral-500 mb-6 text-sm md:text-base px-4">
              Please sign in to view your analysis history
            </p>
            <SignInButton mode="modal">
              <button className="px-5 md:px-6 py-2.5 md:py-3 bg-black text-white rounded-xl font-bold text-sm md:text-base cursor-pointer">
                Sign In
              </button>
            </SignInButton>
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && isSignedIn && filteredAnalyses.length === 0 && (
          <div className="py-10 md:py-16">
            <EmptyState
              title="Start Your Journey"
              description="Record your first session to build your training history and track your improvement."
              actionLabel="Record First Session"
              onAction={() => router.push('/strikesense/upload')}
              variant="history"
            />
          </div>
        )}

        {/* Analysis List */}
        {!loading && !error && filteredAnalyses.length > 0 && (
          <div className="space-y-3 md:space-y-4">
            {filteredAnalyses.map((job) => {
              const strokeInfo = getStrokeInfo(job.stroke_type);

              return (
                <div
                  key={job.id}
                  onClick={() => job.status === "completed" && job.result_json && handleViewAnalysis(job)}
                  className={`bg-neutral-50 border border-neutral-200 rounded-xl md:rounded-2xl p-4 md:p-5 transition-all relative
                    ${job.status === "completed" && job.result_json
                      ? "hover:bg-neutral-100 hover:border-neutral-300 cursor-pointer active:scale-[0.99]"
                      : ""
                    }`}
                >
                  <div className="flex items-start gap-3 md:gap-4">
                    {/* Icon */}
                    <div className="w-11 h-11 md:w-14 md:h-14 rounded-xl bg-black flex items-center justify-center text-xl md:text-2xl flex-shrink-0">
                      {strokeInfo.icon}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-1.5 md:mb-2">
                        <div className="min-w-0">
                          <h3 className="text-sm md:text-lg font-bold text-black truncate">
                            {strokeInfo.label} Analysis
                          </h3>
                          <div className="flex flex-wrap items-center gap-2 md:gap-3 text-[10px] md:text-xs text-neutral-500 mt-0.5 md:mt-1">
                            <span className="flex items-center gap-1">
                              <Calendar className="w-2.5 h-2.5 md:w-3 md:h-3" />
                              {formatDate(job.created_at)}
                            </span>
                            {job.processing_time_sec && (
                              <span className="flex items-center gap-1 hidden sm:flex">
                                <Timer className="w-2.5 h-2.5 md:w-3 md:h-3" />
                                {job.processing_time_sec.toFixed(1)}s
                              </span>
                            )}
                            {job.total_frames && (
                              <span className="flex items-center gap-1">
                                <BarChart3 className="w-2.5 h-2.5 md:w-3 md:h-3" />
                                {job.total_frames} frames
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {getStatusBadge(job.status)}

                          {/* More Menu */}
                          <div className="relative">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuOpenId(menuOpenId === job.id ? null : job.id);
                              }}
                              className="p-1.5 hover:bg-neutral-200 rounded-lg transition"
                            >
                              <MoreVertical className="w-4 h-4 text-neutral-500" />
                            </button>

                            {menuOpenId === job.id && (
                              <div className="absolute right-0 top-8 bg-white border border-neutral-200 rounded-lg shadow-lg py-1 min-w-[120px] z-30">
                                <button
                                  onClick={(e) => handleDelete(e, job.id)}
                                  disabled={deletingId === job.id}
                                  className="w-full px-3 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 disabled:opacity-50"
                                >
                                  {deletingId === job.id ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                  ) : (
                                    <Trash2 className="w-4 h-4" />
                                  )}
                                  Delete
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Action Button - Desktop only */}
                    {job.status === "completed" && job.result_json && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleViewAnalysis(job);
                        }}
                        className="hidden md:flex items-center gap-2 px-4 py-2 bg-black text-white rounded-xl font-bold text-sm"
                      >
                        <Play className="w-4 h-4" />
                        View
                      </button>
                    )}

                    {/* Mobile Arrow */}
                    {job.status === "completed" && job.result_json && (
                      <div className="md:hidden flex items-center">
                        <Play className="w-4 h-4 text-black" />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
