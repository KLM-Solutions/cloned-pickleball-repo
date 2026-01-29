"use client";

import React, { useState, useEffect, useRef } from 'react';
import { X, Sparkles, Bot, BrainCircuit, ChevronRight, Loader2 } from 'lucide-react';

interface AICoachModalProps {
    isOpen: boolean;
    onClose: () => void;
    analyticsData: any;
}

interface AIResponse {
    summary: string;
    takeaways: string[];
    focus_drill: string;
}

export default function AICoachModal({ isOpen, onClose, analyticsData }: AICoachModalProps) {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AIResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const hasFetched = useRef(false);

    useEffect(() => {
        if (isOpen && !hasFetched.current && !result) {
            fetchInsights();
        }
    }, [isOpen]);

    const fetchInsights = async () => {
        try {
            setLoading(true);
            setError(null);

            const response = await fetch('/api/analytics/ai-summary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(analyticsData)
            });

            if (!response.ok) throw new Error("Failed to contact AI Coach");

            const data = await response.json();
            setResult(data);
            hasFetched.current = true;
        } catch (err) {
            setError("Coach isn't available right now. Please check back in a moment.");
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

            <div className="relative w-full max-w-lg max-h-[85vh] flex flex-col bg-[#0F0F12] border border-white/10 rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="p-4 sm:p-6 pb-4 border-b border-white/5 flex items-center justify-between shrink-0">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                            <Bot className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                        </div>
                        <div>
                            <h3 className="text-lg sm:text-xl font-bold text-white">My Coach's Feedback</h3>
                            <p className="text-[10px] sm:text-xs text-indigo-300">Your personalized analysis</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full transition text-white/60 hover:text-white">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4 sm:p-6 overflow-y-auto">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-10 space-y-4">
                            <div className="relative">
                                <div className="w-12 h-12 sm:w-16 sm:h-16 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <BrainCircuit className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-400 opacity-50" />
                                </div>
                            </div>
                            <p className="text-sm sm:text-base text-indigo-300 animate-pulse font-medium">Analyzing your form...</p>
                        </div>
                    ) : error ? (
                        <div className="text-center py-8">
                            <p className="text-red-400 mb-4 text-sm">{error}</p>
                            <button onClick={fetchInsights} className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-sm text-white transition">
                                <span className="flex items-center gap-2">Let's Try Again</span>
                            </button>
                        </div>
                    ) : result ? (
                        <div className="space-y-5 sm:space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                            {/* Summary */}
                            <div className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-xl p-3 sm:p-4">
                                <p className="text-sm sm:text-base text-indigo-100 leading-relaxed italic">"{result.summary}"</p>
                            </div>

                            {/* Takeaways */}
                            <div className="space-y-2 sm:space-y-3">
                                <h4 className="text-xs sm:text-sm font-semibold text-white/80 uppercase tracking-wider flex items-center gap-2">
                                    <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-400" /> Key Takeaways
                                </h4>
                                <ul className="space-y-2">
                                    {result.takeaways.map((point, i) => (
                                        <li key={i} className="flex gap-2.5 sm:gap-3 text-sm text-neutral-300">
                                            <span className="shrink-0 w-1.5 h-1.5 mt-1.5 rounded-full bg-indigo-500" />
                                            {point}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Focus Drill */}
                            <div className="bg-[#1A1A1E] border-l-4 border-emerald-500 rounded-r-xl p-3 sm:p-4">
                                <h4 className="text-emerald-400 font-bold text-xs sm:text-sm mb-0.5 uppercase tracking-wide">
                                    Next Focus Drill
                                </h4>
                                <p className="text-sm sm:text-base text-white font-medium">{result.focus_drill}</p>
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
}
