"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth, useUser } from "@clerk/nextjs";
import {
    ArrowLeft,
    User,
    Save,
    Loader2,
    CheckCircle,
    AlertCircle,
} from "lucide-react";
import { getUserProfile, upsertUserProfile, UserProfile } from "@/lib/supabase-db";

const SKILL_LEVELS = ["2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "5.0+"];

export default function ProfilePage() {
    const router = useRouter();
    const { userId, isLoaded } = useAuth();
    const { user } = useUser();

    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [name, setName] = useState("");
    const [skillLevel, setSkillLevel] = useState("");
    const [dominantHand, setDominantHand] = useState<"left" | "right" | "">("");
    const [injuryHistory, setInjuryHistory] = useState("");

    useEffect(() => {
        if (isLoaded && userId) {
            loadProfile();
        } else if (isLoaded && !userId) {
            setLoading(false);
        }
    }, [isLoaded, userId]);

    const loadProfile = async () => {
        if (!userId) return;
        setLoading(true);
        try {
            const profile = await getUserProfile(userId);
            if (profile) {
                setName(profile.name || user?.firstName || "");
                setSkillLevel(profile.skill_level || "");
                setDominantHand(profile.dominant_hand || "");
                setInjuryHistory(profile.injury_history || "");
            } else {
                setName(user?.firstName || "");
            }
        } catch (err) {
            console.error("Failed to load profile:", err);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!userId) return;
        setSaving(true);
        setError(null);
        setSaved(false);

        try {
            const result = await upsertUserProfile(userId, {
                name: name || null,
                skill_level: skillLevel || null,
                dominant_hand: dominantHand || null,
                injury_history: injuryHistory || null,
            });

            if (result) {
                setSaved(true);
                setTimeout(() => setSaved(false), 3000);
            } else {
                setError("Failed to save profile. Please try again.");
            }
        } catch (err) {
            setError("An error occurred. Please try again.");
        } finally {
            setSaving(false);
        }
    };

    if (!isLoaded || loading) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-black" />
            </div>
        );
    }

    if (!userId) {
        return (
            <div className="min-h-screen bg-white flex items-center justify-center p-4">
                <div className="text-center">
                    <User className="w-16 h-16 text-neutral-300 mx-auto mb-4" />
                    <h2 className="text-xl font-bold mb-2 text-black">Sign In Required</h2>
                    <p className="text-neutral-500 mb-4">Please sign in to view your profile.</p>
                    <button
                        onClick={() => router.push("/")}
                        className="px-6 py-3 bg-black text-white rounded-xl font-bold"
                    >
                        Go Home
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="border-b border-neutral-200 sticky top-0 bg-white z-20">
                <div className="max-w-2xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => router.push("/")}
                            className="p-2 hover:bg-neutral-100 rounded-lg transition"
                        >
                            <ArrowLeft className="w-5 h-5 text-black" />
                        </button>
                        <div className="flex items-center gap-2">
                            <div className="w-10 h-10 rounded-full bg-black flex items-center justify-center">
                                <User className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h1 className="text-lg font-bold text-black">My Profile</h1>
                                <p className="text-xs text-neutral-500">Personalize your experience</p>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-2xl mx-auto px-4 py-8">
                <div className="space-y-6">
                    {/* Name */}
                    <div>
                        <label className="block text-sm font-semibold text-neutral-700 mb-2">
                            Display Name
                        </label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Enter your name"
                            className="w-full px-4 py-3 border border-neutral-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent bg-white text-black placeholder:text-neutral-400"
                        />
                    </div>

                    {/* Skill Level */}
                    <div>
                        <label className="block text-sm font-semibold text-neutral-700 mb-2">
                            Skill Level (DUPR Rating)
                        </label>
                        <div className="grid grid-cols-4 sm:grid-cols-7 gap-2">
                            {SKILL_LEVELS.map((level) => (
                                <button
                                    key={level}
                                    onClick={() => setSkillLevel(level)}
                                    className={`py-2 px-3 rounded-lg text-sm font-semibold border transition ${skillLevel === level
                                        ? "bg-black text-white border-black"
                                        : "bg-white text-neutral-700 border-neutral-300 hover:border-neutral-400"
                                        }`}
                                >
                                    {level}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Dominant Hand */}
                    <div>
                        <label className="block text-sm font-semibold text-neutral-700 mb-2">
                            Dominant Hand
                        </label>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setDominantHand("right")}
                                className={`flex-1 py-3 px-4 rounded-xl text-sm font-semibold border transition ${dominantHand === "right"
                                    ? "bg-black text-white border-black"
                                    : "bg-white text-neutral-700 border-neutral-300 hover:border-neutral-400"
                                    }`}
                            >
                                🤚 Right-Handed
                            </button>
                            <button
                                onClick={() => setDominantHand("left")}
                                className={`flex-1 py-3 px-4 rounded-xl text-sm font-semibold border transition ${dominantHand === "left"
                                    ? "bg-black text-white border-black"
                                    : "bg-white text-neutral-700 border-neutral-300 hover:border-neutral-400"
                                    }`}
                            >
                                ✋ Left-Handed
                            </button>
                        </div>
                    </div>

                    {/* Injury History */}
                    <div>
                        <label className="block text-sm font-semibold text-neutral-700 mb-2">
                            Injury History (Optional)
                        </label>
                        <textarea
                            value={injuryHistory}
                            onChange={(e) => setInjuryHistory(e.target.value)}
                            placeholder="E.g., Tennis elbow, shoulder issues, knee problems..."
                            rows={3}
                            className="w-full px-4 py-3 border border-neutral-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent resize-none bg-white text-black placeholder:text-neutral-400"
                        />
                        <p className="text-xs text-neutral-500 mt-1">
                            This helps us tailor recommendations to avoid aggravating existing conditions.
                        </p>
                    </div>

                    {/* Error Message */}
                    {error && (
                        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm">
                            <AlertCircle className="w-4 h-4" />
                            {error}
                        </div>
                    )}

                    {/* Save Button */}
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="w-full py-4 bg-black text-white rounded-xl font-bold text-sm flex items-center justify-center gap-2 hover:bg-neutral-800 transition disabled:opacity-50"
                    >
                        {saving ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Saving...
                            </>
                        ) : saved ? (
                            <>
                                <CheckCircle className="w-4 h-4" />
                                Saved!
                            </>
                        ) : (
                            <>
                                <Save className="w-4 h-4" />
                                Save Profile
                            </>
                        )}
                    </button>
                </div>
            </main>
        </div>
    );
}
