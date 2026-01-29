"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronRight, Zap, Target, Shield } from "lucide-react";
import { useAuth, SignInButton } from "@clerk/nextjs";

const STROKE_OPTIONS = [
  {
    id: "serve",
    title: "Serve",
    description: "Analyze your underhand serve technique, contact point, and power generation",
    icon: "🎾",
    stats: "Contact • Power • Form"
  },
  {
    id: "dink",
    title: "Dink",
    description: "Perfect your soft game with knee bend analysis and paddle control",
    icon: "🤏",
    stats: "Control • Posture • Touch"
  },
  {
    id: "groundstroke",
    title: "Drive",
    description: "Analyze forehand and backhand drives for maximum power and accuracy",
    icon: "💪",
    stats: "Speed • Rotation • Follow-through"
  },
  {
    id: "overhead",
    title: "Overhead",
    description: "Perfect your smash with extension analysis and timing feedback",
    icon: "⚡",
    stats: "Extension • Timing • Power"
  },
] as const;

export default function Home() {
  const router = useRouter();
  const { isSignedIn, isLoaded } = useAuth();
  const [hoveredStroke, setHoveredStroke] = useState<string | null>(null);
  const [showSignInPrompt, setShowSignInPrompt] = useState(false);
  const [selectedStrokeId, setSelectedStrokeId] = useState<string | null>(null);

  const handleStrokeSelect = (strokeId: string) => {
    // Check if user is signed in
    if (!isSignedIn && isLoaded) {
      setSelectedStrokeId(strokeId);
      setShowSignInPrompt(true);
      return;
    }
    router.push(`/strikesense/guide?stroke=${strokeId}`);
  };

  return (
    <div className="min-h-screen bg-white text-neutral-900 overflow-x-hidden">
      {/* Main Content */}
      <main className="relative z-10 max-w-6xl mx-auto px-4 md:px-6 py-8 md:py-12">


        {/* Hero Section */}
        <div className="text-center mb-10 md:mb-16">
          <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-4 md:mb-6 leading-tight">
            Analyze Your
            <span className="block text-black">
              Pickleball Stroke
            </span>
          </h2>

          <p className="text-sm md:text-lg text-neutral-500 max-w-2xl mx-auto mb-6 md:mb-8 px-4">
            Upload a video and get instant, pro-level feedback on your technique.
            We'll track your joint angles, spot what's holding you back, and help you level up your game.
          </p>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-3 sm:flex sm:items-center sm:justify-center sm:gap-6 md:gap-8 text-xs md:text-sm max-w-md sm:max-w-none mx-auto">
            <div className="flex flex-col sm:flex-row items-center gap-0.5 sm:gap-2">
              <span className="text-lg sm:text-xl md:text-2xl font-bold text-black">33</span>
              <span className="text-neutral-500 text-[10px] sm:text-xs md:text-sm">Body Points Tracked</span>
            </div>
            <div className="hidden sm:block w-px h-6 md:h-8 bg-neutral-200 flex-shrink-0" />
            <div className="flex flex-col sm:flex-row items-center gap-0.5 sm:gap-2">
              <span className="text-lg sm:text-xl md:text-2xl font-bold text-black">30</span>
              <span className="text-neutral-500 text-[10px] sm:text-xs md:text-sm">FPS Analysis</span>
            </div>
            <div className="hidden sm:block w-px h-6 md:h-8 bg-neutral-200 flex-shrink-0" />
            <div className="flex flex-col sm:flex-row items-center gap-0.5 sm:gap-2">
              <span className="text-lg sm:text-xl md:text-2xl font-bold text-black">Real-time</span>
              <span className="text-neutral-500 text-[10px] sm:text-xs md:text-sm">Coaching Feedback</span>
            </div>
          </div>
        </div>

        {/* Stroke Selection */}
        <div className="mb-8">
          {/* Action Links */}
          <div className="text-center mb-4 md:mb-6 flex items-center justify-center gap-2 flex-wrap">
            <button
              onClick={() => router.push('/strikesense/player?stroke=serve&demo=true')}
              className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 rounded-full text-sm font-medium text-white transition-all"
            >
              <span>✨</span>
              <span>Try Demo</span>
            </button>
            {isSignedIn && (
              <button
                onClick={() => router.push('/strikesense/analytics')}
                className="inline-flex items-center gap-2 px-4 py-2 bg-black hover:bg-neutral-800 rounded-full text-sm font-medium text-white transition-all"
              >
                <span>📊</span>
                <span>See My Stats</span>
              </button>
            )}
          </div>

          <h3 className="text-center text-[10px] md:text-sm font-medium text-neutral-400 uppercase tracking-wider mb-6 md:mb-8">
            What are we working on today?
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
            {STROKE_OPTIONS.map((stroke) => (
              <button
                key={stroke.id}
                onClick={() => handleStrokeSelect(stroke.id)}
                onMouseEnter={() => setHoveredStroke(stroke.id)}
                onMouseLeave={() => setHoveredStroke(null)}
                className={`
                  relative group p-4 md:p-6 rounded-2xl border border-neutral-200 
                  bg-neutral-50
                  hover:border-neutral-300 hover:bg-neutral-100
                  transition-all duration-300 ease-out
                  active:scale-[0.98] touch-manipulation
                  ${hoveredStroke === stroke.id ? 'scale-[1.02]' : ''}
                `}
              >
                {/* Icon */}
                <div className="w-12 h-12 md:w-14 md:h-14 rounded-xl mb-3 md:mb-4 flex items-center justify-center text-2xl md:text-3xl bg-black group-hover:scale-110 transition-transform duration-300">
                  {stroke.icon}
                </div>

                {/* Content */}
                <h4 className="text-base md:text-xl font-bold mb-1 md:mb-2 text-black text-left">
                  {stroke.title}
                </h4>
                <p className="text-xs md:text-sm text-neutral-500 mb-2 md:mb-4 line-clamp-2 text-left hidden sm:block">
                  {stroke.description}
                </p>

                {/* Stats tag */}
                <div className="text-[10px] md:text-xs text-neutral-400 font-medium text-left">
                  {stroke.stats}
                </div>

                {/* Arrow indicator */}
                <div className="absolute top-4 md:top-6 right-4 md:right-6 w-7 h-7 md:w-8 md:h-8 rounded-full bg-neutral-200 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300">
                  <ChevronRight className="w-3.5 h-3.5 md:w-4 md:h-4 text-black" />
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* How it works */}
        <div className="mt-12 md:mt-20 text-center">
          <h3 className="text-[10px] md:text-sm font-medium text-neutral-400 uppercase tracking-wider mb-6 md:mb-8">
            Your Path to Better Pickleball
          </h3>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-8 md:gap-16">
            {[
              { step: "1", title: "Upload Video", desc: "Film your shot from any angle", icon: Target },
              { step: "2", title: "Select Player", desc: "Highlight yourself on screen", icon: Zap },
              { step: "3", title: "Get Coaching", desc: "See exactly how to fix your form", icon: Shield },
            ].map((item, i) => (
              <div key={i} className="flex flex-col items-center">
                <div className="w-12 h-12 md:w-14 md:h-14 rounded-full bg-black flex items-center justify-center text-base md:text-lg font-bold mb-2 md:mb-3 text-white">
                  {item.step}
                </div>
                <h4 className="font-semibold text-black mb-0.5 md:mb-1 text-sm md:text-base">{item.title}</h4>
                <p className="text-xs md:text-sm text-neutral-500">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Mobile CTA */}
        <div className="mt-10 md:hidden">
          <button
            onClick={() => handleStrokeSelect('serve')}
            className="w-full py-4 bg-black text-white rounded-2xl font-bold text-base active:scale-[0.98] transition-transform"
          >
            Analyzye My Serve →
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 text-center py-6 md:py-8 border-t border-neutral-200 mt-8">
        <p className="text-xs md:text-sm text-neutral-400 px-4">
          Powered by StrikeSense
        </p>
      </footer>

      {/* Sign In Prompt Modal */}
      {showSignInPrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl p-6 md:p-8 max-w-sm mx-4 shadow-2xl">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-neutral-100 rounded-full flex items-center justify-center">
                <span className="text-3xl">🔐</span>
              </div>
              <h3 className="text-xl font-bold mb-2">Sign In Required</h3>
              <p className="text-neutral-500 text-sm mb-6">
                Please sign in to analyze your pickleball strokes and save your progress.
              </p>
              <div className="flex flex-col gap-3">
                <SignInButton mode="modal">
                  <button
                    onClick={() => setShowSignInPrompt(false)}
                    className="w-full py-3 bg-black text-white rounded-xl font-bold text-sm cursor-pointer hover:bg-neutral-800 transition-colors"
                  >
                    Sign In
                  </button>
                </SignInButton>
                <button
                  onClick={() => {
                    setShowSignInPrompt(false);
                    router.push('/demo');
                  }}
                  className="w-full py-3 bg-neutral-100 text-neutral-700 rounded-xl font-semibold text-sm hover:bg-neutral-200 transition-colors"
                >
                  🎬 Try Demo Instead
                </button>
                <button
                  onClick={() => setShowSignInPrompt(false)}
                  className="w-full py-2 text-neutral-500 text-sm font-medium hover:text-black transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
