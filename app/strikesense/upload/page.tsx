"use client";

import React, { useState, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Upload, ArrowLeft, Cloud, Zap, Shield, Film, CheckCircle } from "lucide-react";
import { useCelebration } from "@/app/hooks/useCelebration";

import { uploadVideo } from "@/lib/supabase";
import CameraAngleValidator from "@/app/components/CameraAngleValidator";
import { ValidationResult, StrokeType as CameraStrokeType } from "@/lib/analysis/cameraValidation";

export const dynamic = 'force-dynamic';

function UploadContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const celebration = useCelebration();
    const strokeType = searchParams.get('stroke') || 'serve';

    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string | null>(null);

    // Camera angle validation state
    const [showValidation, setShowValidation] = useState(false);
    const [pendingFile, setPendingFile] = useState<File | null>(null);
    const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);

    // Triggered when user selects a file - shows validation first
    const handleFileSelect = async (file: File) => {
        if (!file || !file.type.startsWith('video/')) {
            setError('Please pick a video file so we can analyze your form!');
            return;
        }

        const maxSize = 500 * 1024 * 1024;
        if (file.size > maxSize) {
            setError('That video is a bit too heavy (over 500MB). Let\'s try a shorter or smaller one.');
            return;
        }

        // Store file and show validation modal
        setError(null);
        setPendingFile(file);
        setFileName(file.name);
        setShowValidation(true);
    };

    // Called when validation completes
    const handleValidationComplete = (result: ValidationResult, proceed: boolean) => {
        setValidationResult(result);
        setShowValidation(false);

        if (proceed && pendingFile) {
            // Store validation result in session for later reference
            sessionStorage.setItem('cameraValidation', JSON.stringify({
                score: result.score,
                passed: result.passed,
                issues: result.issues.map(i => i.message),
            }));

            proceedWithUpload(pendingFile);
        } else {
            // User chose to re-record
            setPendingFile(null);
            setFileName(null);
        }
    };

    // Cancel validation and go back
    const handleValidationCancel = () => {
        setShowValidation(false);
        setPendingFile(null);
        setFileName(null);
    };

    // Proceed with actual upload after validation
    const proceedWithUpload = async (file: File) => {
        try {
            setIsUploading(true);
            setUploadProgress(10);

            // Simulate variable upload speed
            const totalSize = file.size;
            let loaded = totalSize * 0.1;
            const startTime = Date.now();

            const progressInterval = setInterval(() => {
                const now = Date.now();
                const elapsed = (now - startTime) / 1000; // seconds

                // Add some randomness to speed (simulating network fluctuation)
                const chunk = (totalSize / 15) * (0.5 + Math.random());
                loaded = Math.min(loaded + chunk, totalSize * 0.95);

                const progress = (loaded / totalSize) * 100;
                setUploadProgress(Math.round(progress));

                // Keep progress below 99% until actually done
                if (progress >= 95) {
                    clearInterval(progressInterval);
                }
            }, 800);

            const videoUrl = await uploadVideo(file);

            clearInterval(progressInterval);
            setUploadProgress(100);

            // Celebration!
            if (celebration) celebration.triggerBurst();

            sessionStorage.setItem('videoUrl', videoUrl);
            sessionStorage.setItem('videoFileName', file.name);
            sessionStorage.setItem('strokeType', strokeType);

            setTimeout(() => {
                router.push(`/strikesense/crop?stroke=${strokeType}`);
            }, 1000); // Slightly longer delay to enjoy the burst

        } catch (err: any) {
            console.error('Upload failed:', err);
            setError(err.message || 'We hit a snag uploading your video. Want to give it another shot?');
            setIsUploading(false);
            setUploadProgress(0);
            setPendingFile(null);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    };

    return (
        <div className="min-h-screen bg-white text-neutral-900">
            {/* Camera Angle Validation Modal */}
            {showValidation && pendingFile && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
                    <div className="w-full max-w-sm animate-in fade-in zoom-in-95 duration-200">
                        <CameraAngleValidator
                            videoFile={pendingFile}
                            strokeType={(strokeType as CameraStrokeType) || 'groundstroke'}
                            onValidationComplete={handleValidationComplete}
                            onCancel={handleValidationCancel}
                        />
                    </div>
                </div>
            )}

            {/* Header */}
            <header className="relative z-10 px-4 py-3 md:py-4">
                <div className="max-w-lg mx-auto flex items-center justify-between">
                    <button
                        onClick={() => router.push(`/strikesense/guide?stroke=${strokeType}`)}
                        className="flex items-center gap-1.5 md:gap-2 text-neutral-500 hover:text-black transition p-1"
                    >
                        <ArrowLeft className="w-4 h-4 md:w-5 md:h-5" />
                        <span className="text-xs md:text-sm font-medium">Back</span>
                    </button>
                    <div className="text-[10px] md:text-xs font-bold tracking-widest uppercase text-neutral-400">
                        Step 2 of 3
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 px-4 pb-6 md:pb-8">
                <div className="max-w-lg mx-auto">

                    {/* Hero */}
                    <div className="text-center pt-2 md:pt-4 pb-6 md:pb-8">
                        <div className="inline-flex w-16 h-16 md:w-20 md:h-20 rounded-xl md:rounded-2xl bg-black items-center justify-center text-3xl md:text-4xl mb-4 md:mb-5">
                            🎾
                        </div>
                        <h1 className="text-xl md:text-2xl font-bold mb-1.5 md:mb-2">Let's See Your {strokeType.charAt(0).toUpperCase() + strokeType.slice(1)}</h1>
                        <p className="text-neutral-500 text-sm md:text-base">Upload a video to start your analysis</p>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="mb-4 md:mb-6 p-3 md:p-4 bg-neutral-100 border border-neutral-200 text-neutral-700 rounded-xl md:rounded-2xl text-xs md:text-sm flex items-start gap-2 md:gap-3">
                            <div className="w-4 h-4 md:w-5 md:h-5 rounded-full bg-black flex items-center justify-center flex-shrink-0 mt-0.5">
                                <span className="text-[10px] md:text-xs text-white">!</span>
                            </div>
                            {error}
                        </div>
                    )}

                    {/* Upload Zone */}
                    <div
                        onDrop={handleDrop}
                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onClick={() => !isUploading && fileInputRef.current?.click()}
                        className={`
                            relative rounded-2xl md:rounded-3xl p-6 md:p-8 text-center cursor-pointer transition-all duration-300
                            ${isDragging
                                ? 'bg-black scale-[1.02]'
                                : 'bg-neutral-50 hover:bg-neutral-100 border-2 border-dashed border-neutral-300 hover:border-neutral-400'
                            }
                            ${isUploading ? 'pointer-events-none' : ''}
                            active:scale-[0.98] touch-manipulation
                        `}
                    >
                        {isUploading ? (
                            <div className="py-4 md:py-6">
                                {/* Progress Ring */}
                                <div className="relative w-24 h-24 sm:w-28 sm:h-28 md:w-32 md:h-32 mx-auto mb-5 md:mb-6">
                                    <svg
                                        className="w-full h-full transform -rotate-90"
                                        viewBox="0 0 100 100"
                                    >
                                        <circle
                                            cx="50"
                                            cy="50"
                                            r="42"
                                            stroke="currentColor"
                                            strokeWidth="8"
                                            fill="none"
                                            className="text-neutral-200"
                                        />
                                        <circle
                                            cx="50"
                                            cy="50"
                                            r="42"
                                            stroke="black"
                                            strokeWidth="8"
                                            fill="none"
                                            strokeLinecap="round"
                                            strokeDasharray={264}
                                            strokeDashoffset={264 - (264 * uploadProgress) / 100}
                                            className="transition-all duration-500 ease-out"
                                        />
                                    </svg>
                                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                                        <span className="text-2xl sm:text-3xl md:text-4xl font-black text-black">
                                            {uploadProgress}%
                                        </span>
                                        <span className="text-[10px] sm:text-xs text-neutral-500 font-medium -mt-1 animate-pulse">
                                            sending...
                                        </span>
                                    </div>
                                </div>

                                <h3 className="text-lg sm:text-xl md:text-2xl font-bold mb-2">Uploading your serve...</h3>
                                <p className="text-neutral-500 text-xs sm:text-sm mb-2 truncate max-w-[220px] sm:max-w-[280px] mx-auto px-2">{fileName}</p>
                                <p className="text-neutral-400 text-[10px] sm:text-xs">
                                    {uploadProgress < 50 ? 'Initializing secure transfer...' : 'Almost there...'}
                                </p>
                            </div>
                        ) : (
                            <>
                                <div className={`
                                    w-16 h-16 md:w-20 md:h-20 mx-auto mb-4 md:mb-6 rounded-xl md:rounded-2xl flex items-center justify-center
                                    ${isDragging ? 'bg-white' : 'bg-black'}
                                    transition-all duration-300
                                `}>
                                    <Upload className={`w-8 h-8 md:w-10 md:h-10 ${isDragging ? 'text-black' : 'text-white'}`} />
                                </div>

                                <h3 className={`text-lg md:text-xl font-bold mb-1.5 md:mb-2 ${isDragging ? 'text-white' : 'text-black'}`}>
                                    {isDragging ? 'Drop it here!' : 'Tap to Upload Video'}
                                </h3>
                                <p className={`mb-4 md:mb-6 text-sm ${isDragging ? 'text-white/70' : 'text-neutral-500'}`}>
                                    {isDragging ? 'Release to upload' : 'or drag & drop a file'}
                                </p>

                                <div className="flex gap-1.5 md:gap-2 justify-center flex-wrap">
                                    {['MP4', 'MOV', 'AVI', 'WEBM'].map((format) => (
                                        <span key={format} className={`px-2.5 md:px-3 py-1 rounded-full text-[10px] md:text-xs font-medium ${isDragging ? 'bg-white text-black' : 'bg-neutral-200 text-neutral-600'}`}>
                                            {format}
                                        </span>
                                    ))}
                                </div>
                            </>
                        )}

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="video/*"
                            onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) handleFileSelect(file);
                            }}
                            className="hidden"
                            disabled={isUploading}
                        />
                    </div>

                    {/* Features */}
                    {!isUploading && (
                        <div className="grid grid-cols-3 gap-2 md:gap-3 mt-4 md:mt-6">
                            {[
                                { icon: Cloud, label: 'Cloud Upload', desc: 'Fast & secure' },
                                { icon: Zap, label: 'Fast Analysis', desc: 'Pro-level' },
                                { icon: Shield, label: 'Private', desc: 'Your data safe' },
                            ].map((feature, i) => (
                                <div key={i} className="bg-neutral-50 border border-neutral-200 rounded-xl md:rounded-2xl p-3 md:p-4 text-center">
                                    <feature.icon className="w-5 h-5 md:w-6 md:h-6 mx-auto mb-1.5 md:mb-2 text-black" />
                                    <div className="text-[10px] md:text-xs font-bold text-black mb-0.5">{feature.label}</div>
                                    <div className="text-[9px] md:text-[10px] text-neutral-400">{feature.desc}</div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* File Info */}
                    {!isUploading && (
                        <div className="mt-4 md:mt-6 flex items-center justify-center gap-2 md:gap-4 text-[10px] md:text-xs text-neutral-400 flex-wrap">
                            <div className="flex items-center gap-1">
                                <Film className="w-3.5 h-3.5 md:w-4 md:h-4" />
                                <span>Max 500MB</span>
                            </div>
                            <span className="hidden sm:inline">•</span>
                            <div className="flex items-center gap-1">
                                <span>Best: 30 FPS</span>
                            </div>
                            <span className="hidden sm:inline">•</span>
                            <div className="flex items-center gap-1">
                                <span>3-60 seconds</span>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

export default function UploadPage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen bg-white flex items-center justify-center">
                <div className="text-black font-bold animate-pulse">Loading...</div>
            </div>
        }>
            <UploadContent />
        </Suspense>
    );
}
