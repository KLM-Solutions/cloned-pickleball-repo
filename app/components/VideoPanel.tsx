"use client";

import { Upload, Loader2, Tag, Zap, Play, Pause, SkipBack, SkipForward, CheckCircle, Settings, Volume2, VolumeX, Maximize, AlertTriangle } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { AnalyzeResponse } from "../types";

// Extract a frame from video file for tagging UI
async function extractFrameFromVideo(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        const url = URL.createObjectURL(file);
        video.src = url;
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';

        video.onloadedmetadata = () => {
            // Seek to frame ~0.1s or middle if very short
            video.currentTime = Math.min(0.1, video.duration / 2);
        };

        video.onseeked = () => {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const dataUrl = canvas.toDataURL('image/png');
                    URL.revokeObjectURL(url);
                    resolve(dataUrl);
                } else {
                    reject(new Error('Failed to get canvas context'));
                }
            } catch (e) {
                reject(e);
            }
        };

        video.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error('Failed to load video'));
        };
    });
}

interface VideoPanelProps {
    videoFile: File | null;
    videoUrl: string | null;
    onVideoUpload: (file: File) => void;
    onVideoClick?: (normalizedX: number, normalizedY: number) => void;
    players?: Array<{ id: number; color: string; normalizedCoords: { x: number; y: number } }>;
    isTaggingMode?: boolean;
    onStartAnalysis?: () => void;
    isProcessing?: boolean;
    analysisData?: AnalyzeResponse | null;
    // Advanced Props
    currentTime?: number;
    onTimeUpdate?: (time: number) => void;
    showOverlay?: boolean;
    playbackSpeed?: number;
    onToggleOverlay?: () => void;
    onSpeedChange?: (speed: number) => void;
    sideBySide?: boolean;
}

type UploadState = 'idle' | 'uploading' | 'uploaded';

export default function VideoPanel({
    videoFile,
    videoUrl,
    onVideoUpload,
    onVideoClick,
    players = [],
    isTaggingMode = false,
    onStartAnalysis,
    isProcessing = false,
    // Advanced Props
    currentTime,
    onTimeUpdate,
    showOverlay = true,
    playbackSpeed = 1,
    onToggleOverlay,
    onSpeedChange,
    analysisData,
    sideBySide = false
}: VideoPanelProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const secondaryVideoRef = useRef<HTMLVideoElement>(null);

    // State
    const [uploadState, setUploadState] = useState<UploadState>('idle');
    const [firstFrameUrl, setFirstFrameUrl] = useState<string | null>(null);
    const [isExtractingFrame, setIsExtractingFrame] = useState(false);
    const [isTagMode, setIsTagMode] = useState(false);
    const [taggedPlayer, setTaggedPlayer] = useState<{ x: number; y: number } | null>(null);
    const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);

    // Player State
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const [isMuted, setIsMuted] = useState(true);
    const [showControls, setShowControls] = useState(true);

    // Sync videoFile prop to local state
    useEffect(() => {
        if (videoFile) {
            setLocalVideoUrl(URL.createObjectURL(videoFile));
        }
    }, [videoFile]);

    // Sync prop state to local state
    useEffect(() => {
        if (isTaggingMode !== undefined) {
            setIsTagMode(isTaggingMode);
        }
    }, [isTaggingMode]);

    // Handle File Select
    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        setLocalVideoUrl(url);
        setUploadState('uploading');
        setTimeout(() => {
            onVideoUpload(file);
        }, 2000);
    };

    // Generate first frame for tagging
    useEffect(() => {
        // If we already have analysis data (e.g. demo or finished analysis), skip frame extraction
        // We don't need to tag players if the analysis is done.
        if (analysisData) {
            setUploadState('uploaded');
            return;
        }

        if (videoFile) {
            setIsExtractingFrame(true);
            setUploadState('uploading');
            extractFrameFromVideo(videoFile).then(url => {
                setFirstFrameUrl(url);
                setUploadState('uploaded');

                // Auto-detection logic for initial player tag on single player videos
                // This is a simple heuristic: if it's 1v1 we might assume player is closest... 
                // but actually for now let's just default to requiring a click to be safe, 
                // OR trigger the auto-detect only if we want that feature active.
                // Keeping it manual for now as per "click the player" instruction.

            }).catch(err => {
                console.error("Frame extraction error:", err);
                setUploadState('uploaded'); // Still proceed to allow play even if thumb fails
            }).finally(() => {
                setIsExtractingFrame(false);
            });
        } else if (videoUrl) {
            // Fallback for URL-only (though unlikely in upload flow)
            const video = document.createElement('video');
            video.src = videoUrl;
            video.crossOrigin = "anonymous"; // Enable CORS for frame extraction
            video.muted = true; // Mute to prevent sound issues
            video.playsInline = true; // Play inline
            video.preload = 'auto'; // Preload for faster seeking
            video.onloadedmetadata = () => { // Use onloadedmetadata to ensure duration/dimensions are available
                video.currentTime = Math.min(0.1, video.duration / 2); // Frame 3 approx, or middle if very short
            };
            video.onseeked = () => {
                try {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
                    setFirstFrameUrl(canvas.toDataURL('image/png'));
                    setUploadState('uploaded'); // Set uploaded state here for URL-only case
                } catch (e) {
                    console.warn("Could not extract frame from video URL (CORS restricted?):", e);
                    // Still set state to uploaded so the UI doesn't hang, just without a thumbnail
                    setUploadState('uploaded');
                }
            };
            video.onerror = (e) => {
                // Do not reset state to idle, just warn. This fixes the "Console Error" during demo load.
                console.warn("Error loading video for frame extraction (likely CORS or format):", e);
                setUploadState('uploaded');
            };
        }
    }, [videoFile, videoUrl, analysisData]); // Added analysisData to dependencies

    // The original extractFirstFrame function is no longer needed as its logic is now in the useEffect.
    // Keeping it commented out for reference if needed, but it's effectively replaced.
    /*
    const extractFirstFrame = (url: string) => {
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        video.src = url;
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';

        const captureFrame = () => {
            if (ctx && video.videoWidth > 0 && video.videoHeight > 0) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                setFirstFrameUrl(canvas.toDataURL('image/png'));
            }
            setUploadState('uploaded');
        };

        video.onloadeddata = () => {
            video.currentTime = Math.min(0.5, video.duration / 2);
        };
        video.onseeked = () => {
            requestAnimationFrame(() => requestAnimationFrame(captureFrame));
        };
    };
    */

    // --- Tagging Logic (Drag to Select) ---
    const [selectionBox, setSelectionBox] = useState<{ x: number, y: number, w: number, h: number } | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState<{ x: number, y: number } | null>(null);

    const getNormalizedCoords = (e: React.MouseEvent<HTMLElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / rect.width,
            y: (e.clientY - rect.top) / rect.height
        };
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLElement>) => {
        if (!isTagMode) return;
        setIsDragging(true);
        const coords = getNormalizedCoords(e);
        setDragStart(coords);
        setSelectionBox({ x: coords.x, y: coords.y, w: 0, h: 0 });
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLElement>) => {
        if (!isDragging || !dragStart) return;
        const coords = getNormalizedCoords(e);
        const w = coords.x - dragStart.x;
        const h = coords.y - dragStart.y;
        setSelectionBox({
            x: w > 0 ? dragStart.x : coords.x,
            y: h > 0 ? dragStart.y : coords.y,
            w: Math.abs(w),
            h: Math.abs(h)
        });
    };

    const handleMouseUp = () => {
        if (!isDragging || !selectionBox) return;
        setIsDragging(false);
        setDragStart(null);
        const centerX = selectionBox.x + (selectionBox.w / 2);
        const centerY = selectionBox.y + (selectionBox.h / 2);

        if (selectionBox.w > 0.01 && selectionBox.h > 0.01) {
            setTaggedPlayer({ x: centerX, y: centerY });
            setIsTagMode(false);
            if (onVideoClick) onVideoClick(centerX, centerY);
        } else {
            setTaggedPlayer({ x: selectionBox.x, y: selectionBox.y });
            setIsTagMode(false);
            if (onVideoClick) onVideoClick(selectionBox.x, selectionBox.y);
        }
        setSelectionBox(null);
    };

    // --- Player Logic ---
    const togglePlay = async () => {
        if (!videoRef.current) return;
        try {
            if (videoRef.current.paused) {
                await videoRef.current.play();
            } else {
                videoRef.current.pause();
            }
        } catch (error) {
            // Ignore AbortError which happens if user toggles too fast
            console.log("Playback toggled too fast");
        }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!videoRef.current) return;
        const time = parseFloat(e.target.value);
        videoRef.current.currentTime = time;
        // Immediate sync for secondary video
        if (secondaryVideoRef.current) {
            secondaryVideoRef.current.currentTime = time;
        }
        if (onTimeUpdate) onTimeUpdate(time);
    };

    const stepFrame = (direction: number) => {
        if (!videoRef.current) return;
        const frameTime = 1 / 30; // Assuming 30fps
        const newTime = videoRef.current.currentTime + (direction * frameTime);
        videoRef.current.currentTime = Math.max(0, Math.min(newTime, duration));
        if (secondaryVideoRef.current) secondaryVideoRef.current.currentTime = videoRef.current.currentTime;
    };

    const formatTime = (time: number) => {
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        const ms = Math.floor((time % 1) * 100);
        return `${minutes}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
    };

    // Sync Props
    useEffect(() => {
        if (videoRef.current) videoRef.current.playbackRate = playbackSpeed;
    }, [playbackSpeed]);

    // Sync Props: CurrentTime
    useEffect(() => {
        if (videoRef.current && currentTime !== undefined && Math.abs(videoRef.current.currentTime - currentTime) > 0.1) {
            videoRef.current.currentTime = currentTime;
            // Also sync secondary video if it exists
            if (secondaryVideoRef.current) {
                secondaryVideoRef.current.currentTime = currentTime;
            }
        }
    }, [currentTime]);

    const activeSrc = (showOverlay && videoUrl) ? videoUrl : (localVideoUrl || '');

    // --- Renders ---

    // Skip internal upload UI if parent already provided videoUrl
    if (uploadState === 'idle' && !videoUrl) {
        return (
            <div className="h-full flex items-center justify-center bg-white">
                <div className="max-w-2xl w-full text-center px-4 md:px-8">
                    <div className="flex flex-col items-center gap-6">
                        <Upload className="h-16 w-16 text-gray-400" />
                        <div>
                            <h2 className="text-xl font-semibold text-secondary mb-2">Upload Pickleball Video</h2>
                            <p className="text-gray-500">Upload match footage for stroke analysis</p>
                        </div>
                        <button onClick={() => inputRef.current?.click()} className="px-6 py-3 md:py-2.5 bg-primary hover:bg-primary-dark text-white rounded-lg font-medium transition touch-target">
                            Select Video File
                        </button>
                        <input ref={inputRef} type="file" accept="video/*" onChange={handleFileSelect} className="hidden" />
                    </div>
                </div>
            </div>
        );
    }

    if (uploadState === 'uploading') {
        return (
            <div className="h-full flex items-center justify-center bg-white">
                <div className="flex flex-col items-center gap-6">
                    <Loader2 className="h-16 w-16 text-primary animate-spin" />
                    <h2 className="text-xl font-semibold text-secondary">Uploading Video...</h2>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col pt-0 md:pt-4 items-center bg-white px-0 md:px-8 pb-0 md:pb-4 overflow-y-auto w-full">
            <div className="max-w-5xl w-full">

                {/* VIDEO CONTAINER */}
                <div
                    className="relative w-full aspect-video bg-black rounded-none md:rounded-2xl overflow-hidden shadow-none md:shadow-2xl group border-0 md:border-2 border-gray-200"
                    onMouseEnter={() => setShowControls(true)}
                    onMouseLeave={() => isPlaying && setShowControls(false)}
                >

                    {/* PROCESSING OVERLAY */}
                    {isProcessing && (
                        <div className="absolute inset-0 z-50 bg-black/80 flex flex-col items-center justify-center gap-4 animate-in fade-in duration-300">
                            <Loader2 className="h-12 w-12 text-primary animate-spin" />
                            <div className="text-center">
                                <h3 className="text-lg font-semibold text-white">Analyzing Stroke...</h3>
                                <p className="text-gray-300 text-sm">Tracking player and analyzing technique</p>
                            </div>
                        </div>
                    )}

                    {/* User Guidance Overlay - Only show if NO analysis data (pre-analysis) */}
                    {!isProcessing && !analysisData && (
                        <div className="absolute top-4 left-0 right-0 z-40 flex justify-center pointer-events-none">
                            {!taggedPlayer && players.length === 0 ? (
                                <div className="bg-black/70 backdrop-blur-md text-white px-3 md:px-4 py-2 rounded-full text-xs md:text-sm font-medium border border-white/20 shadow-lg max-w-[90%]">
                                    {isTagMode ? (
                                        <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" /> <span className="hidden sm:inline">Draw a box around the player to track</span><span className="sm:hidden">Draw box around player</span></span>
                                    ) : (
                                        <span className="flex items-center gap-2"><Tag className="h-4 w-4 text-primary" /> <span className="hidden sm:inline">Click "Tag Player" to select your target</span><span className="sm:hidden">Tag player to start</span></span>
                                    )}
                                </div>
                            ) : (
                                <div className="bg-accent/90 backdrop-blur-md text-white px-3 md:px-4 py-2 rounded-full text-xs md:text-sm font-medium border border-white/20 shadow-lg animate-pulse max-w-[90%]">
                                    <span className="flex items-center gap-2"><Zap className="h-4 w-4 fill-white" /> <span className="hidden sm:inline">Target Locked! Click "Start Analysis" below</span><span className="sm:hidden">Ready to analyze</span></span>
                                </div>
                            )}
                        </div>
                    )}

                    {/* CONTENT (Image or Video) */}

                    {/* SHOW LOADING IF EXTRACTING FRAME */}
                    {isExtractingFrame ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900 text-white z-30">
                            <Loader2 className="h-10 w-10 animate-spin text-primary mb-3" />
                            <p className="text-sm font-medium">Extracting frame for tagging...</p>
                        </div>
                    ) : (isTagMode || (!activeSrc && firstFrameUrl)) && firstFrameUrl ? (
                        <div className="relative w-full h-full" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
                            <img src={firstFrameUrl} className="w-full h-full object-contain pointer-events-none select-none" draggable={false} />
                            {selectionBox && (
                                <div
                                    className="absolute border-4 border-dashed border-primary bg-primary/10 rounded-lg animate-pulse"
                                    style={{
                                        left: `${selectionBox.x * 100}%`,
                                        top: `${selectionBox.y * 100}%`,
                                        width: `${selectionBox.w * 100}%`,
                                        height: `${selectionBox.h * 100}%`,
                                        boxShadow: '0 0 0 2px rgba(0,191,165,0.3)'
                                    }}
                                >
                                    <div className="absolute -top-8 left-0 bg-primary text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">
                                        Target Player
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : activeSrc ? (
                        <div className="relative w-full h-full flex items-center justify-center" onClick={togglePlay}>
                            <video
                                ref={videoRef}
                                key={activeSrc}
                                src={activeSrc}
                                className={`${sideBySide ? 'w-1/2' : 'w-full'} h-full object-contain`}
                                playsInline
                                controls={false}
                                muted={isMuted}
                                onTimeUpdate={() => {
                                    if (videoRef.current && onTimeUpdate) onTimeUpdate(videoRef.current.currentTime);
                                }}
                                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                                onPlay={() => {
                                    setIsPlaying(true);
                                    if (secondaryVideoRef.current) secondaryVideoRef.current.play();
                                }}
                                onPause={() => {
                                    setIsPlaying(false);
                                    if (secondaryVideoRef.current) secondaryVideoRef.current.pause();
                                }}
                                onEnded={() => setIsPlaying(false)}
                            />
                            {sideBySide && localVideoUrl && (
                                <video
                                    ref={secondaryVideoRef}
                                    src={localVideoUrl}
                                    className="w-1/2 h-full object-contain border-l border-white/20"
                                    muted={true}
                                    controls={false}
                                    playsInline
                                />
                            )}
                            {/* Play Overlay Button (Big) */}
                            {!isPlaying && !isProcessing && (
                                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                    <div className="w-16 h-16 bg-black/50 rounded-full flex items-center justify-center backdrop-blur-sm">
                                        <Play className="h-8 w-8 text-white fill-white ml-1" />
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        // FIXED: Fallback for when no video source but we have analysis data
                        <div className="relative w-full h-full flex items-center justify-center bg-gray-900">
                            {analysisData && analysisData.frames && analysisData.frames.length > 0 ? (
                                <div className="text-center text-white p-8">
                                    <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
                                    <h3 className="text-xl font-semibold mb-2">Analysis Complete!</h3>
                                    <p className="text-gray-300 mb-4">
                                        {analysisData.frames.length} frames analyzed • {analysisData.strokes?.length || 0} strokes detected
                                    </p>
                                    <p className="text-sm text-gray-400">
                                        Video source not available, but analysis data is ready for review below.
                                    </p>
                                </div>
                            ) : (
                                <div className="text-center text-white p-8">
                                    <AlertTriangle className="h-16 w-16 text-yellow-500 mx-auto mb-4" />
                                    <h3 className="text-xl font-semibold mb-2">Video Source Missing</h3>
                                    <p className="text-gray-300 mb-4">
                                        Unable to load video for playback
                                    </p>
                                    <p className="text-sm text-gray-400">
                                        Please try uploading the video again or check the file path.
                                    </p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* CUSTOM CONTROLS OVERLAY - ONLY SHOW IF NOT PROCESSING & VIDEO LOADED */}
                    {(videoFile || videoUrl) && !isProcessing && !isTagMode && (
                        <div className={`absolute bottom-0 left-0 right-0 p-2 md:p-4 bg-gradient-to-t from-black/90 via-black/60 to-transparent transition-opacity duration-300 ${showControls ? 'opacity-100' : 'opacity-0'}`}>

                            {/* Timeline Slider */}
                            <div className="relative w-full h-1 md:h-1.5 bg-white/20 rounded-full mb-2 md:mb-4 cursor-pointer group/timeline hover:h-2 transition-all">
                                {/* Buffered/Background */}
                                <div className="absolute inset-0 rounded-full overflow-hidden">
                                    {/* Stroke markers if available; fallback to frame markers */}
                                    {analysisData?.strokes && analysisData.strokes.length > 0 ? (
                                        analysisData.strokes.map((s, idx) => {
                                            const startPct = duration ? (s.startSec / duration) * 100 : 0;
                                            const endPct = duration ? (s.endSec / duration) * 100 : startPct + 0.2;
                                            const color = s.type === 'serve' ? 'bg-blue-400' : s.type === 'groundstroke' ? 'bg-green-400' : s.type === 'dink' ? 'bg-yellow-400' : s.type === 'overhead' ? 'bg-red-400' : 'bg-purple-400';
                                            return (
                                                <div
                                                    key={idx}
                                                    className={`absolute top-0 bottom-0 ${color} opacity-60 cursor-pointer`}
                                                    style={{ left: `${startPct}%`, width: `${Math.max(0.5, endPct - startPct)}%` }}
                                                    title={`${s.type} (${Math.round(s.confidence * 100)}%)`}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        if (videoRef.current) {
                                                            videoRef.current.currentTime = s.startSec;
                                                            videoRef.current.pause();
                                                            setIsPlaying(false);
                                                        }
                                                    }}
                                                />
                                            );
                                        })
                                    ) : (
                                        analysisData?.frames?.map((frame, i) => {
                                            if (!duration) return null;
                                            const left = (frame.timestampSec / duration) * 100;
                                            return (
                                                <div
                                                    key={i}
                                                    className="absolute top-0 bottom-0 bg-accent/50 w-0.5 transform -translate-x-1/2"
                                                    style={{ left: `${left}%` }}
                                                />
                                            );
                                        })
                                    )}


                                </div>

                                <div className="absolute top-0 left-0 h-full bg-accent rounded-full" style={{ width: `${(currentTime || 0) / duration * 100}%` }}></div>
                                <input
                                    type="range"
                                    min="0" max={duration || 100} step="0.01"
                                    value={currentTime || 0}
                                    onChange={handleSeek}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />
                            </div>

                            <div className="flex items-center justify-between text-white">
                                <div className="flex items-center gap-2 md:gap-4">
                                    <button onClick={togglePlay} className="hover:text-accent transition">
                                        {isPlaying ? <Pause className="h-5 w-5 md:h-6 md:w-6 fill-current" /> : <Play className="h-5 w-5 md:h-6 md:w-6 fill-current" />}
                                    </button>

                                    <div className="hidden sm:flex items-center gap-1">
                                        <button onClick={() => stepFrame(-1)} className="p-1 hover:text-white text-white/70 transition" title="Previous Frame">
                                            <SkipBack className="h-4 w-4 fill-current" />
                                        </button>
                                        <button onClick={() => stepFrame(1)} className="p-1 hover:text-white text-white/70 transition" title="Next Frame">
                                            <SkipForward className="h-4 w-4 fill-current" />
                                        </button>
                                    </div>

                                    <span className="text-[10px] md:text-xs font-mono opacity-80">
                                        {formatTime(currentTime || 0)} / {formatTime(duration)}
                                    </span>
                                </div>

                                <div className="flex items-center gap-1 md:gap-3">
                                    <button onClick={onToggleOverlay} className={`p-1 md:p-1.5 rounded transition ${showOverlay ? 'text-accent bg-white/10' : 'text-gray-400'}`} title="Toggle Overlay">
                                        <Zap className="h-3.5 w-3.5 md:h-4 md:w-4" />
                                    </button>

                                    <div className="flex items-center bg-white/10 rounded-lg p-0.5 md:p-1">
                                        {[0.5, 1.0, 1.5].map(s => (
                                            <button key={s} onClick={() => onSpeedChange?.(s)} className={`px-1.5 md:px-2 py-0.5 text-[9px] md:text-[10px] rounded ${playbackSpeed === s ? 'bg-accent text-white' : 'text-gray-400 hover:text-white'}`}>
                                                {s}x
                                            </button>
                                        ))}
                                    </div>

                                    <button
                                        onClick={() => {
                                            if (!videoUrl) return;
                                            const a = document.createElement('a');
                                            a.href = videoUrl;
                                            a.download = `pickleball-analysis-${new Date().getTime()}.mp4`;
                                            document.body.appendChild(a);
                                            a.click();
                                            document.body.removeChild(a);
                                        }}
                                        className="hidden sm:block p-1.5 rounded transition text-gray-400 hover:text-white hover:bg-white/10"
                                        title="Download Video"
                                    >
                                        <Upload className="h-4 w-4 rotate-180" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* BOTTOM ACTION BAR (Tagging & Analysis) - Hide if analysis is done OR processing */}
                {
                    !analysisData && !isProcessing && (
                        <div className="mt-3 md:mt-4 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-white p-3 md:p-4 rounded-xl shadow-sm border border-gray-100">
                            <button
                                onClick={() => setIsTagMode(!isTagMode)}
                                disabled={isProcessing}
                                className={`flex items-center justify-center gap-2 px-4 py-3 md:py-2 rounded-lg text-sm font-medium transition touch-target ${isTagMode ? 'bg-secondary text-white' : 'bg-gray-100 text-secondary hover:bg-gray-200'}`}
                            >
                                <Tag className="h-5 w-5 md:h-4 md:w-4" />
                                {isTagMode ? 'Done Tagging' : 'Tag Player'}
                            </button>

                            {((taggedPlayer || players.length > 0)) && !isProcessing && (
                                <button
                                    onClick={onStartAnalysis}
                                    className="flex items-center justify-center gap-2 px-6 py-3 md:py-2 bg-accent hover:bg-orange-600 text-white rounded-lg text-sm font-medium transition shadow-lg shadow-accent/20 touch-target"
                                >
                                    <Zap className="h-5 w-5 md:h-4 md:w-4" />
                                    Start Analysis
                                </button>
                            )}
                        </div>
                    )}

                {/* Bottom spacer or existing closing tags */}
            </div>
        </div>
    );
}
