"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft, User, Grid3x3, Play, Maximize2, ZoomIn, ZoomOut, RotateCcw, Upload, Info } from 'lucide-react';

function CropContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const strokeType = searchParams.get('stroke') || 'overall';

  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });

  const [cropBox, setCropBox] = useState<{ x: number, y: number, width: number, height: number } | null>(null);
  const [activeAction, setActiveAction] = useState<'drawing' | 'moving' | 'resizing' | null>(null);
  const [activeHandle, setActiveHandle] = useState<string | null>(null);
  const [drawStart, setDrawStart] = useState({ x: 0, y: 0 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const [showGrid, setShowGrid] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [showHelp, setShowHelp] = useState(true);
  const [videoLoading, setVideoLoading] = useState(true);
  const [videoError, setVideoError] = useState(false);
  const [videoReady, setVideoReady] = useState(false);

  // Refs for smooth animation (avoid re-renders during drag)
  const cropBoxRef = useRef(cropBox);
  const pendingUpdate = useRef<{ x: number, y: number, width: number, height: number } | null>(null);

  useEffect(() => {
    cropBoxRef.current = cropBox;
  }, [cropBox]);

  useEffect(() => {
    const storedUrl = sessionStorage.getItem('videoUrl');
    if (storedUrl) {
      setVideoUrl(storedUrl);
      setVideoLoading(true);
      setVideoError(false);
      setVideoReady(false);
    }
  }, []);

  // Hide help after 3 seconds
  useEffect(() => {
    const timer = setTimeout(() => setShowHelp(false), 4000);
    return () => clearTimeout(timer);
  }, []);

  // iOS-optimized video initialization
  useEffect(() => {
    if (!videoUrl || !videoRef.current) return;

    const video = videoRef.current;

    setVideoLoading(true);
    setVideoReady(false);
    setVideoError(false);

    const initVideo = async () => {
      try {
        video.load();

        await new Promise<void>((resolve, reject) => {
          const onLoadedMetadata = () => {
            video.removeEventListener('loadedmetadata', onLoadedMetadata);
            video.removeEventListener('error', onError);
            resolve();
          };
          const onError = () => {
            video.removeEventListener('loadedmetadata', onLoadedMetadata);
            video.removeEventListener('error', onError);
            reject(new Error('Video load failed'));
          };
          video.addEventListener('loadedmetadata', onLoadedMetadata);
          video.addEventListener('error', onError);

          setTimeout(() => {
            video.removeEventListener('loadedmetadata', onLoadedMetadata);
            video.removeEventListener('error', onError);
            reject(new Error('Video load timeout'));
          }, 10000);
        });

        setVideoDimensions({
          width: video.videoWidth,
          height: video.videoHeight
        });

        // iOS trick: seek to show first frame
        video.currentTime = 0.01;

        await new Promise<void>((resolve) => {
          const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
          };
          video.addEventListener('seeked', onSeeked);
          setTimeout(resolve, 500);
        });

        setVideoLoading(false);
        setVideoReady(true);

      } catch (error) {
        console.error('Video init error:', error);
        setVideoLoading(false);
        setVideoError(true);
      }
    };

    initVideo();
  }, [videoUrl]);

  const screenToNormalized = useCallback((clientX: number, clientY: number) => {
    if (!containerRef.current || !videoRef.current) return { x: 0, y: 0 };
    const containerRect = containerRef.current.getBoundingClientRect();
    const video = videoRef.current;
    const containerRatio = containerRect.width / containerRect.height;
    const videoRatio = video.videoWidth / video.videoHeight;

    let displayWidth = containerRect.width;
    let displayHeight = containerRect.height;
    let offsetX = 0;
    let offsetY = 0;

    if (videoRatio > containerRatio) {
      displayHeight = containerRect.width / videoRatio;
      offsetY = (containerRect.height - displayHeight) / 2;
    } else {
      displayWidth = containerRect.height * videoRatio;
      offsetX = (containerRect.width - displayWidth) / 2;
    }

    const x = Math.max(0, Math.min(100, ((clientX - (containerRect.left + offsetX)) / displayWidth) * 100));
    const y = Math.max(0, Math.min(100, ((clientY - (containerRect.top + offsetY)) / displayHeight) * 100));
    return { x, y };
  }, []);

  const getEventCoords = (e: React.MouseEvent | React.TouchEvent | MouseEvent | TouchEvent) => {
    if ('touches' in e && e.touches.length > 0) {
      return { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY };
    }
    if ('changedTouches' in e && e.changedTouches.length > 0) {
      return { clientX: e.changedTouches[0].clientX, clientY: e.changedTouches[0].clientY };
    }
    if ('clientX' in e) {
      return { clientX: e.clientX, clientY: e.clientY };
    }
    return { clientX: 0, clientY: 0 };
  };

  const handleInteractionStart = (e: React.MouseEvent | React.TouchEvent, action: 'drawing' | 'moving' | 'resizing' = 'drawing', handle: string | null = null) => {
    e.preventDefault();
    e.stopPropagation();

    const { clientX, clientY } = getEventCoords(e);
    const { x, y } = screenToNormalized(clientX, clientY);

    setActiveAction(action);
    setActiveHandle(handle);
    setShowHelp(false);

    if (action === 'drawing') {
      setDrawStart({ x, y });
      setCropBox({ x, y, width: 0.1, height: 0.1 });
    } else if (action === 'moving' && cropBoxRef.current) {
      setOffset({ x: x - cropBoxRef.current.x, y: y - cropBoxRef.current.y });
    }
  };

  // Optimized move handler using requestAnimationFrame
  const handleInteractionMove = useCallback((e: MouseEvent | TouchEvent) => {
    if (!activeAction) return;

    // Prevent scrolling on iOS
    if ('touches' in e) {
      e.preventDefault();
    }

    const { clientX, clientY } = getEventCoords(e);
    const { x, y } = screenToNormalized(clientX, clientY);

    // Calculate new crop box position
    const prev = cropBoxRef.current;
    if (!prev) return;

    let newBox: { x: number, y: number, width: number, height: number } | null = null;

    if (activeAction === 'drawing') {
      newBox = {
        x: Math.min(x, drawStart.x),
        y: Math.min(y, drawStart.y),
        width: Math.max(2, Math.abs(x - drawStart.x)),
        height: Math.max(2, Math.abs(y - drawStart.y))
      };
    } else if (activeAction === 'moving') {
      newBox = {
        ...prev,
        x: Math.max(0, Math.min(100 - prev.width, x - offset.x)),
        y: Math.max(0, Math.min(100 - prev.height, y - offset.y))
      };
    } else if (activeAction === 'resizing' && activeHandle) {
      let newX = prev.x, newY = prev.y, newW = prev.width, newH = prev.height;
      if (activeHandle.includes('left')) {
        newX = Math.max(0, Math.min(prev.x + prev.width - 5, x));
        newW = prev.x + prev.width - newX;
      } else if (activeHandle.includes('right')) {
        newW = Math.max(5, Math.min(100 - prev.x, x - prev.x));
      }
      if (activeHandle.includes('top')) {
        newY = Math.max(0, Math.min(prev.y + prev.height - 5, y));
        newH = prev.y + prev.height - newY;
      } else if (activeHandle.includes('bottom')) {
        newH = Math.max(5, Math.min(100 - prev.y, y - prev.y));
      }
      newBox = { x: newX, y: newY, width: newW, height: newH };
    }

    if (newBox) {
      pendingUpdate.current = newBox;

      // Use requestAnimationFrame for smooth updates
      if (!animationRef.current) {
        animationRef.current = requestAnimationFrame(() => {
          if (pendingUpdate.current) {
            setCropBox(pendingUpdate.current);
            pendingUpdate.current = null;
          }
          animationRef.current = null;
        });
      }
    }
  }, [activeAction, activeHandle, drawStart, offset, screenToNormalized]);

  const handleInteractionEnd = useCallback(() => {
    setActiveAction(null);
    setActiveHandle(null);

    // Cancel any pending animation frame
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    // Apply any pending update
    if (pendingUpdate.current) {
      setCropBox(pendingUpdate.current);
      pendingUpdate.current = null;
    }

    setCropBox(prev => (prev && (prev.width < 3 || prev.height < 3)) ? null : prev);
  }, []);

  useEffect(() => {
    if (activeAction) {
      const options = { passive: false } as AddEventListenerOptions;
      window.addEventListener('mousemove', handleInteractionMove);
      window.addEventListener('mouseup', handleInteractionEnd);
      window.addEventListener('touchmove', handleInteractionMove, options);
      window.addEventListener('touchend', handleInteractionEnd);
      window.addEventListener('touchcancel', handleInteractionEnd);

      return () => {
        window.removeEventListener('mousemove', handleInteractionMove);
        window.removeEventListener('mouseup', handleInteractionEnd);
        window.removeEventListener('touchmove', handleInteractionMove);
        window.removeEventListener('touchend', handleInteractionEnd);
        window.removeEventListener('touchcancel', handleInteractionEnd);
      };
    }
  }, [activeAction, handleInteractionMove, handleInteractionEnd]);

  const applyPreset = (preset: 'full' | 'center' | 'left' | 'right') => {
    switch (preset) {
      case 'full': setCropBox({ x: 5, y: 5, width: 90, height: 90 }); break;
      case 'center': setCropBox({ x: 25, y: 15, width: 50, height: 70 }); break;
      case 'left': setCropBox({ x: 15, y: 15, width: 35, height: 70 }); break;
      case 'right': setCropBox({ x: 50, y: 15, width: 35, height: 70 }); break;
    }
    setShowHelp(false);
  };

  const resetCrop = () => { setCropBox(null); setZoom(1); };

  const handleStartAnalysis = () => {
    if (!cropBox || !containerRef.current || !videoRef.current) return;

    // CRITICAL FIX: Convert container-relative % to video-relative %
    // The cropBox values are in container % (including letterbox areas)
    // We need to transform them to video content % for the backend

    const containerRect = containerRef.current.getBoundingClientRect();
    const video = videoRef.current;
    const containerRatio = containerRect.width / containerRect.height;
    const videoRatio = video.videoWidth / video.videoHeight;

    let displayWidth = containerRect.width;
    let displayHeight = containerRect.height;
    let offsetXPercent = 0;  // Letterbox offset as % of container
    let offsetYPercent = 0;

    if (videoRatio > containerRatio) {
      // Video is wider: letterbox on top/bottom
      displayHeight = containerRect.width / videoRatio;
      offsetYPercent = ((containerRect.height - displayHeight) / 2) / containerRect.height * 100;
    } else {
      // Video is taller: letterbox on left/right
      displayWidth = containerRect.height * videoRatio;
      offsetXPercent = ((containerRect.width - displayWidth) / 2) / containerRect.width * 100;
    }

    // Video display area as % of container
    const videoWidthPercent = (displayWidth / containerRect.width) * 100;
    const videoHeightPercent = (displayHeight / containerRect.height) * 100;

    // Transform crop coordinates from container % to video %
    // Formula: video_coord = (container_coord - offset) / video_size * 100
    const x1 = Math.max(0, Math.min(1, (cropBox.x - offsetXPercent) / videoWidthPercent));
    const y1 = Math.max(0, Math.min(1, (cropBox.y - offsetYPercent) / videoHeightPercent));
    const x2 = Math.max(0, Math.min(1, (cropBox.x + cropBox.width - offsetXPercent) / videoWidthPercent));
    const y2 = Math.max(0, Math.min(1, (cropBox.y + cropBox.height - offsetYPercent) / videoHeightPercent));

    console.log('[CROP DEBUG] Container offset:', { offsetXPercent, offsetYPercent });
    console.log('[CROP DEBUG] Video display %:', { videoWidthPercent, videoHeightPercent });
    console.log('[CROP DEBUG] CropBox (container %):', cropBox);
    console.log('[CROP DEBUG] Final coords (video %):', { x1, y1, x2, y2 });

    const coords = { x1, y1, x2, y2 };
    sessionStorage.setItem('cropCoords', JSON.stringify(coords));

    router.push(`/strikesense/processing?stroke=${strokeType}`);
  };

  return (
    <div className="h-screen flex flex-col bg-white text-neutral-900 overflow-hidden select-none">
      {/* Header */}
      <header className="relative z-50 h-14 md:h-16 flex items-center justify-between px-3 md:px-4 border-b border-neutral-200 flex-shrink-0">
        <button
          onClick={() => router.push(`/strikesense/upload?stroke=${strokeType}`)}
          className="flex items-center gap-1.5 md:gap-2 text-neutral-500 hover:text-black transition p-1"
        >
          <ArrowLeft className="w-4 h-4 md:w-5 md:h-5" />
          <span className="text-xs md:text-sm font-medium">Back</span>
        </button>
        <div className="flex items-center gap-1.5 md:gap-2">
          <User className="w-3.5 h-3.5 md:w-4 md:h-4 text-black" />
          <span className="text-[10px] md:text-xs font-bold tracking-widest uppercase text-neutral-500">Select Player</span>
        </div>
        <button
          onClick={() => setShowGrid(!showGrid)}
          className={`p-1.5 md:p-2 rounded-lg transition-all ${showGrid ? 'bg-black text-white' : 'text-neutral-500 hover:text-black bg-neutral-100'}`}
        >
          <Grid3x3 className="w-4 h-4 md:w-5 md:h-5" />
        </button>
      </header>

      <div className="flex-1 flex flex-col overflow-hidden relative">
        {/* Help Tooltip */}
        {showHelp && videoUrl && !cropBox && videoReady && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 bg-black text-white px-4 py-2 rounded-xl text-xs md:text-sm font-medium flex items-center gap-2 shadow-lg animate-bounce">
            <Info className="w-4 h-4" />
            Draw a box around the player
          </div>
        )}

        {/* Workspace Area */}
        <div className="flex-1 relative overflow-hidden flex items-center justify-center p-2 md:p-4 min-h-0">
          <div
            ref={containerRef}
            className="relative w-full h-full min-h-[200px] flex items-center justify-center bg-neutral-100 rounded-xl md:rounded-2xl overflow-hidden border border-neutral-200 touch-none"
            onMouseDown={(e) => !activeAction && handleInteractionStart(e, 'drawing')}
            onTouchStart={(e) => !activeAction && handleInteractionStart(e, 'drawing')}
            style={{
              transform: `scale3d(${zoom}, ${zoom}, 1)`,
              transition: activeAction ? 'none' : 'transform 0.2s ease-out',
              cursor: cropBox ? 'default' : 'crosshair',
              willChange: activeAction ? 'transform' : 'auto',
              WebkitTapHighlightColor: 'transparent',
            }}
          >

            {videoUrl ? (
              <>
                {/* Loading Spinner */}
                {videoLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-white/90 z-20">
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-10 h-10 border-4 border-neutral-300 border-t-black rounded-full animate-spin" />
                      <p className="text-xs text-neutral-500">Preparing your video studio...</p>
                    </div>
                  </div>
                )}

                {/* Error State */}
                {videoError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-white/90 z-20">
                    <div className="text-center p-4">
                      <p className="text-sm text-neutral-600 mb-2">We couldn't load that video properly.</p>
                      <button
                        onClick={() => {
                          setVideoLoading(true);
                          setVideoError(false);
                          setVideoReady(false);
                          if (videoRef.current) {
                            videoRef.current.load();
                          }
                        }}
                        className="text-xs text-black hover:underline"
                      >
                        Let's try that again
                      </button>
                    </div>
                  </div>
                )}

                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full h-full object-contain pointer-events-none"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    opacity: videoReady ? 1 : 0,
                    transition: 'opacity 0.3s ease',
                    WebkitTransform: 'translateZ(0)', // Force GPU acceleration on iOS
                  }}
                  preload="auto"
                  playsInline
                  // @ts-ignore - iOS specific attribute
                  webkit-playsinline="true"
                  muted
                  autoPlay={false}
                  controls={false}
                />
              </>
            ) : (
              <div className="text-center p-6 md:p-8 bg-neutral-50 border border-neutral-200 rounded-xl md:rounded-2xl">
                <div className="w-12 h-12 md:w-16 md:h-16 mx-auto mb-3 md:mb-4 rounded-full bg-neutral-200 flex items-center justify-center">
                  <Upload className="w-6 h-6 md:w-8 md:h-8 text-neutral-500" />
                </div>
                <p className="text-xs md:text-sm font-bold text-black uppercase tracking-tight mb-1.5 md:mb-2">No Video Loaded</p>
                <button
                  onClick={() => router.push(`/strikesense/upload?stroke=${strokeType}`)}
                  className="text-xs md:text-sm text-black hover:underline"
                >
                  Upload a video to get started
                </button>
              </div>
            )}

            {videoUrl && videoReady && cropBox && (
              <div
                className="absolute inset-0 pointer-events-none"
                style={{ willChange: activeAction ? 'contents' : 'auto' }}
              >
                {/* Dimmed overlay - using CSS mask for better performance */}
                <div
                  className="absolute inset-0 bg-black/50"
                  style={{
                    clipPath: `polygon(0% 0%, 0% 100%, 100% 100%, 100% 0%, 0% 0%, ${cropBox.x}% ${cropBox.y}%, ${cropBox.x}% ${cropBox.y + cropBox.height}%, ${cropBox.x + cropBox.width}% ${cropBox.y + cropBox.height}%, ${cropBox.x + cropBox.width}% ${cropBox.y}%, ${cropBox.x}% ${cropBox.y}%)`,
                  }}
                />

                {/* Crop box */}
                <div
                  className="absolute border-2 border-black pointer-events-auto bg-transparent shadow-lg"
                  onMouseDown={(e) => handleInteractionStart(e, 'moving')}
                  onTouchStart={(e) => handleInteractionStart(e, 'moving')}
                  style={{
                    left: `${cropBox.x}%`,
                    top: `${cropBox.y}%`,
                    width: `${cropBox.width}%`,
                    height: `${cropBox.height}%`,
                    cursor: 'move',
                    transform: 'translateZ(0)', // Force GPU layer
                    willChange: activeAction === 'moving' ? 'left, top' : 'auto',
                  }}
                >

                  {/* Resizing Handles - Larger touch targets for mobile */}
                  {[
                    { id: 'top-left', class: 'top-0 left-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize' },
                    { id: 'top-right', class: 'top-0 right-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize' },
                    { id: 'bottom-left', class: 'bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize' },
                    { id: 'bottom-right', class: 'bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize' },
                    { id: 'top', class: 'top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 cursor-ns-resize' },
                    { id: 'bottom', class: 'bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 cursor-ns-resize' },
                    { id: 'left', class: 'top-1/2 left-0 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize' },
                    { id: 'right', class: 'top-1/2 right-0 translate-x-1/2 -translate-y-1/2 cursor-ew-resize' },
                  ].map(h => (
                    <div
                      key={h.id}
                      className={`absolute w-12 h-12 md:w-10 md:h-10 flex items-center justify-center z-10 ${h.class}`}
                      onMouseDown={(e) => handleInteractionStart(e, 'resizing', h.id)}
                      onTouchStart={(e) => handleInteractionStart(e, 'resizing', h.id)}
                      style={{ WebkitTapHighlightColor: 'transparent' }}
                    >
                      <div className="w-5 h-5 md:w-4 md:h-4 bg-white border-2 border-black rounded-full shadow-lg" />
                    </div>
                  ))}

                  {/* Grid Overlay */}
                  {showGrid && (
                    <div className="absolute inset-0 pointer-events-none opacity-30">
                      <div className="absolute left-1/3 top-0 bottom-0 w-px bg-black" />
                      <div className="absolute left-2/3 top-0 bottom-0 w-px bg-black" />
                      <div className="absolute top-1/3 left-0 right-0 h-px bg-black" />
                      <div className="absolute top-2/3 left-0 right-0 h-px bg-black" />
                    </div>
                  )}

                  {/* Label */}
                  <div className="absolute -top-9 md:-top-10 left-0 bg-black text-white px-3 py-1.5 rounded-lg text-[10px] md:text-xs font-bold uppercase tracking-widest shadow-lg flex items-center gap-1.5">
                    <User className="w-3 h-3" />
                    This is Me
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Floating Zoom Controls */}
          <div className="absolute bottom-4 md:bottom-6 right-4 md:right-6 flex flex-col gap-2 z-20">
            <button
              onClick={() => setZoom(Math.min(2, zoom + 0.25))}
              className="w-11 h-11 md:w-10 md:h-10 bg-white border border-neutral-200 text-black rounded-xl flex items-center justify-center hover:bg-neutral-100 active:scale-95 transition-all shadow-sm"
              style={{ WebkitTapHighlightColor: 'transparent' }}
            >
              <ZoomIn className="w-5 h-5" />
            </button>
            <button
              onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
              className="w-11 h-11 md:w-10 md:h-10 bg-white border border-neutral-200 text-black rounded-xl flex items-center justify-center hover:bg-neutral-100 active:scale-95 transition-all shadow-sm"
              style={{ WebkitTapHighlightColor: 'transparent' }}
            >
              <ZoomOut className="w-5 h-5" />
            </button>
            <button
              onClick={resetCrop}
              className="w-11 h-11 md:w-10 md:h-10 bg-white border border-neutral-200 text-black rounded-xl flex items-center justify-center hover:bg-neutral-100 active:scale-95 transition-all shadow-sm"
              style={{ WebkitTapHighlightColor: 'transparent' }}
            >
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Bottom Control Panel */}
        <div className="bg-neutral-50 border-t border-neutral-200 px-3 md:px-4 py-4 md:py-5 flex-shrink-0 z-50">
          <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center gap-3 md:gap-4">

            {/* Quick Presets */}
            <div className="flex-1 w-full">
              <div className="flex items-center justify-between mb-2 md:mb-3 px-1">
                <span className="text-[9px] md:text-[10px] uppercase font-bold tracking-widest text-neutral-400">Quick Zones</span>
                {cropBox && <span className="text-[9px] md:text-[10px] font-bold text-black">{Math.round(cropBox.width)}% x {Math.round(cropBox.height)}%</span>}
              </div>
              <div className="grid grid-cols-4 gap-2">
                {[
                  { id: 'left', label: 'Left', icon: <div className="w-1.5 h-3 border-l-2 border-current" /> },
                  { id: 'center', label: 'Center', icon: <div className="w-3 h-3 border-x-2 border-current" /> },
                  { id: 'right', label: 'Right', icon: <div className="w-1.5 h-3 border-r-2 border-current" /> },
                  { id: 'full', label: 'Full', icon: <Maximize2 className="w-3 h-3" /> }
                ].map(p => (
                  <button
                    key={p.id}
                    onClick={() => applyPreset(p.id as any)}
                    className="bg-white border border-neutral-200 hover:border-black hover:bg-neutral-100 text-black py-3.5 rounded-xl transition-all flex flex-col items-center gap-1 active:scale-95"
                    style={{ WebkitTapHighlightColor: 'transparent' }}
                  >
                    <div className="text-black">{p.icon}</div>
                    <span className="text-[9px] md:text-[10px] font-bold uppercase tracking-tight text-neutral-500">{p.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="hidden md:block w-px h-16 bg-neutral-200 mx-2" />

            {/* CTA */}
            <div className="flex flex-col gap-2 w-full md:w-64">
              <button
                onClick={handleStartAnalysis}
                disabled={!videoUrl || !cropBox || !videoReady}
                className="w-full bg-black hover:bg-neutral-800 disabled:opacity-30 disabled:cursor-not-allowed text-white py-4 rounded-xl font-bold text-sm uppercase tracking-widest active:scale-[0.98] transition-all flex items-center justify-center gap-2"
                style={{ WebkitTapHighlightColor: 'transparent' }}
              >
                <Play className="w-4 h-4 fill-white" />
                Looks Good, Analyze!
              </button>
              {!cropBox && videoReady && (
                <p className="text-[10px] text-center text-neutral-400 font-medium">
                  Draw a box around yourself to start
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CropPage() {
  return (
    <React.Suspense fallback={
      <div className="h-screen bg-white flex items-center justify-center">
        <div className="text-black font-bold animate-pulse">Loading...</div>
      </div>
    }>
      <CropContent />
    </React.Suspense>
  );
}