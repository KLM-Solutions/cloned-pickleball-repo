/**
 * Camera Angle Validation System
 * 
 * Extracts 5 frames from first 10 seconds of video and sends to Gemini
 * for camera angle analysis. Returns validation result with score and feedback.
 * 
 * Thresholds per stroke type:
 * - serve/overhead: 70 (side view critical)
 * - groundstroke: 65 (more forgiving)
 * - dink: 75 (precise angle matters)
 * - footwork: 60 (full body visibility main concern)
 * - overall: 65 (general purpose)
 */

export type StrokeType = 'serve' | 'groundstroke' | 'dink' | 'overhead' | 'footwork' | 'overall';

export interface ValidationIssue {
    type: 'angle' | 'distance' | 'framing' | 'stability' | 'visibility';
    severity: 'high' | 'medium' | 'low';
    message: string;
    correction?: string;
}

export interface ValidationResult {
    score: number;           // 0-100
    passed: boolean;
    threshold: number;
    issues: ValidationIssue[];
    suggestion?: string;     // Primary corrective suggestion
    analysis?: string;       // Summary from Gemini
    framesSampled: number;
    analysisTimeMs: number;
}

// Thresholds per stroke type
const STROKE_THRESHOLDS: Record<StrokeType, number> = {
    serve: 70,
    overhead: 70,
    groundstroke: 65,
    dink: 75,
    footwork: 60,
    overall: 65,
};

/**
 * Extract frames from video at regular intervals and convert to base64
 */
async function extractFramesAsBase64(
    videoElement: HTMLVideoElement,
    numFrames: number = 5,
    maxDuration: number = 10
): Promise<string[]> {
    const frames: string[] = [];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) return frames;

    const duration = Math.min(videoElement.duration, maxDuration);
    const interval = duration / numFrames;

    // Use smaller resolution for faster upload
    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / videoElement.videoWidth);
    canvas.width = videoElement.videoWidth * scale;
    canvas.height = videoElement.videoHeight * scale;

    for (let i = 0; i < numFrames; i++) {
        const time = i * interval + 0.5; // Start 0.5s in

        await new Promise<void>((resolve) => {
            videoElement.currentTime = time;
            videoElement.onseeked = () => resolve();
        });

        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        // Convert to base64 JPEG with 70% quality for smaller size
        const base64 = canvas.toDataURL('image/jpeg', 0.7);
        frames.push(base64);
    }

    return frames;
}

/**
 * Main validation function
 * Extracts frames and sends to Gemini API for analysis
 */
export async function validateCameraAngle(
    videoFile: File | string,
    strokeType: StrokeType = 'groundstroke'
): Promise<ValidationResult> {
    const startTime = performance.now();
    const threshold = STROKE_THRESHOLDS[strokeType];

    return new Promise((resolve) => {
        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.crossOrigin = 'anonymous';

        video.onloadedmetadata = async () => {
            try {
                // Extract 5 frames from first 10 seconds
                const frames = await extractFramesAsBase64(video, 5, 10);

                if (frames.length === 0) {
                    resolve({
                        score: 0,
                        passed: false,
                        threshold,
                        issues: [{ type: 'visibility', severity: 'high', message: 'Could not extract frames from video' }],
                        suggestion: 'Please try uploading the video again',
                        framesSampled: 0,
                        analysisTimeMs: performance.now() - startTime,
                    });
                    return;
                }

                // Send frames to Gemini API for analysis
                const response = await fetch('/api/validate-camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        frames,
                        strokeType,
                    }),
                });

                if (!response.ok) {
                    throw new Error('API request failed');
                }

                const result = await response.json();

                resolve({
                    score: result.score || 50,
                    passed: result.passed ?? (result.score >= threshold),
                    threshold,
                    issues: result.issues || [],
                    suggestion: result.suggestion || null,
                    analysis: result.analysis || null,
                    framesSampled: result.framesSampled || frames.length,
                    analysisTimeMs: Math.round(performance.now() - startTime),
                });

            } catch (error) {
                console.error('Validation error:', error);
                // Fallback: allow user to continue with warning
                resolve({
                    score: 60,
                    passed: false,
                    threshold,
                    issues: [{ type: 'visibility', severity: 'low', message: 'Could not analyze video' }],
                    suggestion: 'Video analysis failed. You can continue anyway.',
                    framesSampled: 0,
                    analysisTimeMs: performance.now() - startTime,
                });
            }
        };

        video.onerror = () => {
            resolve({
                score: 0,
                passed: false,
                threshold,
                issues: [{ type: 'visibility', severity: 'high', message: 'Could not load video' }],
                suggestion: 'Please try a different video file',
                framesSampled: 0,
                analysisTimeMs: performance.now() - startTime,
            });
        };

        // Load video
        if (typeof videoFile === 'string') {
            video.src = videoFile;
        } else {
            video.src = URL.createObjectURL(videoFile);
        }

        video.load();
    });
}

/**
 * Get the threshold for a stroke type
 */
export function getThreshold(strokeType: StrokeType): number {
    return STROKE_THRESHOLDS[strokeType] || 65;
}
