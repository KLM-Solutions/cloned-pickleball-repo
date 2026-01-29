import { NextRequest, NextResponse } from 'next/server';

/**
 * API Route: Camera Angle Validation using Gemini
 * 
 * Receives 5 video frames as base64 images and stroke type.
 * Sends to Gemini for analysis and returns validation result.
 */

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { frames, strokeType } = body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return NextResponse.json(
                { error: 'No frames provided' },
                { status: 400 }
            );
        }

        const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
        if (!GEMINI_API_KEY) {
            console.error('GEMINI_API_KEY not configured');
            return NextResponse.json(
                { error: 'Gemini API not configured' },
                { status: 500 }
            );
        }

        // Build the prompt for Gemini
        const strokeThresholds: Record<string, number> = {
            serve: 70,
            overhead: 70,
            groundstroke: 65,
            dink: 75,
            footwork: 60,
            overall: 65,
        };

        const threshold = strokeThresholds[strokeType] || 65;

        const prompt = `You are a pickleball coach analyzing camera setup for video analysis.

Analyze these ${frames.length} frames from a video recording of a "${strokeType}" stroke practice session.
Evaluate if the camera angle is suitable for biomechanical analysis.

SCORING CRITERIA (rate each 0-100):
1. **Camera Angle** (ideal: 90° side view for serves/overheads, 45° rear for groundstrokes)
   - Is the camera positioned at the correct angle to see the full swing?
   - Can you see the player's body from a useful perspective?

2. **Framing** (ideal: player centered, full body visible)
   - Is the player fully visible in the frame?
   - Is there enough space around the player?
   - Is the player reasonably centered?

3. **Distance** (ideal: 8-15 feet)
   - Is the player too small or too large in frame?
   - Can you see details of the swing motion?

4. **Stability** (check across frames)
   - Does the camera appear stable?
   - Is there excessive shake or movement?

5. **Lighting**
   - Is the video bright enough?
   - Can you clearly see the player and their movements?

RESPOND IN THIS EXACT JSON FORMAT:
{
  "score": <overall score 0-100>,
  "passed": <true if score >= ${threshold}>,
  "issues": [
    {
      "type": "<angle|framing|distance|stability|visibility>",
      "severity": "<high|medium|low>",
      "message": "<brief description of issue>"
    }
  ],
  "suggestion": "<most important corrective action, e.g., 'Move 2 steps to your left for a better side view'>",
  "analysis": "<1-2 sentence summary of camera setup quality>"
}

Be specific with suggestions - give actionable advice like "Move X steps in Y direction" or "Raise camera to chest height".
If the setup looks good, respond with high score and empty issues array.`;

        // Format frames for Gemini API
        const imageParts = frames.map((frame: string) => ({
            inlineData: {
                mimeType: 'image/jpeg',
                data: frame.replace(/^data:image\/\w+;base64,/, ''),
            },
        }));

        // Call Gemini API
        const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;

        const geminiResponse = await fetch(geminiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                contents: [
                    {
                        parts: [
                            { text: prompt },
                            ...imageParts,
                        ],
                    },
                ],
                generationConfig: {
                    temperature: 0.2,
                    maxOutputTokens: 1024,
                },
            }),
        });

        if (!geminiResponse.ok) {
            const errorText = await geminiResponse.text();
            console.error('Gemini API error:', errorText);
            return NextResponse.json(
                { error: 'Gemini API request failed' },
                { status: 500 }
            );
        }

        const geminiResult = await geminiResponse.json();

        // Extract the response text
        const responseText = geminiResult.candidates?.[0]?.content?.parts?.[0]?.text;

        // Log the LLM response for debugging
        console.log('=== GEMINI CAMERA VALIDATION RESPONSE ===');
        console.log('Stroke Type:', strokeType);
        console.log('Frames Analyzed:', frames.length);
        console.log('Raw Response:', responseText);
        console.log('==========================================');

        if (!responseText) {
            console.error('No response from Gemini');
            return NextResponse.json(
                { error: 'No response from Gemini' },
                { status: 500 }
            );
        }

        // Parse JSON from response (handle markdown code blocks)
        let jsonStr = responseText;
        const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (jsonMatch) {
            jsonStr = jsonMatch[1];
        }

        try {
            const validationResult = JSON.parse(jsonStr.trim());

            // Ensure required fields exist
            return NextResponse.json({
                score: validationResult.score || 50,
                passed: validationResult.passed ?? (validationResult.score >= threshold),
                threshold,
                issues: validationResult.issues || [],
                suggestion: validationResult.suggestion || null,
                analysis: validationResult.analysis || null,
                framesSampled: frames.length,
            });
        } catch (parseError) {
            console.error('Failed to parse Gemini response:', responseText);
            // Return a fallback response
            return NextResponse.json({
                score: 60,
                passed: false,
                threshold,
                issues: [{ type: 'visibility', severity: 'low', message: 'Could not fully analyze video' }],
                suggestion: 'Please ensure good lighting and a clear view of the player.',
                framesSampled: frames.length,
            });
        }

    } catch (error) {
        console.error('Validation API error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
