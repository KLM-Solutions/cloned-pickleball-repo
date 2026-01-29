import { NextResponse } from 'next/server';
import OpenAI from 'openai';
import { auth } from '@clerk/nextjs/server';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
    try {
        const { userId } = await auth();
        if (!userId) {
            return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
        }

        const data = await req.json();

        // Construct a focused prompt
        const systemPrompt = `You are a professional pickleball coach specializing in biomechanics.
        Analyze the provided player data (Skill Score, Injury Risks, Trends, Stroke Metrics).
        
        Your goal is to provide encouragement and specific, actionable feedback.
        
        Return a JSON object with this exact structure:
        {
            "summary": "2 sentences summarizing their current form and progress.",
            "takeaways": ["Point 1 (Positive)", "Point 2 (Critical Issue)", "Point 3 (Trend observation)"],
            "focus_drill": "Name of Drill: One sentence description of how to do it."
        }`;

        const response = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: JSON.stringify(data) }
            ],
            response_format: { type: "json_object" },
            temperature: 0.7,
        });

        const content = response.choices[0].message.content;
        if (!content) throw new Error("No content from OpenAI");

        const result = JSON.parse(content);
        return NextResponse.json(result);

    } catch (error: any) {
        console.error("AI Summary Error:", error);
        return NextResponse.json(
            { error: "Failed to generate insights", details: error.message },
            { status: 500 }
        );
    }
}
