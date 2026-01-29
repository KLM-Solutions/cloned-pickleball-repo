/**
 * Clerk Webhook Endpoint
 * 
 * Receives user events from Clerk and syncs them to Supabase.
 * Configure this endpoint in your Clerk dashboard:
 * https://dashboard.clerk.com -> Webhooks -> Add Endpoint
 * 
 * Endpoint URL: https://your-domain.com/api/webhook/clerk
 * Events to subscribe: user.created, user.updated, user.deleted
 **/

import { NextResponse } from "next/server";
import { Webhook } from "svix";
import { headers } from "next/headers";
import { createOrUpdateUser, deleteUser } from "@/lib/supabase-users";

// Clerk webhook secret from environment
const CLERK_WEBHOOK_SECRET = process.env.CLERK_WEBHOOK_SECRET;

interface ClerkUserEvent {
    data: {
        id: string;
        email_addresses?: Array<{ email_address: string }>;
        first_name?: string | null;
        last_name?: string | null;
        profile_image_url?: string | null;
        image_url?: string | null;
    };
    type: "user.created" | "user.updated" | "user.deleted";
}

export async function POST(request: Request): Promise<NextResponse> {
    console.log("=== Clerk Webhook Received ===");

    // Verify webhook signature if secret is configured
    if (CLERK_WEBHOOK_SECRET) {
        try {
            const headerPayload = await headers();
            const svix_id = headerPayload.get("svix-id");
            const svix_timestamp = headerPayload.get("svix-timestamp");
            const svix_signature = headerPayload.get("svix-signature");

            if (!svix_id || !svix_timestamp || !svix_signature) {
                console.error("Missing Svix headers");
                return NextResponse.json({ error: "Missing headers" }, { status: 400 });
            }

            const body = await request.text();
            const wh = new Webhook(CLERK_WEBHOOK_SECRET);

            const evt = wh.verify(body, {
                "svix-id": svix_id,
                "svix-timestamp": svix_timestamp,
                "svix-signature": svix_signature,
            }) as ClerkUserEvent;

            return await handleWebhookEvent(evt);
        } catch (err: any) {
            console.error("Webhook verification failed:", err.message);
            return NextResponse.json({ error: "Verification failed" }, { status: 400 });
        }
    } else {
        // No secret configured - parse directly (not recommended for production)
        console.warn("CLERK_WEBHOOK_SECRET not configured - skipping verification");
        try {
            const body = await request.json();
            return await handleWebhookEvent(body as ClerkUserEvent);
        } catch (err: any) {
            console.error("Failed to parse webhook body:", err.message);
            return NextResponse.json({ error: "Invalid body" }, { status: 400 });
        }
    }
}

async function handleWebhookEvent(evt: ClerkUserEvent): Promise<NextResponse> {
    const { type, data } = evt;

    console.log("Event Type:", type);
    console.log("User ID:", data.id);

    try {
        switch (type) {
            case "user.created":
            case "user.updated": {
                const email = data.email_addresses?.[0]?.email_address || null;
                const profileImageUrl = data.profile_image_url || data.image_url || null;

                await createOrUpdateUser({
                    id: data.id,
                    email,
                    firstName: data.first_name,
                    lastName: data.last_name,
                    profileImageUrl,
                });

                console.log(`User ${type === "user.created" ? "created" : "updated"}: ${data.id}`);
                return NextResponse.json({
                    success: true,
                    action: type,
                    userId: data.id
                });
            }

            case "user.deleted": {
                await deleteUser(data.id);
                console.log(`User deleted: ${data.id}`);
                return NextResponse.json({
                    success: true,
                    action: "deleted",
                    userId: data.id
                });
            }

            default:
                console.log(`Unhandled event type: ${type}`);
                return NextResponse.json({ received: true, type });
        }
    } catch (error: any) {
        console.error("Webhook processing error:", error.message);
        return NextResponse.json(
            { error: "Failed to process webhook", details: error.message },
            { status: 500 }
        );
    }
}

// Health check
export async function GET(): Promise<NextResponse> {
    return NextResponse.json({
        status: "ok",
        endpoint: "Clerk Webhook",
        configured: !!CLERK_WEBHOOK_SECRET,
        timestamp: new Date().toISOString(),
    });
}
