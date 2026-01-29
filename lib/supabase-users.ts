/**
 * Supabase User Management
 * 
 * Handles user CRUD operations for Clerk webhook sync.
 * The users table stores Clerk user IDs as the primary key.
 */

import { createClient } from '@supabase/supabase-js';

// Use service key for server-side operations
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

export interface User {
    id: string;              // Clerk user_id (primary key)
    email: string | null;
    first_name: string | null;
    last_name: string | null;
    profile_image_url: string | null;
    created_at: string;
    updated_at: string;
}

export interface CreateUserData {
    id: string;
    email?: string | null;
    firstName?: string | null;
    lastName?: string | null;
    profileImageUrl?: string | null;
}

/**
 * Create or update a user in Supabase
 * Called by Clerk webhook when user signs up or updates profile
 */
export async function createOrUpdateUser(userData: CreateUserData): Promise<User | null> {
    const { id, email, firstName, lastName, profileImageUrl } = userData;

    const { data, error } = await supabase
        .from('users')
        .upsert({
            id,
            email: email || null,
            first_name: firstName || null,
            last_name: lastName || null,
            profile_image_url: profileImageUrl || null,
            updated_at: new Date().toISOString(),
        }, {
            onConflict: 'id',
        })
        .select()
        .single();

    if (error) {
        console.error('Failed to upsert user:', error);
        throw new Error(`Failed to upsert user: ${error.message}`);
    }

    return data;
}

/**
 * Get user by Clerk ID
 */
export async function getUserById(userId: string): Promise<User | null> {
    const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', userId)
        .single();

    if (error) {
        if (error.code === 'PGRST116') {
            return null; // User not found
        }
        console.error('Failed to get user:', error);
        throw new Error(`Failed to get user: ${error.message}`);
    }

    return data;
}

/**
 * Delete user by Clerk ID
 * Called when user deletes their account
 * Also deletes related data (cascade delete)
 */
export async function deleteUser(userId: string): Promise<void> {
    // Delete user profile first
    const { error: profileError } = await supabase
        .from('user_profiles')
        .delete()
        .eq('user_id', userId);

    if (profileError) {
        console.error('Failed to delete user profile:', profileError);
        // Continue anyway - profile might not exist
    }

    // Delete all analysis jobs for this user
    const { error: jobsError } = await supabase
        .from('analysis_jobs')
        .delete()
        .eq('user_id', userId);

    if (jobsError) {
        console.error('Failed to delete user analysis jobs:', jobsError);
        // Continue anyway
    }

    // Finally delete the user
    const { error } = await supabase
        .from('users')
        .delete()
        .eq('id', userId);

    if (error) {
        console.error('Failed to delete user:', error);
        throw new Error(`Failed to delete user: ${error.message}`);
    }

    console.log(`Cascade deleted user ${userId}: profile, jobs, and user record`);
}

/**
 * Check if user exists
 */
export async function userExists(userId: string): Promise<boolean> {
    const { data, error } = await supabase
        .from('users')
        .select('id')
        .eq('id', userId)
        .single();

    if (error && error.code !== 'PGRST116') {
        console.error('Failed to check user:', error);
    }

    return !!data;
}
